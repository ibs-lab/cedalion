"""Implements the mSPoC algorithm for multimodal data."""

import warnings

import numpy as np
import xarray as xr

from cedalion import cite
from cedalion.sigdecomp.multimodal.utils import (
    validate_dimension_labels,
    validate_dimension_sizes,
    validate_time_coverage,
    validate_time_shifts,
    embed_lags,
    validate_l1_reg,
    validate_l2_reg,
    shrink_cov,
)
from cedalion.sigdecomp.multimodal.solvers import solve_filter


class mSPoC:
    """Multimodal Source Power Co-modulation algorithm based on :cite:t:`Dahne2013`.

    Given two vector-valued time series X(t), and Y(t), mSPoC finds component pairs
    Sx = Wx.T @ X, and Sy = Wy.T @ Y, such that the covariance between the
    temporally-embedded bandpower of Sx and the time course of Sy is maximized. The
    solution to that optimization problem is captured by the spatial (Wx, Wy), and
    temporal (Wt) filters.

    X(t) must be of shape Ntx x Nx, where Nx is the number of channels and Ntx the
    number of time points, and it is band-pass filtered in the frequency band of
    interest. Bandpower of Sx is then approximated by its variance within epochs, one
    per time point of Y(t): the X samples within a power_window centered at it (one
    sampling period of Y by default, while longer windows overlap and give a smoother
    bandpower). Epochs at the edges of X have fewer samples. X(t) must have at least
    twice the sampling rate of Y(t). Y(t) is of shape Nty x Ny. Both signals must span
    the same time domain, i.e. their first, and last, time points can differ by at
    most one sampling period of Y. Optionally, X and Y can have a trial dimension, in
    which case epochs and time lags are kept within trials, and X can be already
    temporally embedded, with a lag dimension.

    Args:
        N_components (int): Number of component pairs the algorithm will find.
        time_lags (list): List of time lags to consider in the temporal embedding.
            Either these or the lag_name dim of a pre-embedded X in fit must be given,
            not both. Duplicated lags are removed with a warning.
        N_restarts (int): Number of times the algorithm is repeated.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance value used for convergence criterion when comparing
            correlations of consecutive runs.
        random_state (int, optional): Seed for the random initialization of the
            filters and the "orthogonal" deflation bases, making fits reproducible.
            Defaults to None (not reproducible).
        l1_reg (float or list of floats): Relative L1 thresholds kappa_x and kappa_y of
            the wx and wy steps, each in [0, 1). If a single float is provided, then
            kappa_x = kappa_y. Defaults to 0 (no sparsity).
        l2_reg (float, str, or list): Shrinkage intensities gamma_x and gamma_y of the
            metrics Cxx and Cyy, each a number in [0, 1], or "lw" / "auto" to estimate
            it with Ledoit-Wolf. If a single value is provided, then gamma_x = gamma_y.
            Defaults to 0 (no shrinkage).
        n_units (int, optional): Number of units used by Ledoit-Wolf to estimate the
            variance of the covariance. Defaults to None, i.e. the number of trials if
            trials are given in fit, otherwise one unit per epoch.
        deflation (str): Deflation method for further components (see deflate),
            "orthogonal" (orthogonal filters, L1 sparsity only for the first
            component) or "schur" (removes the found sources from the data, keeping
            L1 sparsity for every component). Defaults to "orthogonal".
        power_window (float, optional): Length (in time units) of the epochs where the
            bandpower is estimated, at least one sampling period of Y. If None, 
            defaults to one sampling period of Y (non-overlapping epochs).

    Attributes:
        Wx (DataArray): Spatial filters of X, dims (featureX_name, mSPoC_X).
        Wy (DataArray): Spatial filters of Y, dims (featureY_name, mSPoC_Y).
        Wt (DataArray): Temporal filters over the lagged bandpower, dims (lag_name,
            mSPoC_T).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
        time_lags (ndarray): Validated time lags, sorted and without duplicates.
        N_lags (int): Number of time lags.
        lag_name (str): Name of the lag dim of Wt, i.e. lag_name in fit, or
            "time_shift" if X is embedded by the model.
        optimal_shift (ndarray): Time lag where |Wt| peaks, for each component. Only
            informative: Wt already accounts for the lags, so Sx and Sy are aligned.
        gamma_x, gamma_y (float): Shrinkage intensities used for Cxx and Cyy (0
            without L2 regularization).
    """

    def __init__(
        self,
        N_components: int = None,
        time_lags=None,
        N_restarts: int = 2,
        max_iter: int = 200,
        tol: float = 1e-5,
        random_state: int | None = None,
        l1_reg: float | list[float, float] = 0,
        l2_reg: float | str | list = 0,
        n_units: int | None = None,
        deflation: str = "orthogonal",
        power_window: float | None = None,
    ):
        # General parameters
        self.N_components = N_components
        self.N_restarts = N_restarts
        self.max_iter = max_iter
        self.tol = tol
        # Seed for the random initialization of the filters and the deflation bases
        self.random_state = random_state
        # Validate regularization parameters
        self.l1_reg = validate_l1_reg(l1_reg, relative=True)
        self.l2_reg = validate_l2_reg(l2_reg)
        self.n_units = n_units
        if deflation not in ("orthogonal", "schur"):
            raise ValueError(
                f"deflation must be 'orthogonal' or 'schur', but got {deflation!r}."
            )
        self.deflation = deflation
        # Epoch length for the bandpower (see epoch_mapping)
        self.power_window = power_window
        self.Ntx = None
        self.Nty = None
        self.Nx = None
        self.Ny = None
        self.N_features = None
        self.N_samples = None
        # (optional) trial dimension sizes
        self.Nkx = None
        self.Nky = None

        # Labels for sample and feature dimensions
        self.sample_name = None
        self.featureX_name = None
        self.featureY_name = None

        # Optional (trial) dimension label
        self.trial_name = None

        # Labels for latent space dimension
        self.latent_featureX_name = self.__class__.__name__ + "_X"
        self.latent_featureY_name = self.__class__.__name__ + "_Y"
        self.latent_featureT_name = self.__class__.__name__ + "_T"

        # Filters
        self.Wx = None
        self.Wy = None
        self.Wt = None

        # Spatial patterns (populated by build_spatial_patterns)
        self.Ax = None
        self.Ay = None

        # L2 shrinkage intensities (populated in fit)
        self.gamma_x = None
        self.gamma_y = None

        # Time lags given here, kept apart from the validated ones set in fit
        self.init_time_lags = time_lags
        self.time_lags = None
        self.N_lags = None
        self.optimal_shift = None

        # Name of the lag dim of Wt (and optional label of a pre-computed lag dim of X)
        self.lag_name = None

    def fit(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        featureX_name: str = "channel",
        featureY_name: str = "channel",
        trial_name: str | None = None,
        lag_name: str | None = None,
    ):
        """Train mSPoC model on datasets X and Y.

        Implement the pseudo-code of Algorithm 1 of :cite:t:`Dahne2013`, extended to 
        include also L1 and L2 (shrinkage) regularization. Filters are 
        The algorithm finds one component triplet at a time, following an alternating 
        approach (see one_unit_algorithm), and then finds further components by 
        via deflation. After training, the filter attributes Wx, Wy, and Wt are updated.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                ([lag_name], [trial_name], sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                ([trial_name], sample_name, featureY_name).
            featureX_name (str): Name of the feature dimension for X.
            featureY_name (str): Name of the feature dimension for Y.
            trial_name (str, optional): Label for trial dimension, defaults to None.
                Epochs and time lags are kept within trials, then trials are pooled.
            lag_name (str, optional): Label of the lag dimension if X is already
                temporally embedded. Its coords (in time units) are used as time
                lags, so time_lags must not be given at initialization. Defaults to
                None.
        """

        cite("Dahne2013")
        # Labels for sample and feature directions
        self.sample_name = "time"
        self.featureX_name = featureX_name
        self.featureY_name = featureY_name
        self.trial_name = trial_name

        # Validate inputs, ordered as (..., sample_name, feature_name)
        X, Y = self.validate_inputs_fit(X, Y, trial_name, lag_name)

        # Validated time lags, from time_lags or from the pre-computed lag dim
        self.set_time_lags(X, lag_name)

        # Time-embedded X (or the pre-computed lags), dims (lag, [trial], time, feature)
        X = embed_lags(X, self.time_lags, lag_name, self.lag_name)

        # Defaults power_window to one sampling period of Y, if not given
        dty = (Y.time[1] - Y.time[0]).item()
        self.power_window = self.power_window or dty

        # Drop the Y samples whose epochs overlap the zero padding of a raw X
        # (avoids artificial, biased, lower powers)
        if lag_name is None:
            Y = self.crop_padding(X, Y)

        # Run main algorithm for filter estimation
        Wx, Wy, Wt = self.estimate_filters(X, Y)

        # Convert back to xarray with the right dimensions and coordinates
        self.Wx, self.Wy, self.Wt = self.convert_filters_to_DataArray(
            Wx, Wy, Wt, X, Y
        )

        # Lag where |Wt| peaks (the sign of Wt carries the sign of the coupling)
        self.optimal_shift = self.time_lags[np.argmax(np.abs(Wt), axis=0)]

    def transform(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
        return_unfiltered: bool = False,
    ) -> tuple[xr.DataArray, ...]:
        """Get reconstructed sources of the X and Y dataset.

        The X component is constructed by computing the bandpower of the X projection
        along Wx in each epoch (see epoch_mapping), and then applying a linear temporal
        filter over its lagged copies using Wt. The Y component is constructed as the
        linear projection of Y along Wy. If X is not pre-embedded, the Y samples whose
        epochs overlap its zero padding are dropped (see crop_padding), so Sx and Sy
        start after the largest time lag of each trial.

        Optionally, the (time) unfiltered X source is also returned, both as
        bandpower and as pure projection, computed on the unshifted X (lag 0 of
        a pre-embedded X).

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                ([lag_name], [trial_name], sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                ([trial_name], sample_name, featureY_name).
            trial_name (str, optional): Label for trial dimension. Independent of the
                one used in fit, so continuous and trial data can be mixed.
            lag_name (str, optional): Label of the lag dimension if X is already
                temporally embedded. Its coords must match the training time lags.
            return_unfiltered (bool): Whether to also return the X source before the
                temporal filter (Px and sx). With a pre-embedded X, its lags must
                include 0. Defaults to False.

        Return:
            tuple: A tuple (Sx, Sy), or (Sx, Sy, Px, sx) if return_unfiltered, where:
                Sx (DataArray): Reconstructed source of modality X, with the
                    dimensions of Sy.
                Sy (DataArray): Reconstructed source of modality Y.
                Px (DataArray): Bandpower of the unshifted X source, without temporal
                    filter, with the dimensions and coords of Sx.
                sx (DataArray): Unshifted X source itself (not bandpower), Wx.T @ X,
                    with dims ([trial_name], sample_name, latent_featureX_name) and
                    the time axis of X.

        Raises:
            ValueError: If return_unfiltered and a pre-embedded X has no lag 0.
        """

        # Check that X and Y match the training data
        X, Y = self.validate_inputs_transform(X, Y, trial_name, lag_name)

        # Unshifted X (lag 0 of a pre-embedded X), for the unfiltered sources
        if return_unfiltered:
            if lag_name is not None:
                if 0 not in X[lag_name]:
                    raise ValueError(
                        f"return_unfiltered needs lag 0 in the {lag_name} dim of X."
                    )
                else:
                    X0 = X.sel({lag_name: 0}, drop=True)
            else:
                X0 = X

        # Time-embedded X (or the pre-computed lags), dims (lag, [trial], time, feature)
        X = embed_lags(X, self.time_lags, lag_name, self.lag_name)

        # Drop the Y samples whose epochs overlap the zero padding of a raw X
        if lag_name is None:
            Y = self.crop_padding(X, Y)

        # X source of each lag copy, dims (lag, [trial], time, latent_featureX)
        sx = xr.dot(X, self.Wx, dim=self.featureX_name)
        sx = sx.transpose(..., self.sample_name, self.latent_featureX_name)

        # Bandpower of each lag copy and component, i.e. its variance within each epoch
        epochs = self.epoch_mapping(X, Y)
        Csx = epoch_covariances(sx.data, epochs)
        phi_x = np.diagonal(Csx, axis1=-2, axis2=-1)

        # Reconstructed sources: temporal filter over the lagged bandpower
        Sx = np.einsum("l...c,lc->...c", phi_x, self.Wt.data)
        Sy = Y @ self.Wy

        # Sx with the dims and coords of Sy, and the component labels of Wx
        Sx = Sy.copy(data=Sx)
        Sx = Sx.rename({self.latent_featureY_name: self.latent_featureX_name})
        Sx = Sx.assign_coords(
            {self.latent_featureX_name: self.Wx[self.latent_featureX_name].data}
        )

        if not return_unfiltered:
            return Sx, Sy
        else:
            # Unfiltered sources: the unshifted X source, and its bandpower in each epoch
            sx = xr.dot(X0, self.Wx, dim=self.featureX_name)
            sx = sx.transpose(..., self.latent_featureX_name)
            Csx = epoch_covariances(sx.data, epochs)
            phi_x = np.diagonal(Csx, axis1=-2, axis2=-1)
            Px = Sx.copy(data=phi_x)

            return Sx, Sy, Px, sx

    def build_spatial_patterns(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
    ):
        """Compute the spatial patterns Ax and Ay of the learned filters Wx and Wy.

        Following :cite:t:`haufe_interpretation_2014`, the patterns are
        A = Cov(X) W Cov(S)^{-1}, with S = X W the sources. The covariances are the
        ones of training, i.e. of X within the epochs (averaged over lags, trials, and
        epochs), and of Y at the epochs. Ax and Ay have the same dims and coords as Wx
        and Wy, respectively.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
            trial_name (str, optional): Label for trial dimension of X and Y.
            lag_name (str, optional): Label of the lag dimension if X is already
                temporally embedded.
        """
        cite("haufe_interpretation_2014")

        # Inputs as in training, with the features in the order of the filters
        X, Y = self.validate_inputs_transform(X, Y, trial_name, lag_name)
        X = X.sel({self.featureX_name: self.Wx[self.featureX_name].values})
        Y = Y.sel({self.featureY_name: self.Wy[self.featureY_name].values})
        X = embed_lags(X, self.time_lags, lag_name, self.lag_name)
        if lag_name is None:
            Y = self.crop_padding(X, Y)

        # Covariance of X within the epochs (averaged over lags, trials, and epochs)
        Cxxe = epoch_covariances(X.data, self.epoch_mapping(X, Y))
        Cxx = Cxxe.reshape(-1, self.Nx, self.Nx).mean(axis=0)

        # Covariance of Y at the epochs (pooled over trials)
        Cyy = np.cov(Y.data.reshape(-1, self.Ny), rowvar=False)

        # Patterns A = C W (W' C W)^-1 (pseudo-inverse in case sources are dependent)
        Wx = self.Wx.data
        Wy = self.Wy.data
        Ax = Cxx @ Wx @ np.linalg.pinv(Wx.T @ Cxx @ Wx)
        Ay = Cyy @ Wy @ np.linalg.pinv(Wy.T @ Cyy @ Wy)

        self.Ax = self.Wx.copy(data=Ax)
        self.Ay = self.Wy.copy(data=Ay)

    def set_time_lags(self, X: xr.DataArray, lag_name: str | None = None):
        """Set the validated time lags used for the temporal embedding of the power.

        Time lags are either time_lags at initialization, or the lag_name coords of
        an already embedded X (in time units), as checked in validate_inputs_fit. In
        both cases they are validated (positive, sorted, within the data domain), and
        duplicated lags are removed with a warning (keeping the first lag copy of a
        pre-embedded X, see embed_lags).

        Args:
            X (DataArray): Input data for modality X, with a time coordinate.
            lag_name (str, optional): Label of the lag dimension of a pre-embedded X.
        """
        lags = self.init_time_lags if lag_name is None else X[lag_name].values
        lags = np.asarray(lags)

        # Remove duplicated lags, keeping one of each
        unique_lags, counts = np.unique(lags, return_counts=True)
        if np.any(counts > 1):
            warnings.warn(
                f"Duplicated time lags {unique_lags[counts > 1]} found. Only one of "
                "each is kept.",
                UserWarning,
                stacklevel=3,
            )
            lags = unique_lags

        duration = X.time[-1].item() - X.time[0].item()
        self.time_lags = validate_time_shifts(T=duration, time_shifts=lags)
        self.N_lags = len(self.time_lags)

        # Name of the lag dim of Wt
        self.lag_name = "time_shift" if lag_name is None else lag_name

    def epoch_mapping(self, X: xr.DataArray, Y: xr.DataArray) -> list[slice]:
        """X samples within a power_window centered at each Y sample.

        Args:
            X (DataArray): Input data for modality X, with a time coordinate.
            Y (DataArray): Input data for modality Y, with a time coordinate.

        Returns:
            list: One slice of X samples per Y sample.

        Raises:
            ValueError: If power_window is shorter than one sampling period of Y.
        """
        tx = X.time.data
        ty = Y.time.data
        dty = ty[1] - ty[0]

        # Window length: power_window, or one sampling period of Y by default
        window = self.power_window

        # Shorter windows would leave X samples between two Y samples out
        if window < dty * (1 - 1e-6):
            raise ValueError(
                f"power_window ({window:.4g}) must be at least one sampling period of "
                f"Y ({dty:.4g})."
            )

        # X samples in [ty - w/2, ty + w/2), tol sends edge samples to the later epoch
        tol = 1e-6 * (tx[1] - tx[0])
        starts = np.searchsorted(tx, ty - window / 2 - tol)
        stops = np.searchsorted(tx, ty + window / 2 - tol)

        # Epochs with < 2 X samples have 0 covariance -> fake 0 power -> bias
        if np.any(stops - starts < 2):
            raise ValueError(
                "Some Y samples have fewer than 2 X samples in their window. Crop Y to "
                "the time span of X, or increase power_window."
            )

        epochs = [slice(a, b) for a, b in zip(starts, stops)]
        return epochs

    def crop_padding(self, X: xr.DataArray, Y: xr.DataArray) -> xr.DataArray:
        """Keep only the Y samples whose epochs start after the zero padding of X.

        The lagged copies of a raw X are zero padded at the start of each trial (see
        embed_lags), up to the largest time lag. Epochs overlapping that padding would
        mix zeros into their covariances, which would artificialy reduce the power.
        To avoid this bias, their Y samples are dropped. As all trials
        share the same time axis, the same samples are dropped in each of them.

        Args:
            X (DataArray): Input data for modality X, with a time coordinate.
            Y (DataArray): Input data for modality Y, with a time coordinate.

        Returns:
            DataArray: Y without the samples whose epochs overlap the zero padding.

        Raises:
            ValueError: If no Y sample is left.
        """
        tx = X.time.data
        ty = Y.time.data

        # Window length: power_window, or one sampling period of Y by default
        window = self.power_window

        # First non-padded sample of the largest lag copy (as in temporal_embedding)
        first = np.searchsorted(tx, tx[0] + self.time_lags.max())
        if first == len(tx):
            raise ValueError("The time lags are longer than the data.")

        # Keep the Y samples whose epoch starts at or after it (tol as in epoch_mapping)
        tol = 1e-6 * (tx[1] - tx[0])
        t_min = tx[first] + window / 2
        Y_cropped = Y.sel({self.sample_name: ty >= t_min - tol})
        if Y_cropped.sizes[self.sample_name] == 0:
            raise ValueError("No epochs left after the zero padding of the time lags.")

        return Y_cropped

    def estimate_filters(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate the filters Wx, Wy, and Wt from the epochs of X and Y.

        Main method that runs the multicomponent mSPoC algorithm. Each component is
        found by the one-unit algorithm (see one_unit_algorithm), and further
        components via deflation of the filters Wx and Wy found so far (see deflate).
        N_components are computed, and the ones with the largest absolute correlation 
        are kept, in descending order.

        If L2 regularization is used, the metrics Cxx and Cyy of the wx and wy steps
        are shrunk once, in the full space, towards a scaled identity, and then
        projected onto the deflated subspaces ("orthogonal" deflation) or kept as
        they are ("schur" deflation). The shrinkage intensities are stored in
        gamma_x and gamma_y (0 without L2 regularization).

        The algorithm is configured by the attributes N_components, N_restarts,
        max_iter, tol, random_state, l1_reg, l2_reg, n_units (if None, one unit per
        trial if trials are given in fit, otherwise one unit per epoch), and
        deflation.

        Args:
            X (DataArray): Time-embedded X, with dims (lag_name, [trial_name],
                sample_name, featureX_name).
            Y (DataArray): Y at the epochs, with dims ([trial_name], sample_name,
                featureY_name).

        Returns:
            tuple: A tuple (Wx, Wy, Wt) with the filters for X, Y, and the temporal
            embedding, with shapes (Nx, N_components), (Ny, N_components), and
            (N_lags, N_components).
        """

        # A new generator on each fit, so every fit is reproducible on its own
        rng = np.random.default_rng(self.random_state)

        N_lags, Nx, Ny = X.shape[0], X.shape[-1], Y.shape[-1]

        # Covariances of X in each epoch (one per Y sample), (lag, [trial], NE, Nx, Nx)
        epochs = self.epoch_mapping(X, Y)
        Cxxe = epoch_covariances(X.data, epochs)

        # Pool the epochs over trials (trial-major, the same for X and Y)
        Cxxe = Cxxe.reshape(N_lags, -1, Nx, Nx)
        Y_epochs = Y.data.reshape(-1, Ny)
        NE = len(Y_epochs)

        # Center Y over the epochs
        Y_epochs = Y_epochs - Y_epochs.mean(axis=0)

        # Metrics of the wx and wy steps, averaged over the (lagged) epochs
        Cxx = Cxxe.mean(axis=(0, 1))
        Cyy = (Y_epochs.T @ Y_epochs) / NE

        # Ledoit-Wolf units: the trials, or the epochs, by default
        if self.n_units is not None:
            n_units = self.n_units
        else:
            n_units = self.Nkx if self.Nkx is not None else NE
        # L2 penalty: shrinkage towards the scaled identity (given, or Ledoit-Wolf)
        self.gamma_x = self.gamma_y = 0.0
        if self.l2_reg[0]:
            # Samples of X ordered by trial, so that contiguous units are trials
            X_samples = np.moveaxis(X.data, 0, -2).reshape(-1, Nx)
            Cxx, self.gamma_x = shrink_cov(
                Cxx, self.l2_reg[0], X=X_samples, n_units=n_units
            )
        if self.l2_reg[1]:
            Cyy, self.gamma_y = shrink_cov(
                Cyy, self.l2_reg[1], X=Y_epochs, n_units=n_units
            )

        # Singular metrics would amplify numerical noise in the wx and wy steps
        for C, name in ((Cxx, "X"), (Cyy, "Y")):
            eigvals = np.linalg.eigvalsh(C)
            if eigvals[0] < 1e-10 * eigvals[-1]:
                raise ValueError(
                    f"The covariance of {name} is singular (rank-deficient data), so "
                    "mSPoC is not reliable. Use a non-zero l2_reg."
                )

        # L1 sub-problems use FISTA (inside majorization-minimization for wx)
        if self.l1_reg[0] or self.l1_reg[1]:
            cite("beck_fast_2009")
        if self.l1_reg[0]:
            cite("sriperumbudur_majorization-minimization_2011")

        # Initialize filters
        Wx = np.zeros([Nx, self.N_components])
        Wy = np.zeros([Ny, self.N_components])
        Wt = np.zeros([N_lags, self.N_components])
        corr = np.zeros(self.N_components)

        for i in range(self.N_components):
            # Deflate the problem w.r.t. the filters found so far (see deflate)
            if i == 0:  # (no deflation for first component)
                Bx = np.eye(Nx)
                By = np.eye(Ny)
                Cxxe_i = Cxxe
                Y_i = Y_epochs
                Cxx_i = Cxx
                Cyy_i = Cyy
            else:
                Bx, By, Cxxe_i, Y_i, Cxx_i, Cyy_i = deflate(
                    Wx[:, :i], Wy[:, :i], Cxxe, Y_epochs, Cxx, Cyy, rng, self.deflation
                )
            # Run one-unit algorithm in the deflated space
            corr[i], wx, wy, wt = self.one_unit_algorithm(
                Cxxe_i, Y_i, Cxx_i, Cyy_i, rng
            )
            # Project into original space
            Wx[:, i] = Bx @ wx
            Wy[:, i] = By @ wy
            Wt[:, i] = wt

        # Order components in descending absolute correlation
        order = np.argsort(np.abs(corr))[::-1]
        Wx = Wx[:, order]
        Wy = Wy[:, order]
        Wt = Wt[:, order]

        return Wx, Wy, Wt

    def one_unit_algorithm(self, Cxxe, Y, Cxx, Cyy, rng):
        """Run the one-unit algorithm of mSPoC to compute one single set of filters.

        The filters wt, wy, and wx are updated one at a time, each as the solution of
        a single-filter sub-problem (see solvers.solve_filter) that maximizes the
        covariance between the temporally filtered power of wx' X and wy' Y:

        - wt: linear problem, c = <Phi_x sy>, with metric I (unit norm).
        - wy: linear problem, c = <Y h(phi_x)>, with metric Cyy and relative L1
          threshold kappa_y.
        - wx: quadratic problem, A = <sy h(Cxxe)>, with metric Cxx (SPoC) and relative
          L1 threshold kappa_x.

        In the wx step, A = sum_{i, n} wt_i sy(n) Cxxe(i, n) / NE, i.e. the temporal
        filter is applied together with sy, and h(Cxxe) is never built for efficiency.

        Args:
            Cxxe (ndarray): Covariances of the lagged copies of X within each epoch,
                shape (N_lags, NE, Nx, Nx).
            Y (ndarray): Y at the epochs, centered, shape (NE, Ny).
            Cxx (ndarray): Metric of the wx step, i.e. the (shrunk) covariance of X
                averaged over the epochs, shape (Nx, Nx).
            Cyy (ndarray): Metric of the wy step, i.e. the (shrunk) covariance of Y
                over the epochs, shape (Ny, Ny).
            rng (Generator): Random generator for the initialization of wx and wy.

        Returns:
            tuple: A tuple (corr, wx, wy, wt) with the correlation between the
                reconstructed sources and the filters of the best restart.
        """

        Nx, Ny, NE = Cxxe.shape[-1], Y.shape[1], Y.shape[0]

        def embedded_power(wx):
            """Epoch-wise power of wx' X, lagged and centered (N_lags, NE)."""
            Phi_x = np.einsum("lnij,i,j->ln", Cxxe, wx, wx)
            Phi_x = Phi_x - Phi_x.mean(axis=1, keepdims=True)
            return Phi_x

        # To keep track of best model
        corr_best = 0.0
        wx_best = wy_best = wt_best = None
        for _ in range(self.N_restarts):
            # Initialize random filters
            wx = rng.standard_normal(Nx)
            wy = rng.standard_normal(Ny)
            Phi_x = embedded_power(wx)
            # For convergence condition
            converged = False
            corr_old = -np.inf
            for _ in range(self.max_iter):
                # Temporal filter wt (linear problem, unit norm)
                sy = Y @ wy
                ct = Phi_x @ sy / NE
                wt = solve_filter(c=ct)
                # Spatial filter wy (linear problem, metric Cyy)
                cy = Y.T @ (wt @ Phi_x) / NE
                wy = solve_filter(c=cy, B=Cyy, kappa=self.l1_reg[1], w_init=wy)
                sy = Y @ wy
                # Spatial filter wx (quadratic problem, metric Cxx), wt applied with sy
                Ax = np.tensordot(np.outer(wt, sy), Cxxe, axes=2) / NE
                wx = solve_filter(A=Ax, B=Cxx, kappa=self.l1_reg[0])
                # Epoch-wise bandpower with temporal embedding
                Phi_x = embedded_power(wx)
                # Compute correlation
                corr = np.corrcoef(wt @ Phi_x, sy)[0, 1]
                # Check for convergence
                if np.abs(corr - corr_old) < self.tol:
                    converged = True
                    break
                corr_old = corr

            # Save filters for best converged model
            if converged and np.abs(corr) > np.abs(corr_best):
                corr_best, wx_best, wy_best, wt_best = corr, wx, wy, wt

        if wx_best is None:
            warnings.warn(
                f"mSPoC did not converge in any of the {self.N_restarts} restarts "
                "(reached maximum number of iterations). Returning last computed "
                "filters and correlation.",
                stacklevel=4,
            )
            corr_best, wx_best, wy_best, wt_best = corr, wx, wy, wt

        return corr_best, wx_best, wy_best, wt_best

    def validate_inputs_fit(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Validates fit function inputs and returns it with correct dims and labels.

        This method ensures that the input data to the fit function, X and Y, i.e. those
        used for training, have the expected dimension labels and sizes, and returns
        them with the sample and feature dimensions last. It also initializes the
        number of samples (Ntx, Nty), features (Nx, Ny, N_features), trials (Nkx, Nky),
        and epochs (N_samples), and the number of components (N_features if None).

        The time lags come from exactly one origin: time_lags given at initialization,
        or the lag_name dim of an already embedded X. Passing both or none raises an
        error.

        X and Y must span the same time domain, i.e. their first, and last, time
        points can differ by at most one sampling period of Y (see
        validate_time_coverage).
        Sizes are checked with the epochs, i.e. the samples of Y, as the samples of
        both modalities: X must have at least two samples per epoch, i.e. twice the
        sampling rate of Y, the number of trials must match between X and Y, and
        N_components cannot exceed N_features.
        The number of epochs is not checked against the number of features, since
        singular covariances are caught later, after the (optional) L2 shrinkage.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
            trial_name (str, optional): Label for trial dimension of X and Y.
            lag_name (str, optional): Label for lag dimension of a pre-embedded X.

        Returns:
            tuple: A tuple (X, Y) where:
                X (DataArray): Input data for modality X with dims (..., sample_name,
                    featureX_name), the optional lag and trial dims keeping their
                    input order.
                Y (DataArray): Input data for modality Y with dims ([trial_name],
                    sample_name, featureY_name).

        Raises:
            ValueError: If the time lags are given twice or not at all, if X or Y do
                not have the expected dimensions, if X and Y do not span the same
                time domain, if X has less than two samples per epoch, if the number of
                trials differs between X and Y, or if N_components exceeds N_features.
        """

        # Check that the time lags come from exactly one origin
        if self.init_time_lags is not None and lag_name is not None:
            raise ValueError(
                "Time lags given both at initialization and as lag_name. Pass only "
                "one."
            )
        if self.init_time_lags is None and lag_name is None:
            raise ValueError("No time lags given: pass time_lags or lag_name.")

        # Check that X and Y have correct dimension labels
        validate_dimension_labels(
            X,
            Y,
            sample_name=self.sample_name,
            featureX_name=self.featureX_name,
            featureY_name=self.featureY_name,
            trial_name=trial_name,
            lag_name=lag_name,
        )

        # Bring dimensions to a pre-established order
        X = X.transpose(..., self.sample_name, self.featureX_name)
        Y = Y.transpose(..., self.sample_name, self.featureY_name)

        # Read dimensions sizes (samples, features, and optional trials)
        self.Ntx, self.Nx = X.sizes[self.sample_name], X.sizes[self.featureX_name]
        self.Nty, self.Ny = Y.sizes[self.sample_name], Y.sizes[self.featureY_name]
        if trial_name is not None:
            self.Nkx, self.Nky = X.sizes[trial_name], Y.sizes[trial_name]
        else:
            self.Nkx, self.Nky = None, None
        self.N_features = min(self.Nx, self.Ny)

        # Check that X and Y span the same time domain, up to the sampling period of Y
        validate_time_coverage(X, Y, self.sample_name)

        # Each epoch needs at least two X samples
        if self.Ntx < 2 * self.Nty:
            raise ValueError(
                f"Number of X samples ({self.Ntx}) must be at least twice the "
                f"number of Y samples ({self.Nty})"
            )

        # Sizes, with the epochs (samples of Y) as samples of both modalities
        self.N_components, self.N_samples = validate_dimension_sizes(
            self.Nty, self.Nty, self.N_features, self.N_components, self.Nkx, self.Nky
        )

        return X, Y

    def validate_inputs_transform(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Validates that to-be-transformed data have expected dim. labels and sizes.

        This method ensures that the model is fitted, and that X and Y have the same
        dimension labels and features (number and labels, in any order) than the ones
        used during training. The number of time points and trials can differ from
        training, but trials must be the same between X and Y, and X and Y must span
        the same time domain (see validate_time_coverage). Pre-computed lags must be
        exactly the ones used during training (in any order, and possibly duplicated,
        see embed_lags).

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
            trial_name (str, optional): Label for trial dimension of X and Y.
            lag_name (str, optional): Label for lag dimension of a pre-embedded X.

        Returns:
            tuple: A tuple (X, Y) where:
                X (DataArray): Input data for modality X with dims (..., sample_name,
                    featureX_name), the optional lag and trial dims keeping their
                    input order.
                Y (DataArray): Input data for modality Y with dims ([trial_name],
                    sample_name, featureY_name).
        """

        if self.Wx is None:
            raise ValueError("The model is not fitted yet: call fit before transform.")

        # Check that X and Y have correct dimension labels
        validate_dimension_labels(
            X,
            Y,
            sample_name=self.sample_name,
            featureX_name=self.featureX_name,
            featureY_name=self.featureY_name,
            trial_name=trial_name,
            lag_name=lag_name,
        )

        # Bring dimensions to a pre-established order
        X = X.transpose(..., self.sample_name, self.featureX_name)
        Y = Y.transpose(..., self.sample_name, self.featureY_name)

        # Read dimensions sizes (samples, features, and optional trials)
        Ntx, Nx = X.sizes[self.sample_name], X.sizes[self.featureX_name]
        Nty, Ny = Y.sizes[self.sample_name], Y.sizes[self.featureY_name]
        if trial_name is not None:
            Nkx, Nky = X.sizes[trial_name], Y.sizes[trial_name]
        else:
            Nkx, Nky = None, None

        # Validate sizes between modalities and against training
        if Nkx != Nky:
            raise ValueError(
                f"Number of trials must be the same between X ({Nkx}) and Y ({Nky})"
            )
        validate_time_coverage(X, Y, self.sample_name)
        if Ntx < 2 * Nty:
            raise ValueError(
                f"Number of X samples ({Ntx}) must be at least twice the "
                f"number of Y samples ({Nty})"
            )
        if Nx != self.Nx:
            raise ValueError(
                f"X should have {self.Nx} number of features, but {Nx} was found!"
            )
        if Ny != self.Ny:
            raise ValueError(
                f"Y should have {self.Ny} number of features, but {Ny} was found!"
            )
        # Feature labels must match training, as the projections align them by label
        for A, W, name in (
            (X, self.Wx, self.featureX_name),
            (Y, self.Wy, self.featureY_name),
        ):
            if set(A[name].values) != set(W[name].values):
                raise ValueError(
                    f"The {name} labels of the data must match those used in training."
                )

        # Validate lags
        if lag_name is not None:
            lags = np.unique(X[lag_name].values)
            if not np.array_equal(lags, self.time_lags):
                raise ValueError(
                    f"X should have time lags {self.time_lags}, but {lags} were "
                    "found!"
                )

        return X, Y

    def convert_filters_to_DataArray(
        self,
        Wx: np.ndarray,
        Wy: np.ndarray,
        Wt: np.ndarray,
        X: xr.DataArray,
        Y: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Convert filters Wx, Wy, Wt from numpy to DataArray format.

        Builds the DataArrays with the right dimensions and coordinates.

        Args:
            Wx (ndarray): Filter matrix for modality X with shape (Nx, N_components).
            Wy (ndarray): Filter matrix for modality Y with shape (Ny, N_components).
            Wt (ndarray): Filter matrix for time embedding with shape
                (N_lags, N_components).
            X (DataArray): DataArray containing the features of modality X.
            Y (DataArray): DataArray containing the features of modality Y.

        Returns:
            tuple: A tuple containing the DataArray versions of Wx, Wy, and Wt, with
                dims (featureX_name, latent_featureX_name), (featureY_name,
                latent_featureY_name), and (lag_name, latent_featureT_name).
        """

        component_name_x = [f"Sx{i + 1}" for i in range(self.N_components)]
        component_name_y = [f"Sy{i + 1}" for i in range(self.N_components)]
        componentt_name = [f"St{i + 1}" for i in range(self.N_components)]

        coords_x = {
            self.featureX_name: X[self.featureX_name],
            self.latent_featureX_name: component_name_x,
        }
        Wx_xr = xr.DataArray(Wx, coords=coords_x)

        coords_y = {
            self.featureY_name: Y[self.featureY_name],
            self.latent_featureY_name: component_name_y,
        }
        Wy_xr = xr.DataArray(Wy, coords=coords_y)

        Wt_xr = xr.DataArray(
            Wt,
            coords={
                self.lag_name: self.time_lags,
                self.latent_featureT_name: componentt_name,
            },
        )

        return Wx_xr, Wy_xr, Wt_xr


def epoch_covariances(X: np.ndarray, epochs: list[slice]) -> np.ndarray:
    """Covariance matrices of X within each epoch.

    Args:
        X (ndarray): Data with time in the second to last axis, shape (..., time, Nx).
        epochs (list): One slice of time samples per epoch (see epoch_mapping).

    Returns:
        ndarray: Epoch-wise covariances, as np.cov per epoch, shape (..., NE, Nx, Nx).
    """
    C_list = []
    for epoch in epochs:
        # Samples of the epoch, centered
        Xe = X[..., epoch, :]
        Xe = Xe - Xe.mean(axis=-2, keepdims=True)

        # Covariance within the epoch, as np.cov
        n = Xe.shape[-2]
        C = np.einsum("...ti,...tj->...ij", Xe, Xe) / (n - 1)
        C_list.append(C)

    # Stack the epochs before the two feature axes
    Cxxe = np.stack(C_list, axis=-3)
    return Cxxe


def deflate(
    Wx: np.ndarray,
    Wy: np.ndarray,
    Cxxe: np.ndarray,
    Y: np.ndarray,
    Cxx: np.ndarray,
    Cyy: np.ndarray,
    rng: np.random.Generator,
    method: str = "orthogonal",
) -> tuple[np.ndarray, ...]:
    """Deflate the mSPoC problem, so contribution from previous components is removed.

    Returns the inputs of the one-unit algorithm for the next component, together
    with the bases Bx and By, to map filters to deflated space (B @ w).

    Two methods are available:
    - "orthogonal": restrict the search to the subspaces orthogonal to the columns of
      Wx and Wy, using orthonormal bases of their complements (see
      get_orthonormal_matrix) to project Cxxe, Y, Cxx, and Cyy. Filters are
      orthogonal, but since later components are solved in a rotated basis, L1
      sparsity only holds for the first one.
    - "schur": remove the sources of Wx and Wy from the data (Schur complement),
      x <- Px x with Px = I - Cxx Wx (Wx' Cxx Wx)^-1 Wx', and likewise for y. The
      search stays in the original coordinates (B = I), and the metrics Cxx and Cyy
      are kept, so L1 sparsity holds for every component, and without L1 the new
      sources are uncorrelated (w.r.t. the metrics) with the previous ones.

    Args:
        Wx (ndarray): Filters of X found so far, shape (Nx, Nc).
        Wy (ndarray): Filters of Y found so far, shape (Ny, Nc).
        Cxxe (ndarray): Epoch-wise covariances of X, shape (N_lags, NE, Nx, Nx).
        Y (ndarray): Y at the epochs, shape (NE, Ny).
        Cxx (ndarray): Metric of the wx step, shape (Nx, Nx).
        Cyy (ndarray): Metric of the wy step, shape (Ny, Ny).
        rng (Generator): Random generator for the completion of the bases (only
            used by "orthogonal").
        method (str): Deflation method, "orthogonal" or "schur". Defaults to
            "orthogonal".

    Returns:
        tuple: A tuple (Bx, By, Cxxe, Y, Cxx, Cyy) with the bases, of shapes
            (Nx, Nx - Nc) and (Ny, Ny - Nc) for "orthogonal" and the identities for
            "schur", and the deflated inputs.
    """

    Nx, Ny = len(Wx), len(Wy)

    if method == "schur":
        # Oblique projections removing the sources of the found filters
        Bx = np.eye(Nx)
        By = np.eye(Ny)

        Px = np.eye(Nx) - Cxx @ Wx @ np.linalg.solve(Wx.T @ Cxx @ Wx, Wx.T)
        Py = np.eye(Ny) - Cyy @ Wy @ np.linalg.solve(Wy.T @ Cyy @ Wy, Wy.T)

        Cxxe_deflated = Px @ Cxxe @ Px.T
        Y_deflated = Y @ Py.T
        Cxx_deflated = Cxx
        Cyy_deflated = Cyy

    elif method == "orthogonal":

        Bx = get_orthonormal_matrix(Wx, rng)
        By = get_orthonormal_matrix(Wy, rng)

        Cxxe_deflated = Bx.T @ Cxxe @ Bx
        Y_deflated = Y @ By
        Cxx_deflated = Bx.T @ Cxx @ Bx
        Cyy_deflated = By.T @ Cyy @ By
    else:
        raise ValueError(f"Unknown deflation method {method}. Use 'orthogonal' or "
                         "'schur'.")

    return Bx, By, Cxxe_deflated, Y_deflated, Cxx_deflated, Cyy_deflated


def get_orthonormal_matrix(
    W: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate an orthonormal basis for an N-dim. space with basis vectors from W.

    The columns of W are some of the basis vectors.

    Args:
        W (np.ndarray): A N x Nc array representing the given vectors.
        rng (Generator, optional): Random generator for the completion of the basis.
            Defaults to None (a new, unseeded one).

    Returns:
        basis (np.ndarray): An N x N - Nc orthonormal basis matrix,
        where the columns of W are not present.
    """
    N, Nc = W.shape
    # Normalize w
    W = np.stack([w / np.linalg.norm(w) for w in W.T]).T
    # Generate random vectors to form the rest of the basis
    if rng is None:
        rng = np.random.default_rng()
    random_vectors = rng.standard_normal((N, N - Nc))
    # Combine w with the random vectors
    B = np.column_stack([W, random_vectors])
    # Perform QR decomposition to orthogonalize
    B, _ = np.linalg.qr(B)

    # Ensure W is in the basis (numerical precision can cause slight misalignment)
    for w in W.T:
        # Find the column most aligned with w
        similarity = np.abs(B.T @ w)
        w_idx = np.argmax(similarity)

        # Remove w from B
        B = np.delete(B, w_idx, 1)

    return B
