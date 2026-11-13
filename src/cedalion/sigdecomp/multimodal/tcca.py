"""Temporally embedded CCA-like models for multimodal data.

This module implements classes for performing source decomposition on multimodal data
using temporal embedding techniques. The latter is based on
:cite:t:`biesmann_temporal_2010`.

Includes:
- ElasticNetTCCA: Class for performing Elastic Net regularized temporally embedded CCA.
- StructuredSparseTCCA: Class for performing structured sparse temporally embedded CCA.
- tCCA: Class for performing standard temporally embedded CCA.
"""

import warnings

import numpy as np
import xarray as xr

from cedalion import cite
from cedalion.sigdecomp.multimodal.utils import (
    validate_dimension_labels,
    validate_dimension_sizes,
    validate_sample_alignment,
    validate_time_shifts,
    embed_lags,
    filter_to_pattern,
    validate_l1_reg,
    validate_l2_reg,
    validate_solver,
)

from cedalion.sigdecomp.multimodal.cca import MultimodalSourceDecomposition


class MultimodalSourceDecompositionWithTemporalEmbedding(MultimodalSourceDecomposition):
    """Decomposing multimodal data using linear filters and temporal embedding.

    This main class is inherited by other source decomposition methods, such as tCCA. It
    implements methods to validate input dimensions, apply normalization, and transform
    data from two modalities using filters learned during training. It assumes modality
    Y is delayed with respect to modality X.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features
            between modalities.
        time_lags (np.ndarray): Time lags for temporal embedding. Either these or the
            lag_name dim of a pre-embedded X in fit must be given, not both.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Relative tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance.
            Defaults to True.
        random_state (int, optional): Seed for the random initialization of the
            solver, making fits reproducible. Defaults to None (not reproducible).
        shift_source (bool): Deprecated, will be removed in a future release.
            Whether transform shifts each X source by its optimal time shift (see
            estimate_optimal_shift), from being aligned with the Y sources to the time
            of X. The X sources are temporally filtered (convolved over the lags), so
            the shift only moves this slow, smoothed signal in time and does not
            recover the original fast activity of X. Shifted X sources are also no
            longer aligned with the Y sources. Defaults to False.
    """

    def __init__(
        self,
        N_components: int = None,
        time_lags: np.ndarray = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
        shift_source=False,
    ):
        super(MultimodalSourceDecompositionWithTemporalEmbedding, self).__init__(
            N_components=N_components,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
        )

        # Sample name must be always time
        self.sample_name = "time"

        # Time embedding attributes. Lags given here are kept apart from the validated
        # ones set in fit, so refitting with lag_name still sees whether they were given
        self.init_time_lags = time_lags
        self.time_lags = None
        self.N_lags = None
        self.optimal_shift = None
        self.shift_source = shift_source

        # Optional label of a pre-computed lag dimension
        self.lag_name = None

    # ==================== VALIDATION ====================

    def validate_inputs_fit(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Same as the parent validation, but with sizes checked in the embedded space.

        CCA runs on X embedded with N_lags lags, i.e. on Nx * N_lags features, which
        sets the maximum number of components and the minimum number of samples. The
        lags come from exactly one origin: time_lags given at initialization, or the
        lag_name dim of an already embedded X. Passing both or none raises an error.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
            trial_name (str, optional): Label for trial dimension of X and Y.
            lag_name (str, optional): Label for lag dimension of a pre-embedded X.

        Returns:
            tuple: A tuple (X, Y) where:
                X (DataArray): Input for modality X. shape: ([lag_name], [trial_name],
                    sample_name, featureX_name)
                Y (DataArray): Input for modality Y. shape: ([trial_name], sample_name,
                    featureY_name)
        """

        # Check that the lags come from exactly one origin
        if self.init_time_lags is not None and lag_name is not None:
            raise ValueError(
                "Time lags given both at initialization and as lag_name. Pass only one."
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

        # Define final number of features, counting X once per lag (embedded space)
        if lag_name is not None:
            N_lags = X.sizes[lag_name]
        else:
            N_lags = len(self.init_time_lags)
        self.N_features = min(self.Nx * N_lags, self.Ny)

        # Validate dimension sizes
        # (and only then return number of components and total number of samples)
        self.N_components, self.N_samples = validate_dimension_sizes(
            self.Ntx, self.Nty, self.N_features, self.N_components, self.Nkx, self.Nky
        )

        # Check that X and Y are aligned, i.e. share the same sample coordinates
        validate_sample_alignment(X, Y, self.sample_name)

        return X, Y

    def validate_inputs_transform(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Same as the parent validation, plus the lags of a pre-embedded X.

        Pre-computed lags must be exactly the ones used during training (in any order,
        and possibly duplicated, since they are later reordered by embed_lags).
        """
        X, Y = super().validate_inputs_transform(X, Y, trial_name, lag_name)

        # Validate lags
        if lag_name is not None:
            lags = np.unique(X[lag_name].values)
            if not np.array_equal(lags, self.time_lags):
                raise ValueError(
                    f"X should have time lags {self.time_lags}, but {lags} were found!"
                )

        return X, Y

    # ==================== TEMPORAL EMBEDDING ====================

    def set_time_lags(self, X: xr.DataArray, lag_name: str | None = None):
        """Set the validated time lags used for the embedding during training.

        Lags are either time_lags at initialization, or the lag_name coords of an
        already embedded X (in time units), as checked in validate_inputs_fit. In both
        cases they are validated (positive, sorted), and duplicated lags are removed
        with a warning (keeping the first lag copy of a pre-embedded X, see
        embed_lags). Lags given at initialization must also be shorter than the data
        (per trial, if any), since the model zero-pads them. Lags of a pre-embedded X
        hold real signal history, so they can exceed the duration of a trial.
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

        # Lags embedded by the model are zero padded, so they must fit in the data.
        # Lags of a pre-embedded X were built outside (e.g. on the continuous
        # recording, before epoching), so they are not bounded by the trial length
        if lag_name is None:
            duration = X.time[-1].item() - X.time[0].item()
        else:
            duration = np.inf
        self.time_lags = validate_time_shifts(T=duration, time_shifts=lags)
        self.N_lags = len(self.time_lags)

        # Name of the lag dim of the embedded X (and of the filters Wx)
        self.lag_name = "time_shift" if lag_name is None else lag_name

    # ==================== FILTERS ====================

    def convert_filters_to_DataArray(
        self, Wx: np.ndarray, Wy: np.ndarray, X: xr.DataArray, Y: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Same as the parent conversion, but unfolding the lags of the embedded Wx.

        CCA in the embedded space returns Wx with shape (N_lags * Nx, N_components),
        with features stacked lag by lag (as X is concatenated in fit). It is reshaped
        to (N_lags, Nx, N_components), giving Wx the dims (lag_name, featureX_name,
        latent_featureX_name), with the lag coords of X.

        Args:
            Wx (ndarray): Filter matrix for the embedded X, shape (N_lags * Nx,
                N_components).
            Wy (ndarray): Filter matrix for modality Y with shape (Ny, N_components).
            X (DataArray): Embedded X with dims (lag_name, sample_name, featureX_name).
            Y (DataArray): DataArray containing the features of modality Y.

        Returns:
            A tuple (Wx_xr, Wy_xr) containing the DataArray versions of Wx and Wy.
        """
        Wx = Wx.reshape(self.N_lags, self.Nx, self.N_components)
        X = X.transpose(self.lag_name, self.sample_name, self.featureX_name)
        return super().convert_filters_to_DataArray(Wx, Wy, X, Y)

    def estimate_optimal_shift(
        self, X_emb: xr.DataArray, Y: xr.DataArray, mode: str = "pattern"
    ):
        """Find optimal time shifts for X, one per component.

        Two modes are available:
            - "pattern": picks the lag where the spatio-temporal pattern of Wx
              (:cite:t:`haufe_interpretation_2014`) is strongest, i.e. the lag with
              the largest L2 norm of the pattern over X features.
            - "correlation": picks the time-shifted X that produces the biggest
              correlation between reconstructed sources sx and sy.

        Args:
            X_emb (DataArray): Normalized time-embedded version of X with dimensions
                (lag_name, sample_name, featureX_name).
            Y (DataArray): Normalized input data for modality Y with dimensions
                (sample_name, featureY_name). Only used in "correlation".
            mode (str): Either "pattern" or "correlation". Defaults to "pattern".
        """

        if mode == "pattern":
            cite("haufe_interpretation_2014")

            # Patterns of the embedded Wx, with dims (lag, featureX, latent)
            A = filter_to_pattern(
                self.Wx, X_emb, self.latent_featureX_name, self.sample_name
            )

            # Pattern strength per lag and component, shape (lag, latent)
            strength = np.sqrt((A**2).sum(self.featureX_name))
            strength = strength.transpose(self.lag_name, self.latent_featureX_name).data
            self.optimal_shift = self.time_lags[np.argmax(strength, axis=0)]
            return
        elif mode != "correlation":
            raise ValueError(
                f"mode must be 'pattern' or 'correlation', but '{mode}' was given."
            )

        # Project each lag of X with its own lag of Wx separately
        sx_list = [
            X_emb.sel({self.lag_name: lag}) @ self.Wx.sel({self.lag_name: lag})
            for lag in self.time_lags
        ]
        sy = Y @ self.Wy

        # Iterate over components
        self.optimal_shift = np.zeros(self.N_components)
        for i in range(self.N_components):
            # Calculate correlations
            corrxy = [np.corrcoef(s[:, i], sy[:, i])[0, 1] for s in sx_list]

            # Find optimal lag
            max_corr_ndx = np.argmax(np.abs(corrxy))
            self.optimal_shift[i] = self.time_lags[max_corr_ndx]

    def shift_by_optimal(self, X: xr.DataArray) -> xr.DataArray:
        """Shift X by optimal time shift using zero padding (per trial, if any)."""

        # Time and latent first, so the positional indexing below acts on them only
        dims = X.dims
        X = X.transpose(self.sample_name, self.latent_featureX_name, ...)

        ti = X.time[0].data
        Nt = len(X.time)
        X_shifted = xr.zeros_like(X)
        for i in range(self.N_components):
            start = X.time.searchsorted(ti + self.optimal_shift[i])
            X_shifted[: Nt - start, i] = X[start:, i].data

        return X_shifted.transpose(*dims)

    def to_filter_space(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Same as the parent method, but temporally embedding X before flattening.

        Returns X with dims (lag_name, sample_name, featureX_name) and Y with dims
        (sample_name, featureY_name).
        """
        X, Y = self.validate_inputs_transform(X, Y, trial_name, lag_name)

        # Construct a time-embedded version of X (or take the pre-computed lags)
        X = embed_lags(X, self.time_lags, lag_name, self.lag_name)

        # Flatten (trial, sample) -> sample, after embedding so lags stay within trials
        if trial_name is not None:
            X = self.flatten_trials(X, trial_name)
            Y = self.flatten_trials(Y, trial_name)

        return self.normalization_transform(X, Y)

    def transform(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Apply the linear transformation on the input data using learnt filters.

        This method validates the dimension labels and sizes of the input data to ensure
        consistency with the training data, performs temporal embedding on X, applies
        normalization using the stored parameters, and then projects the normalized
        data onto a lower-dimensional space using the learned filters Wx and Wy. It
        retrieves the transformed arrays, a.k.a reconstructed sources, with the X
        sources aligned with the Y sources. If shift_source is True (deprecated), the X
        sources are instead shifted by their optimal time shifts to the time of X, and
        both outputs are truncated at the end by the smallest shift to remove zero
        padding.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            trial_name (str, optional): Label for trial dimension. Independent of the
                one used in fit, so continuous and trial data can be mixed.
            lag_name (str, optional): Label of the lag dimension if X is already
                temporally embedded. Its coords must match the training time lags.

        Returns:
            tuple: A tuple (X_new, Y_new) where:
                X_new (DataArray): Transformed data for modality X.
                Y_new (DataArray): Transformed data for modality Y.
        """

        # Validate, embed, flatten trials and normalize as during training
        X_in, Y_in = X, Y
        X, Y = self.to_filter_space(X, Y, trial_name, lag_name)

        # Project
        X_new = X @ self.Wx
        Y_new = Y @ self.Wy

        # Restore the trial axis before shifting, so shifts stay within trials
        if trial_name is not None:
            X_new = self.unflatten_trials(X_new, X_in)
            Y_new = self.unflatten_trials(Y_new, Y_in)

        # Shift X by optimal shift found during training
        if self.shift_source:
            warnings.warn(
                "shift_source is deprecated and will be removed in a future release. "
                "The X sources are temporally filtered, so shifting them to the time "
                "of X does not recover the original fast activity of X, and they are "
                "no longer aligned with the Y sources.",
                FutureWarning,
                stacklevel=2,
            )
            X_new = self.shift_by_optimal(X_new)
            # Truncate to remove zero padding
            minimal_shift = np.min(self.optimal_shift)
            X_new = X_new.sel(time=X_new.time <= X_new.time[-1].data - minimal_shift)
            Y_new = Y_new.sel(time=Y_new.time <= Y_new.time[-1].data - minimal_shift)

        return X_new, Y_new


class ElasticNetTCCA(MultimodalSourceDecompositionWithTemporalEmbedding):
    """Perform Elastic Net Canonical Correlation Analysis (CCA) between X_emb and Y.

    Apply temporally embedded CCA (tCCA) with L1 + L2 regularization, a.k.a elastic net.
    The algorithm finds sparse (L1) and normalized (L2) vectors Wx, and Wy as the
    solution to the following constrained optimization problem:

    maximize    Wx^T Cxy Wy
    subject to  Wx^T Cx Wx = 1,  Wy^T Cy Wy = 1,
                ||Wx||_1 <= c1x,  ||Wy||_1 <= c1y,
                ||Wx||^2_2 <= c2x,  ||Wy||^2_2 <= c2y

    where Cx, Cy, and Cxy are the individual and cross-covariance matrices between X_emb
    and Y datasets, and the last four constraints correspond to the standard L1-norm and
    L2-norm penalization terms. c1x and c1y controls sparsity while c2x and c2y controls
    the magnitude of the vectors. PLS algorithms are also captured by this algorithm by
    sending Cx and Cy to the identity matrices.

    The temporally embedded matrix X_emb is constructed by concatenating time-shifted
    versions of the original X. Y is assumed to be the delayed signals with respect to X
    so time shifts are always positive.

    For the one-unit algorithm, the filters are estimated by alternating between the
    linear sub-problems obtained when one of the filters is fixed (based on
    :cite:t:`parkhomenko_sparse_2009`):

    - Update Wx:
        maximize Wx^T cx - lambda_x ||Wx||_1  subject to  Wx^T Bx Wx = 1,
        with cx = Cxy Wy
    - Update Wy:
        maximize Wy^T cy - lambda_y ||Wy||_1  subject to  Wy^T By Wy = 1,
        with cy = Cxy^T Wx

    where the L1 thresholds are set relative to the largest entry of the linear term,
    lambda_x = 2 kappa_x ||cx||_inf (same for Wy). Each sub-problem is solved by
    solvers.solve_filter, either in the original space with Bx = Cx and By = Cy
    ("alternating" solver, truly sparse filters), or in the whitened space with
    Bx = By = I and K = Cx^(-1/2) Cxy Cy^(-1/2) in place of Cxy ("whitening" solver,
    faster but the filters are in general not sparse). See ElasticNetCCA for a detailed
    explanation of both solvers. Note that closely spaced lags of slow signals can make
    Cx nearly singular, in which case the "alternating" solver raises a ValueError and
    L2 regularization or the "whitening" solver should be used.

    For PLS, Cx and Cy are the identity, so both solvers coincide. If L2 regularization
    is used, Cx and Cy are first shrunk towards a scaled identity,
    Cx <- (1 - gamma_x) Cx + gamma_x m_x I (same for Cy), with
    m_x = trace(Cx) / (Nx N_lags) the mean variance of X_emb. The shrinkage intensities
    are either given or estimated with Ledoit-Wolf
    (:cite:t:`ledoit_well-conditioned_2004`).

    Multiple components are obtained via a deflation method, subtracting from Cxy the
    rank-1 term rho Cx Wx Wy^T Cy on each iteration, with rho = Wx^T Cxy Wy (equivalent
    to subtracting from K its 1-rank approximation). The returned vectors Wx and Wy are
    ordered in descending order w.r.t. the canonical correlations (only approximately
    if L1 is used).

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between
            modalities.
        l1_reg (float or list of floats): list containing the relative L1 thresholds
            kappa_x and kappa_y (see above), each in [0, 1). If a single float is
            provided, then kappa_x = kappa_y.
        l2_reg (float, str, or list): Shrinkage intensities gamma_x and gamma_y (see
            above), each a number in [0, 1], or "lw" / "auto" to estimate it with
            Ledoit-Wolf. If a single value is provided, then gamma_x = gamma_y.
        n_units (int, optional): Number of contiguous blocks of samples (units) used
            by Ledoit-Wolf to estimate the variance of the covariance. Defaults to
            None, i.e. the number of trials if trials are given in fit, otherwise one
            unit per sample (i.i.d.). Only used if l2_reg is "lw" / "auto".
        time_lags (np.ndarray): Time lags for temporal embedding. Exclusive
            with lag_name in fit.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Relative tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance.
            Defaults to True.
        random_state (int, optional): Seed for the random initialization of the
            solver, making fits reproducible. Defaults to None (not reproducible).
        pls (bool): Whether to perform PLS regression. Defaults to False.
        shift_source (bool): Deprecated, will be removed in a future release.
            Whether transform shifts each X source by its optimal time shift (see
            estimate_optimal_shift), from being aligned with the Y sources to the time
            of X. The X sources are temporally filtered (convolved over the lags), so
            the shift only moves this slow, smoothed signal in time and does not
            recover the original fast activity of X. Shifted X sources are also no
            longer aligned with the Y sources. Defaults to False.
        solver (str): Space where the sub-problems are solved, either "alternating"
            (original space) or "whitening" (whitened space), see above. Defaults to
            "alternating".

    Attributes:
        N_components (int): Number of components extracted.
        Ntx (int): Number of samples (per trial, if any) of modality X during training.
        Nty (int): Number of samples (per trial, if any) of modality Y during training.
        Nkx (int): Number of trials of modality X during training (None if no trials).
        Nky (int): Number of trials of modality Y during training (None if no trials).
        N_samples (int): Total number of samples, pooled over trials (Ntx * Nkx).
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        N_features (int): Maximum number of components, i.e. the minimum of
            Nx * N_lags and Ny.
        time_lags (np.ndarray): Validated time lags used for the embedding.
        N_lags (int): Number of time lags.
        lag_name (str): Label of the lag dimension of Wx ("time_shift", unless a
            pre-embedded X is given in fit).
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for X.
        latent_featureY_name (str): Label of transformed latent space dimension for Y.
        gamma_x (float): Shrinkage intensity used for the L2 regularization of X,
            either given in l2_reg or estimated with Ledoit-Wolf (None for PLS).
        gamma_y (float): Shrinkage intensity used for the L2 regularization of Y,
            either given in l2_reg or estimated with Ledoit-Wolf (None for PLS).
        Wx (DataArray): Linear filters for dataset X with dimensions
            (lag_name, featureX_name, latent_featureX_name).
        Wy (DataArray): Linear filters for dataset Y with dimensions
            (featureY_name, latent_featureY_name).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
        optimal_shift (np.ndarray): Optimal time shift for each component.
    """

    def __init__(
        self,
        N_components: int = None,
        l1_reg: float | list[float, float] = 0,
        l2_reg: float | str | list = 0,
        n_units: int | None = None,
        time_lags=None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
        pls: bool = False,
        shift_source=False,
        solver: str = "alternating",
    ):
        super(ElasticNetTCCA, self).__init__(
            N_components=N_components,
            time_lags=time_lags,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
            shift_source=shift_source,
        )

        # CCA or PLS
        self.pls = pls
        self.solver = validate_solver(solver)

        # Validate regularization parameters
        self.l1_reg = validate_l1_reg(l1_reg, relative=True)
        self.l2_reg = validate_l2_reg(l2_reg)
        self.n_units = n_units

    def fit(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        featureX_name: str = "channel",
        featureY_name: str = "channel",
        trial_name: str | None = None,
        lag_name: str | None = None,
    ):
        """Find the canonical vectors Wx, and Wy for the datasets X and Y.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            featureX_name (str, optional): Label for X-feature dimension,
                set to 'channel' by default.
            featureY_name (str, optional): Label for Y-feature dimension,
                set to 'channel' by default.
            trial_name (str, optional): Label for trial dimension, defaults to None.
                Embedding is done per trial, then trials are pooled.
            lag_name (str, optional): Label of the lag dimension if X is already
                temporally embedded. Its coords (in time units) are used as time lags,
                so time_lags must not be given at initialization. Defaults to None.
        """

        cite("biesmann_temporal_2010")
        cite("parkhomenko_sparse_2009")

        # Labels for feature directions
        self.featureX_name = featureX_name
        self.featureY_name = featureY_name
        self.trial_name = trial_name

        # Validates input data, including optional trial and lag dims
        X, Y = self.validate_inputs_fit(X, Y, trial_name, lag_name)

        # Validated time lags, from time_lags or from the pre-computed lag dim
        self.set_time_lags(X, lag_name)

        # Construct a time-embedded version of X (or take the pre-computed lags)
        X = embed_lags(X, self.time_lags, lag_name, self.lag_name)

        # Flatten (trial, sample) -> sample, after embedding so lags stay within trials.
        # Returns X as (lag, sample, featureX_name) and Y as (sample, featureY_name)
        if trial_name is not None:
            X = self.flatten_trials(X, trial_name)
            Y = self.flatten_trials(Y, trial_name)

        # Center and diagonal-scale data
        X, Y = self.normalization_fit(X, Y)

        # Concatenate over the feature dimension
        X_concat = xr.concat(X, dim=self.featureX_name)

        Wx, Wy = self.estimate_filters(X_concat.data, Y.data)

        # Convert back to xarray, unfolding Wx into (lag, feature, component)
        self.Wx, self.Wy = self.convert_filters_to_DataArray(Wx, Wy, X, Y)

        # Estimate optimal time lag
        self.estimate_optimal_shift(X, Y)


class StructuredSparseTCCA(MultimodalSourceDecompositionWithTemporalEmbedding):
    """Structured sparse Canonical Correlation Analysis (ssCCA) between X_emb and Y.

    The sstCCA algorithm is a temporally embedded extension of
    :cite:t:`chen_structure-constrained_2013`, and it assumes the underlying X and Y
    features are linked through a graph structure. It finds sparse (L1) vectors Wx, and
    Wy as the solution to the following constrained optimization problem:

    maximize    Wx^T Cxy Wy
    subject to  Wx^T Cx Wx = 1,  Wy^T Cy Wy = 1,
                ||Wx||_1 <= c1x,  ||Wy||_1 <= c1y,
                Wx^T Lx Wx <= c2x,  Wy^T Ly Wy <= c2y

    where Cx, Cy, and Cxy are the individual and cross-covariance matrices between X_emb
    and Y datasets. The second constraint is the standard L1-norm penalization term,
    while the last constraint incorporates local information of the spatial distribution
    of the features through the Laplacian matrices Lx and Ly. These terms encourage
    filter components that are linked on the graphical structure to have similar values,
    making them to vary smoothly across the graph. The c1x and c1y controls sparsity
    while c2x and c2y controls the relative importance of the graph structure.

    The temporally embedded matrix X_emb is constructed by concatenating time-shifted
    versions of the original X. Y is assumed to be the delayed signals with respect to X
    so time shifts are always positive.

    For the one-unit algorithm, first Cx and Cy are shrunk towards their scaled
    Laplacians, Cx <- (1 - gamma_x) Cx + gamma_x m_x Lx (same for Cy), with
    m_x = trace(Cx) / (Nx N_lags) and the shrinkage intensities either given or
    estimated with Ledoit-Wolf (:cite:t:`ledoit_well-conditioned_2004`). In the
    embedded space, Lx is copied over the diagonal blocks of Cx, one per lag. Then the
    filters are estimated by alternating between the linear sub-problems obtained when
    one of the filters is fixed (based on :cite:t:`parkhomenko_sparse_2009`):

    - Update Wx:
        maximize Wx^T cx - lambda_x ||Wx||_1  subject to  Wx^T Bx Wx = 1,
        with cx = Cxy Wy
    - Update Wy:
        maximize Wy^T cy - lambda_y ||Wy||_1  subject to  Wy^T By Wy = 1,
        with cy = Cxy^T Wx

    where the L1 thresholds are set relative to the largest entry of the linear term,
    lambda_x = 2 kappa_x ||cx||_inf (same for Wy). Each sub-problem is solved by
    solvers.solve_filter, either in the original space with Bx = Cx and By = Cy
    ("alternating" solver, truly sparse filters), or in the whitened space with
    Bx = By = I and K = Cx^(-1/2) Cxy Cy^(-1/2) in place of Cxy ("whitening" solver,
    faster but the filters are in general not sparse). See ElasticNetCCA for a detailed
    explanation of both solvers. Note that closely spaced lags of slow signals can make
    Cx nearly singular, in which case the "alternating" solver raises a ValueError and
    L2 regularization or the "whitening" solver should be used.

    Multiple components are obtained via a deflation method, subtracting from Cxy the
    rank-1 term rho Cx Wx Wy^T Cy on each iteration, with rho = Wx^T Cxy Wy (equivalent
    to subtracting from K its 1-rank approximation). The returned vectors Wx and Wy are
    ordered in descending order w.r.t. the canonical correlations (only approximately
    if L1 is used).

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between
            modalities.
        Lx (ndarray): Laplacian matrix for modality X, with shape (Nx, Nx). Only used
            if gamma_x is non-zero (see l2_reg).
        Ly (ndarray): Laplacian matrix for modality Y, with shape (Ny, Ny). Only used
            if gamma_y is non-zero (see l2_reg).
        time_lags (np.ndarray): Time lags for temporal embedding. Exclusive
            with lag_name in fit.
        l1_reg (float or list of floats): list containing the relative L1 thresholds
            kappa_x and kappa_y (see above), each in [0, 1). If a single float is
            provided, then kappa_x = kappa_y.
        l2_reg (float, str, or list): Shrinkage intensities gamma_x and gamma_y (see
            above), each a number in [0, 1], or "lw" / "auto" to estimate it with
            Ledoit-Wolf. If a single value is provided, then gamma_x = gamma_y.
        n_units (int, optional): Number of contiguous blocks of samples (units) used
            by Ledoit-Wolf to estimate the variance of the covariance. Defaults to
            None, i.e. the number of trials if trials are given in fit, otherwise one
            unit per sample (i.i.d.). Only used if l2_reg is "lw" / "auto".
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Relative tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance.
            Defaults to True.
        random_state (int, optional): Seed for the random initialization of the
            solver, making fits reproducible. Defaults to None (not reproducible).
        pls (bool): Whether to perform PLS regression. Defaults to False.
        shift_source (bool): Deprecated, will be removed in a future release.
            Whether transform shifts each X source by its optimal time shift (see
            estimate_optimal_shift), from being aligned with the Y sources to the time
            of X. The X sources are temporally filtered (convolved over the lags), so
            the shift only moves this slow, smoothed signal in time and does not
            recover the original fast activity of X. Shifted X sources are also no
            longer aligned with the Y sources. Defaults to False.
        solver (str): Space where the sub-problems are solved, either "alternating"
            (original space) or "whitening" (whitened space), see above. Defaults to
            "alternating".

    Attributes:
        N_components (int): Number of components extracted.
        Ntx (int): Number of samples (per trial, if any) of modality X during training.
        Nty (int): Number of samples (per trial, if any) of modality Y during training.
        Nkx (int): Number of trials of modality X during training (None if no trials).
        Nky (int): Number of trials of modality Y during training (None if no trials).
        N_samples (int): Total number of samples, pooled over trials (Ntx * Nkx).
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        N_features (int): Maximum number of components, i.e. the minimum of
            Nx * N_lags and Ny.
        time_lags (np.ndarray): Validated time lags used for the embedding.
        N_lags (int): Number of time lags.
        lag_name (str): Label of the lag dimension of Wx ("time_shift", unless a
            pre-embedded X is given in fit).
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for X.
        latent_featureY_name (str): Label of transformed latent space dimension for Y.
        gamma_x (float): Shrinkage intensity used for the L2 regularization of X,
            either given in l2_reg or estimated with Ledoit-Wolf (None for PLS).
        gamma_y (float): Shrinkage intensity used for the L2 regularization of Y,
            either given in l2_reg or estimated with Ledoit-Wolf (None for PLS).
        Wx (DataArray): Linear filters for dataset X with dimensions
            (lag_name, featureX_name, latent_featureX_name).
        Wy (DataArray): Linear filters for dataset Y with dimensions
            (featureY_name, latent_featureY_name).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
        optimal_shift (np.ndarray): Optimal time shift for each component.
    """

    def __init__(
        self,
        N_components: int = None,
        Lx=None,
        Ly=None,
        time_lags=None,
        l1_reg: float | list[float, float] = 0,
        l2_reg: float | str | list = 0,
        n_units: int | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
        shift_source: bool = False,
        pls: bool = False,
        solver: str = "alternating",
    ):
        super(StructuredSparseTCCA, self).__init__(
            N_components=N_components,
            time_lags=time_lags,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
            shift_source=shift_source,
        )

        # CCA or PLS
        self.pls = pls
        self.solver = validate_solver(solver)

        # Validate regularization parameters
        self.l1_reg = validate_l1_reg(l1_reg, relative=True)
        self.l2_reg = validate_l2_reg(l2_reg)
        self.n_units = n_units
        self.Lx = Lx
        self.Ly = Ly

        # The Laplacians only enter through the L2 shrinkage
        if (Lx is not None and not self.l2_reg[0]) or (
            Ly is not None and not self.l2_reg[1]
        ):
            warnings.warn(
                "Lx and Ly are ignored when the corresponding l2_reg is 0, since the "
                "graph structure only enters through the L2 shrinkage. Set a non-zero "
                "l2_reg to use them.",
                UserWarning,
                stacklevel=2,
            )

    def fit(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        featureX_name: str = "channel",
        featureY_name: str = "channel",
        trial_name: str | None = None,
        lag_name: str | None = None,
    ):
        """Find the canonical vectors Wx, and Wy for the datasets X and Y.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            featureX_name (str, optional): Label for X-feature dimension,
                set to 'channel' by default.
            featureY_name (str, optional): Label for Y-feature dimension,
                set to 'channel' by default.
            trial_name (str, optional): Label for trial dimension, defaults to None.
                Embedding is done per trial, then trials are pooled.
            lag_name (str, optional): Label of the lag dimension if X is already
                temporally embedded. Its coords (in time units) are used as time lags,
                so time_lags must not be given at initialization. Defaults to None.
        """

        cite("biesmann_temporal_2010")
        cite("chen_structure-constrained_2013")
        cite("parkhomenko_sparse_2009")
        # Labels for sample and feature directions
        self.featureX_name = featureX_name
        self.featureY_name = featureY_name

        self.trial_name = trial_name

        # Validates input data, including optional trial and lag dims
        X, Y = self.validate_inputs_fit(X, Y, trial_name, lag_name)

        # Validated time lags, from time_lags or from the pre-computed lag dim
        self.set_time_lags(X, lag_name)

        # Construct a time-embedded version of X (or take the pre-computed lags)
        X = embed_lags(X, self.time_lags, lag_name, self.lag_name)

        # Flatten (trial, sample) -> sample, after embedding so lags stay within trials.
        # Returns X as (lag, sample, featureX_name) and Y as (sample, featureY_name)
        if trial_name is not None:
            X = self.flatten_trials(X, trial_name)
            Y = self.flatten_trials(Y, trial_name)

        # Copy Laplace matrix over diagonals (without Lx, identity is used as in ssCCA)
        new_Lx = None
        if self.Lx is not None:
            new_Lx = np.zeros([self.Nx * self.N_lags, self.Nx * self.N_lags])
            for i in range(self.N_lags):
                segment = slice(i * self.Nx, (i + 1) * self.Nx)
                new_Lx[segment, segment] = self.Lx

        # Center and diagonal-scale data
        X, Y = self.normalization_fit(X, Y)

        # Concatenate over the feature dimension
        X_concat = xr.concat(X, dim=self.featureX_name)

        Wx, Wy = self.estimate_filters(X_concat.data, Y.data, Lx=new_Lx, Ly=self.Ly)

        # Convert back to xarray, unfolding Wx into (lag, feature, component)
        self.Wx, self.Wy = self.convert_filters_to_DataArray(Wx, Wy, X, Y)

        # Estimate optimal time lag
        self.estimate_optimal_shift(X, Y)


class tCCA(ElasticNetTCCA):
    """Perform tCCA between two datasets X and Y.

    This algorithm is a particular case of the one implemented in the ElasticNetTCCA
    class. See there for a detailed explanation of the algorithm. It uses the
    "whitening" solver, which without L1 gives the same filters as the "alternating"
    one, but is faster and robust to singular covariances.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between
            modalities.
        time_lags (np.ndarray): Time lags for temporal embedding. Exclusive
            with lag_name in fit.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Relative tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance.
            Defaults to True.
        random_state (int, optional): Seed for the random initialization of the
            solver, making fits reproducible. Defaults to None (not reproducible).
        shift_source (bool): Deprecated, will be removed in a future release.
            Whether transform shifts each X source by its optimal time shift (see
            estimate_optimal_shift), from being aligned with the Y sources to the time
            of X. The X sources are temporally filtered (convolved over the lags), so
            the shift only moves this slow, smoothed signal in time and does not
            recover the original fast activity of X. Shifted X sources are also no
            longer aligned with the Y sources. Defaults to False.

    Attributes:
        N_components (int): Number of components extracted.
        Ntx (int): Number of samples (per trial, if any) of modality X during training.
        Nty (int): Number of samples (per trial, if any) of modality Y during training.
        Nkx (int): Number of trials of modality X during training (None if no trials).
        Nky (int): Number of trials of modality Y during training (None if no trials).
        N_samples (int): Total number of samples, pooled over trials (Ntx * Nkx).
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        N_features (int): Maximum number of components, i.e. the minimum of
            Nx * N_lags and Ny.
        time_lags (np.ndarray): Validated time lags used for the embedding.
        N_lags (int): Number of time lags.
        lag_name (str): Label of the lag dimension of Wx ("time_shift", unless a
            pre-embedded X is given in fit).
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for X.
        latent_featureY_name (str): Label of transformed latent space dimension for Y.
        gamma_x (float): Shrinkage intensity of X, always 0 (no L2 regularization).
        gamma_y (float): Shrinkage intensity of Y, always 0 (no L2 regularization).
        Wx (DataArray): Linear filters for dataset X with dimensions
            (lag_name, featureX_name, latent_featureX_name).
        Wy (DataArray): Linear filters for dataset Y with dimensions
            (featureY_name, latent_featureY_name).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
        optimal_shift (np.ndarray): Optimal time shift for each component.
    """

    def __init__(
        self,
        N_components: int = None,
        time_lags: np.ndarray = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
        shift_source=False,
    ):
        super(tCCA, self).__init__(
            N_components=N_components,
            time_lags=time_lags,
            l1_reg=0,
            l2_reg=0,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
            pls=False,
            shift_source=shift_source,
            solver="whitening",
        )
