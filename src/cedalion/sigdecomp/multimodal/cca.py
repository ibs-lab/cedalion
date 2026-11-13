"""CCA-like models for multimodal data source decomposition.

Includes:
- ElasticNetCCA: CCA with L1 + L2 regularization
- StructuredSparseCCA: CCA with L1 + graph-structure regularization
- RidgeCCA: CCA with L2 regularization
- SparseCCA: CCA with L1 regularization
- CCA: Standard CCA
- SparsePLS: PLS with L1 regularization
- PLS: Standard PLS
"""

import warnings

import numpy as np
import xarray as xr

from cedalion import cite
from cedalion.sigdecomp.multimodal.utils import (
    validate_dimension_labels,
    validate_dimension_sizes,
    validate_sample_alignment,
    standardize,
    validate_l1_reg,
    validate_l2_reg,
    validate_solver,
    filter_to_pattern,
    shrink_cov,
    inv_sqrt_cov,
)
from cedalion.sigdecomp.multimodal.solvers import solve_filter


class MultimodalSourceDecomposition:
    """Base class for source decomposition models using linear filters.

    Multimodal data, X and Y, get decomposed into latent sources using linear filters.

    This main class is inherited by other source decomposition methods, such as
    ElasticNetCCA, ssCCA, and PLS. It implements methods to validate input dimensions,
    apply normalization, estimate the filters, transform data from two modalities using
    the learned filters, and build the spatial patterns of those filters.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features
            between modalities.
        max_iter: Maximum number of iterations for the algorithm.
        tol: Relative tolerance for convergence.
        scale: Whether to scale the data during normalization to unit variance.
            Defaults to True.
        random_state (int, optional): Seed for the random initialization of the
            solver, making fits reproducible. Defaults to None (not reproducible).
    """

    def __init__(
        self,
        N_components: int = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
    ):
        # General parameters
        self.N_components = N_components
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.random_state = random_state
        self.Ntx = None
        self.Nty = None
        self.Nx = None
        self.Ny = None
        self.N_features = None
        self.N_samples = None
        self.Nkx = None  # trial dim size (optional)
        self.Nky = None

        # Labels for sample, feature, and trial (optional) dimensions
        self.sample_name = None
        self.featureX_name = None
        self.featureY_name = None
        self.trial_name = None

        # Labels for latent space dimension
        self.latent_featureX_name = self.__class__.__name__ + "_X"
        self.latent_featureY_name = self.__class__.__name__ + "_Y"

        # Normalization parameters
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        # Filters
        self.Wx = None
        self.Wy = None

        # Regularization and solver used by estimate_filters (set by the subclasses)
        self.l1_reg = [0, 0]
        self.l2_reg = [0, 0]
        self.n_units = None
        self.pls = False
        self.solver = "alternating"

        # L2 shrinkage intensities (populated in fit)
        self.gamma_x = None
        self.gamma_y = None

        # Spatial patterns (populated by build_spatial_patterns)
        self.Ax = None
        self.Ay = None

    def validate_inputs_fit(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Validates inputs and returns it with the correct dimensions and labels.

        This method ensures that the input data to the fit function, X and Y, i.e. those
        used for training, have the expected dimension labels and sizes and returns them
        with the dimensions ordered as (sample_name, feature_name). Since samples are
        paired by position, X and Y must also share the same sample coordinates (see
        validate_sample_alignment). It also initializes the number of samples, and
        features. Optional trial and lag dims are placed first, and samples are counted
        over all trials.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
            trial_name (str, optional): Label for trial dimension of X and Y.
            lag_name (str, optional): Label for lag dimension of X.

        Returns:
            tuple: A tuple (X, Y) where:
                X (DataArray): Input for modality X. shape: ([lag_name], [trial_name],
                    sample_name, featureX_name)
                Y (DataArray): Input for modality Y. shape: ([trial_name], sample_name,
                    featureY_name)
        """

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

        # Define final number of features
        self.N_features = min(self.Nx, self.Ny)

        # Validate dimension sizes (only then return component and sample numbers)
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
        """Validates that the to-be-transformed data have the expected dims. and sizes.

        This method ensures that X and Y have the same dimension labels and number of
        features than the ones used during training. The number of time points and
        trials can differ from training, but must be the same between X and Y, and X
        and Y must share the same sample coordinates (see validate_sample_alignment).

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
            trial_name (str, optional): Label for trial dimension of X and Y.
            lag_name (str, optional): Label for lag dimension of X.

        Returns:
            tuple: A tuple (X, Y) where:
                X (DataArray): Input data for modality X ordered as
                    ([lag_name], [trial_name], sample_name, featureX_name).
                Y (DataArray): Input data for modality Y ordered as
                    ([trial_name], sample_name, featureY_name).
        """

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

        # Validate dimension sizes between modalities
        _ = validate_dimension_sizes(Ntx, Nty, Nkx=Nkx, Nky=Nky)

        # Check that X and Y are aligned, i.e. share the same sample coordinates
        validate_sample_alignment(X, Y, self.sample_name)

        # Validate features are the same as the ones used during training
        if Nx != self.Nx:
            raise ValueError(
                f"X should have {self.Nx} number of features, but {Nx} was found!"
            )
        if Ny != self.Ny:
            raise ValueError(
                f"Y should have {self.Ny} number of features, but {Ny} was found!"
            )

        return X, Y

    def flatten_trials(self, A: xr.DataArray, trial_name: str) -> xr.DataArray:
        """Collapse (trial, sample) into one sample dim. Raises if trial dim is missing.

        The original sample dim is first renamed to flat so the stacked dim can take
        the sample name without clashing with its own MultiIndex level. The stacked dim
        takes the position of the trial dim, so validated dim orders are kept.
        """
        if trial_name in A.dims:
            # Final dim order: as now, with sample replacing trial
            order = [d for d in A.dims if d != self.sample_name]
            order[order.index(trial_name)] = self.sample_name
            A = A.rename({self.sample_name: '_flat'})
            A = A.stack({self.sample_name: (trial_name, '_flat')}).transpose(*order)
        else:
            raise ValueError(f"Trial dimension '{trial_name}' not found in input data.")

        return A

    def unflatten_trials(self, A: xr.DataArray, ref: xr.DataArray) -> xr.DataArray:
        """Restore (trial, sample) dims from a sample dim produced by flatten_trials.

        The (trial, sample) dims take the position of the flattened sample dim (unstack
        alone would append them after all other dims), so e.g. sources come back as
        (trial, sample, component), as (sample, component) for continuous data.
        Coords of the original input `ref` (e.g. a trial_type label on the trial dim)
        are copied back, since unstack would otherwise broadcast them to 2D.
        """

        # Final dim order: as now, with (trial, sample) replacing the flattened sample
        trial_name = A.indexes[self.sample_name].names[0]
        order = []
        for d in A.dims:
            order += [trial_name, self.sample_name] if d == self.sample_name else [d]

        A = A.unstack(self.sample_name).rename({ '_flat': self.sample_name})
        A = A.transpose(*order)
        coords = {c: ref[c] for c in ref.coords if set(ref[c].dims) <= set(A.dims)}
        A = A.assign_coords(coords)

        return A

    def normalization_fit(
        self, X: xr.DataArray, Y: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Normalize input and save normalization parameters (mean and std) for later.

        This method centers and scales the data for both modalities along the sample
        dimension. It computes the mean and standard deviation for X and Y using the
        provided standardization function, updating the corresponding class attributes.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.

        Returns:
            tuple: A tuple (X, Y) of standardized data arrays for modalities X and Y.
        """
        # Center and diagonal-scale data
        X, self.x_mean, self.x_std = standardize(
            X, dim=self.sample_name, scale=self.scale
        )
        Y, self.y_mean, self.y_std = standardize(
            Y, dim=self.sample_name, scale=self.scale
        )

        return X, Y

    def normalization_transform(
        self, X: xr.DataArray, Y: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Applies normalization input data using trained parameters.

        This method standardizes the input data arrays X and Y using the normalization
        parameters (mean and standard deviation) computed during the fitting process.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.

        Returns:
            tuple: A tuple (X, Y) of normalized data arrays.
        """
        # Center and diagonal-scale data
        X = (X - self.x_mean) / self.x_std
        Y = (Y - self.y_mean) / self.y_std

        return X, Y

    def convert_filters_to_DataArray(
        self, Wx: np.ndarray, Wy: np.ndarray, X: xr.DataArray, Y: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Convert filters Wx and Wy from numpy to DataArray with right dims and coords.

        Args:
            Wx (ndarray): Filter matrix for modality X with shape (Nx, N_components).
            Wy (ndarray): Filter matrix for modality Y with shape (Ny, N_components).
            X (DataArray): DataArray containing the features of modality X.
            Y (DataArray): DataArray containing the features of modality Y.

        Returns:
            A tuple (Wx_xr, Wy_xr) containing the DataArray versions of Wx and Wy,
                respectively.
        """

        component_name_x = [f"Sx{i + 1}" for i in range(self.N_components)]
        component_name_y = [f"Sy{i + 1}" for i in range(self.N_components)]

        coords_x = {d: X[d] for d in X.dims if d != self.sample_name}
        coords_x[self.latent_featureX_name] = component_name_x
        Wx_xr = xr.DataArray(Wx, coords=coords_x)

        coords_y = {d: Y[d] for d in Y.dims if d != self.sample_name}
        coords_y[self.latent_featureY_name] = component_name_y
        Wy_xr = xr.DataArray(Wy, coords=coords_y)

        return Wx_xr, Wy_xr

    def estimate_filters(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Lx: np.ndarray = None,
        Ly: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate the canonical vectors Wx, and Wy for the datasets X and Y.

        Main method that estimates the canonical vectors Wx, and Wy for the normalized
        datasets X and Y using an Elastic Net CCA algorithm with the option of
        incorporating Laplace matrices, Lx and Ly that transforms the algorithm into a
        structured sparse CCA. It assumes X and Y have shapes (samples, features).

        The filters are estimated one component at a time, alternating between the
        linear sub-problems for Wx and Wy, each solved by solvers.solve_filter. With the
        "alternating" solver the sub-problems are solved in the original space, with
        the (shrunk) covariances as constraint metrics, while with the "whitening"
        solver they are solved in the whitened space, where the metrics are the
        identity.

        The algorithm is configured by the attributes N_components, max_iter, tol,
        l1_reg, l2_reg, n_units (the number of trials Nkx if None), pls, solver, and
        random_state. The shrinkage intensities used for Cx and Cy are stored in
        gamma_x and gamma_y (0 without L2 regularization, None for PLS). The
        "alternating" solver raises a ValueError if Cx or Cy is singular, and a
        UserWarning is raised if, without L2 regularization, the ranks of X and Y add
        up to more than the number of samples minus one (trivial correlations).

        Args:
            X (ndarray): Input data for modality X with shape (Nt, Nx).
            Y (ndarray): Input data for modality Y with shape (Nt, Ny).
            Lx (ndarray, optional): Laplacian matrix for modality X, used as shrinkage
                target. Defaults to None (identity).
            Ly (ndarray, optional): Laplacian matrix for modality Y, used as shrinkage
                target. Defaults to None (identity).

        Returns:
            tuple: A tuple (Wx, Wy) with the canonical vectors for X and Y, ordered by
            descending canonical correlations (only approximately if L1 is used).
        """

        solver = self.solver
        l1_reg, l2_reg = self.l1_reg, self.l2_reg

        # Number of independent "units" for LW (trials if not None, otherwise raw samples)
        n_units = self.n_units if self.n_units is not None else self.Nkx

        Nt = X.shape[0]

        # Covariance matrices
        Cxy = (X.T @ Y) / (Nt - 1)

        # CCA or PLS (no shrinkage, as Cx and Cy are the identity, i.e. None for the
        # solver)
        self.gamma_x = self.gamma_y = None
        Cxx = Cyy = None
        if not self.pls:
            Cxx = (X.T @ X) / (Nt - 1)
            Cyy = (Y.T @ Y) / (Nt - 1)

            # L2 penalty (fixed or LW estimated) as shrinkage towards (scaled) I or L
            self.gamma_x = self.gamma_y = 0.0
            if l2_reg[0]:
                Cxx, self.gamma_x = shrink_cov(
                    Cxx, l2_reg[0], L=Lx, X=X, n_units=n_units
                )
            if l2_reg[1]:
                Cyy, self.gamma_y = shrink_cov(
                    Cyy, l2_reg[1], L=Ly, X=Y, n_units=n_units
                )

            # Rank of Cx and Cy: inverted by alternating solver => error if singular
            ranks = []
            for C, name in ((Cxx, "X"), (Cyy, "Y")):
                eigvals = np.linalg.eigvalsh(C)
                if solver == "alternating" and eigvals[0] < 1e-10 * eigvals[-1]:
                    raise ValueError(
                        f"The covariance of {name} is singular (rank-deficient data), "
                        "so the 'alternating' solver is not reliable. Use a non-zero "
                        "l2_reg, or the 'whitening' solver."
                    )
                ranks.append(np.sum(eigvals > 1e-10 * eigvals[-1]))

            # Warm inconsistent rank when no L2
            if not (self.gamma_x or self.gamma_y) and sum(ranks) > Nt - 1:
                warnings.warn(
                    f"The ranks of X ({ranks[0]}) and Y ({ranks[1]}) add up to more "
                    f"than the number of samples minus one ({Nt - 1}), so the leading "
                    "canonical correlations are trivially 1 (overfitting). Use a "
                    "non-zero l2_reg, or reduce the number of features.",
                    UserWarning,
                    stacklevel=3,
                )

        # Whitening: solve in the whitened space, where the metrics are the identity
        whiten = solver == "whitening" and not self.pls
        if whiten:
            Cxx_inv_sqrt, Cyy_inv_sqrt = inv_sqrt_cov(Cxx), inv_sqrt_cov(Cyy)
            Cxy = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
            Cxx = Cyy = None

        # L1 sub-problems: If Identify metric-> soft-thresholding, otherwise FISTA
        if l1_reg[0] or l1_reg[1]:
            cite("beck_fast_2009" if Cxx is not None else "witten_penalized_2009")

        # A new generator on each fit, so every fit is reproducible on its own
        rng = np.random.default_rng(self.random_state)

        Wx = np.zeros((X.shape[1], self.N_components))
        Wy = np.zeros((Y.shape[1], self.N_components))
        for k in range(self.N_components):

            wx = np.zeros(X.shape[1])
            wy = rng.random(Y.shape[1])

            for _ in range(self.max_iter):
                wx_old, wy_old = wx, wy

                # Alternate one filter at a time, each step being a linear sub-problem
                wx = solve_filter(c=Cxy @ wy, B=Cxx, kappa=l1_reg[0], w_init=wx_old)
                wy = solve_filter(c=Cxy.T @ wx, B=Cyy, kappa=l1_reg[1], w_init=wy_old)

                # Check convergence via relative change in wx and wy
                tol_x = self.tol * np.linalg.norm(wx)
                tol_y = self.tol * np.linalg.norm(wy)
                condition_norm_x = np.linalg.norm(wx - wx_old) < tol_x
                condition_norm_y = np.linalg.norm(wy - wy_old) < tol_y
                if condition_norm_x and condition_norm_y:
                    break

            # Deflation: rank-1 subtraction of the whitened K, mapped back to Cxy
            rho = wx @ Cxy @ wy
            bx = wx if Cxx is None else Cxx @ wx
            by = wy if Cyy is None else Cyy @ wy
            Cxy = Cxy - rho * np.outer(bx, by)

            # Store component
            Wx[:, k] = wx
            Wy[:, k] = wy

        # Project back to original unwhitened space
        if whiten:
            Wx = Cxx_inv_sqrt @ Wx
            Wy = Cyy_inv_sqrt @ Wy

        return Wx, Wy

    def to_filter_space(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Bring X and Y to the space in which the filters Wx and Wy were learned.

        Inputs are validated against the training data, trials (if any) are flattened
        into the sample dim, and data are normalized with the training parameters.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
            trial_name (str, optional): Label for trial dimension of X and Y.
            lag_name (str, optional): Label for lag dimension of X.

        Returns:
            tuple: A tuple (X, Y) of normalized data with dims (..., sample_name,
                feature_name).
        """
        X, Y = self.validate_inputs_transform(X, Y, trial_name, lag_name)

        # Flatten (trial, sample) -> sample
        if trial_name is not None:
            X = self.flatten_trials(X, trial_name)
            Y = self.flatten_trials(Y, trial_name)

        X, Y = self.normalization_transform(X, Y)

        return X, Y

    def build_spatial_patterns(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        trial_name: str | None = None,
        lag_name: str | None = None,
    ):
        """Compute the spatial patterns Ax and Ay of the learned filters Wx and Wy.

        Following :cite:t:`haufe_interpretation_2014`, the patterns are
        A = Cov(X) W Cov(S)^{-1}, with S = X W the sources, computed with X and Y in the
        same space as during training. Ax and Ay have the same dims and coords as Wx
        and Wy, respectively.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
            trial_name (str, optional): Label for trial dimension of X and Y.
            lag_name (str, optional): Label of the lag dimension if X is already
                temporally embedded (only for temporally embedded models).
        """
        cite("haufe_interpretation_2014")

        X, Y = self.to_filter_space(X, Y, trial_name, lag_name)
        self.Ax = filter_to_pattern(
            self.Wx, X, self.latent_featureX_name, self.sample_name
        )
        self.Ay = filter_to_pattern(
            self.Wy, Y, self.latent_featureY_name, self.sample_name
        )

    def transform(
        self, X: xr.DataArray, Y: xr.DataArray, trial_name: str | None = None
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Apply the linear transformation on the input data using learnt filters.

        This method validates the dimension labels and sizes of the input data to ensure
        consistency with the training data, applies normalization using the stored
        parameters, and then projects the normalized data onto a lower-dimensional space
        using the learned filters Wx and Wy. It retrieves the transformed arrays, a.k.a
        reconstructed sources.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            trial_name (str, optional): Label for trial dimension. Independent of the
                one used in fit, so continuous and trial data can be mixed.

        Returns:
            tuple: A tuple (X_new, Y_new) where:
                X_new (DataArray): Transformed data for modality X.
                Y_new (DataArray): Transformed data for modality Y.
        """

        # Validate, flatten trials and normalize as during training
        X_in, Y_in = X, Y
        X, Y = self.to_filter_space(X, Y, trial_name)

        # Project
        X_new = X @ self.Wx
        Y_new = Y @ self.Wy

        # Restore the trial axis -> (trial, sample, component)
        if trial_name is not None:
            X_new = self.unflatten_trials(X_new, X_in)
            Y_new = self.unflatten_trials(Y_new, Y_in)

        return X_new, Y_new


class ElasticNetCCA(MultimodalSourceDecomposition):
    """Perform Elastic Net Canonical Correlation Analysis (CCA) between two datasets.

    Apply CCA with L1 + L2 regularization, a.k.a elastic net. The algorithm finds sparse
    (L1) and normalized (L2) vectors Wx, and Wy as the solution to the following
    constrained optimization problem:

    maximize    Wx^T Cxy Wy
    subject to  Wx^T Cx Wx = 1,  Wy^T Cy Wy = 1,
                ||Wx||_1 <= c1x,  ||Wy||_1 <= c1y,
                ||Wx||^2_2 <= c2x,  ||Wy||^2_2 <= c2y

    where Cx, Cy, and Cxy are the individual and cross-covariance matrices between X and
    Y datasets, and the last four constraints correspond to the standard L1-norm and
    L2-norm penalization terms.
    c1x and c1y controls sparsity while c2x and c2y controls the magnitude of the
    vectors. PLS algorithms are also captured by this algorithm by sending Cx and Cy to
    the identity matrices.

    For the one-unit algorithm, the filters are estimated by alternating between the
    linear sub-problems obtained when one of the filters is fixed:

    - Update Wx:
        maximize Wx^T cx - lambda_x ||Wx||_1  subject to  Wx^T Bx Wx = 1,
        with cx = Cxy Wy
    - Update Wy:
        maximize Wy^T cy - lambda_y ||Wy||_1  subject to  Wy^T By Wy = 1,
        with cy = Cxy^T Wx

    Each sub-problem is solved by solvers.solve_filter. Without L1, the solution is
    given in closed form, Wx ∝ Bx^(-1) cx (same for Wy). With L1, it is found with
    FISTA (:cite:t:`beck_fast_2009`), or with a single soft-thresholding if Bx is the
    identity (:cite:t:`parkhomenko_sparse_2009` and :cite:t:`witten_penalized_2009`),
    which brings some components to zero. The L1 thresholds are set relative
    to the largest entry of the linear term, lambda_x = 2 kappa_x ||cx||_inf (same for
    Wy). Two solvers are available, differing in the space where the sub-problems are
    solved:

    - "alternating": in the original space, with Bx = Cx and By = Cy. The L1 penalty
      acts on the filters Wx and Wy themselves, so they are truly sparse. It requires
      Cx and Cy to be positive definite, otherwise a ValueError is raised (use L2 if
      the data are rank-deficient, e.g. average-referenced EEG, or there are more
      features than samples).
    - "whitening": in the whitened space, u = Cx^(1/2) Wx and v = Cy^(1/2) Wy, where
      Bx = By = I and Cxy is replaced by K = Cx^(-1/2) Cxy Cy^(-1/2). The updates reduce
      to the standard alternating power method for the SVD of K, with a
      soft-thresholding step if L1 is used (:cite:t:`parkhomenko_sparse_2009`), and the
      filters are recovered as Wx = Cx^(-1/2) u and Wy = Cy^(-1/2) v. It is faster, but
      the L1 penalty acts on u and v, so Wx and Wy are in general not sparse.

    For PLS, Cx and Cy are the identity, so both solvers coincide.

    If L2 regularization is used, Cx and Cy are first shrunk towards a scaled identity,
    Cx <- (1 - gamma_x) Cx + gamma_x m_x I (same for Cy), with m_x = trace(Cx) / Nx. The
    shrinkage intensities are either given or estimated with Ledoit-Wolf
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
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Relative tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance.
            Defaults to True.
        random_state (int, optional): Seed for the random initialization of the
            solver, making fits reproducible. Defaults to None (not reproducible).
        pls (bool): Whether to perform PLS regression. Defaults to False.
        solver (str): Space where the sub-problems are solved, either "alternating"
            (original space) or "whitening" (whitened space), see above. Defaults to
            "alternating".

    Attributes:
        N_components (int): Number of components extracted.
        Ntx (int): Number of samples (per trial, if any) of modality X during training.
        Nty (int): Number of samples (per trial, if any) of modality Y during training.
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        Nkx (int): Number of trials of modality X during training (None if no trials).
        Nky (int): Number of trials of modality Y during training (None if no trials).
        N_features (int): Number of features in X and Y, i.e. the minimum of Nx and Ny.
        N_samples (int): Total number of samples. Must be the same in X and Y.
            It coincides with Ntx = Nty if no trial dimension is present, otherwise
            pooled over all trials, i.e. Ntx * Nkx = Nty * Nky.
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for
            X modality.
        latent_featureY_name (str): Label of transformed latent space dimension for
            Y modality.
        gamma_x (float): Shrinkage intensity used for the L2 regularization of X,
            either given in l2_reg or estimated with Ledoit-Wolf (None for PLS).
        gamma_y (float): Shrinkage intensity used for the L2 regularization of Y,
            either given in l2_reg or estimated with Ledoit-Wolf (None for PLS).
        Wx (DataArray): Linear filters for dataset X with dimensions
            (featureX_name, latent_featureX_name)
        Wy (DataArray): Linear filters for dataset Y with dimensions
            (featureY_name, latent_featureY_name).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
    """

    def __init__(
        self,
        N_components: int = None,
        l1_reg: float | list[float, float] = 0,
        l2_reg: float | str | list = 0,
        n_units: int | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
        pls: bool = False,
        solver: str = "alternating",
    ):
        super(ElasticNetCCA, self).__init__(
            N_components=N_components,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
        )

        # CCA or PLS
        self.pls = pls

        # Validate solver
        self.solver = validate_solver(solver)

        # Validate regularization parameters
        self.l1_reg = validate_l1_reg(l1_reg, relative=True)
        self.l2_reg = validate_l2_reg(l2_reg)
        self.n_units = n_units

    def fit(
        self,
        X: xr.DataArray,
        Y: xr.DataArray,
        sample_name: str = "time",
        featureX_name: str = "channel",
        featureY_name: str = "channel",
        trial_name: str | None = None,
    ):
        """Find the canonical vectors Wx, and Wy for the datasets X and Y.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            sample_name (str, optional): Label for sample dimension, defaults to 'time'.
            featureX_name (str, optional): Label for X-feature dimension,
                set to 'channel' by default.
            featureY_name (str, optional): Label for Y-feature dimension,
                set to 'channel' by default.
            trial_name (str, optional): Label for trial dimension, defaults to None.
        """

        cite("parkhomenko_sparse_2009")
        # Labels for sample and feature directions
        self.sample_name = sample_name
        self.featureX_name = featureX_name
        self.featureY_name = featureY_name
        self.trial_name = trial_name

        # Validates and order input data: ([trial_name], sample_name, feature_name)
        X, Y = self.validate_inputs_fit(X, Y, trial_name)

        # Flatten (trial, sample) -> sample.
        if trial_name is not None:
            X = self.flatten_trials(X, trial_name)
            Y = self.flatten_trials(Y, trial_name)

        # Center and diagonal-scale data
        X, Y = self.normalization_fit(X, Y)

        # Run main algorithm for filter estimation
        Wx, Wy = self.estimate_filters(X.data, Y.data)

        # Convert back to xarray with the right dimensions and coordinates
        self.Wx, self.Wy = self.convert_filters_to_DataArray(Wx, Wy, X, Y)


class StructuredSparseCCA(MultimodalSourceDecomposition):
    """Structured sparse Canonical Correlation Analysis (ssCCA) between two datasets.

    The ssCCA algorithm is based on :cite:t:`chen_structure-constrained_2013` and it
    assumes the underlying X and Y features are linked through a graph structure. It
    finds sparse (L1) vectors Wx, and Wy as the solution to the following constrained
    optimization problem:

    maximize    Wx^T Cxy Wy
    subject to  Wx^T Cx Wx = 1,  Wy^T Cy Wy = 1,
                ||Wx||_1 <= c1x,  ||Wy||_1 <= c1y,
                Wx^T Lx Wx <= c2x,  Wy^T Ly Wy <= c2y

    where Cx, Cy, and Cxy are the individual and cross-covariance matrices between X and
    Y datasets. The second constraint is the standard L1-norm penalization term, while
    the last constraint incorporates local information of the spatial distribution of
    the features through the Laplacian matrices Lx and Ly. These terms encourage filter
    components that are linked on the graphical structure to have similar values, making
    them to vary smoothly across the graph. The c1x and c1y controls sparsity while c2x
    and c2y controls the relative importance of the graph structure.

    For the one-unit algorithm, first Cx and Cy are shrunk towards their scaled
    Laplacians, Cx <- (1 - gamma_x) Cx + gamma_x m_x Lx (same for Cy), with
    m_x = trace(Cx) / Nx and the shrinkage intensities either given or estimated with
    Ledoit-Wolf (:cite:t:`ledoit_well-conditioned_2004`), and then the filters are
    estimated by alternating between the linear sub-problems obtained when one of the
    filters is fixed (based on :cite:t:`parkhomenko_sparse_2009`):

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
    explanation of both solvers.

    Multiple components are obtained via a deflation method, subtracting from Cxy the
    rank-1 term rho Cx Wx Wy^T Cy on each iteration, with rho = Wx^T Cxy Wy (equivalent
    to subtracting from K its 1-rank approximation). The returned vectors Wx and Wy are
    ordered in descending order w.r.t. the canonical correlations (only approximately
    if L1 is used).

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between
            modalities.
        Lx (ndarray): Laplacian matrix for modality X. Only used if gamma_x is
            non-zero (see l2_reg).
        Ly (ndarray): Laplacian matrix for modality Y. Only used if gamma_y is
            non-zero (see l2_reg).
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
        N_features (int): Number of features in X and Y, i.e. the minimum of Nx and Ny.
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
            (featureX_name, latent_featureX_name)
        Wy (DataArray): Linear filters for dataset Y with dimensions
            (featureY_name, latent_featureY_name).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
    """

    def __init__(
        self,
        N_components: int = None,
        Lx: np.ndarray = None,
        Ly: np.ndarray = None,
        l1_reg: float | list[float, float] = 0,
        l2_reg: float | str | list = 0,
        n_units: int | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
        pls: bool = False,
        solver: str = "alternating",
    ):
        super(StructuredSparseCCA, self).__init__(
            N_components=N_components,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
        )

        # Validate regularization parameters
        self.l1_reg = validate_l1_reg(l1_reg, relative=True)
        self.l2_reg = validate_l2_reg(l2_reg)
        self.n_units = n_units
        self.Lx = Lx
        self.Ly = Ly
        self.pls = pls
        self.solver = validate_solver(solver)

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
        sample_name: str = "time",
        featureX_name: str = "channel",
        featureY_name: str = "channel",
        trial_name: str | None = None,
    ):
        """Find the canonical vectors Wx, and Wy for the datasets X and Y.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            sample_name (str, optional): Label for sample dimension,
                set to 'time' by default.
            featureX_name (str, optional): Label for X-feature dimension,
                set to 'channel' by default.
            featureY_name (str, optional): Label for Y-feature dimension,
                set to 'channel' by default.
            trial_name (str, optional): Label for trial dimension, defaults to None.
        """

        cite("chen_structure-constrained_2013")
        cite("parkhomenko_sparse_2009")
        # Labels for sample and feature directions
        self.sample_name = sample_name
        self.featureX_name = featureX_name
        self.featureY_name = featureY_name
        self.trial_name = trial_name

        # Validates input data and returns it ordered as ([trial_name], sample_name,
        # feature_name)
        X, Y = self.validate_inputs_fit(X, Y, trial_name)

        # Flatten (trial, sample) -> sample. No-op for continuous input
        if trial_name is not None:
            X = self.flatten_trials(X, trial_name)
            Y = self.flatten_trials(Y, trial_name)

        # Center and diagonal-scale data
        X, Y = self.normalization_fit(X, Y)

        Wx, Wy = self.estimate_filters(X.data, Y.data, Lx=self.Lx, Ly=self.Ly)

        # Convert back to xarray with the right dimensions and coordinates
        self.Wx, self.Wy = self.convert_filters_to_DataArray(Wx, Wy, X, Y)


class RidgeCCA(ElasticNetCCA):
    """Perform CCA between two datasets X and Y with L2 regularization, a.k.a ridge CCA.

    This algorithm is a particular case of the one implemented in the ElasticNetCCA
    class. See there for a detailed explanation of the algorithm. It uses the
    "whitening" solver, which without L1 gives the same filters as the "alternating"
    one, but is faster and robust to singular covariances.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between
            modalities.
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

    Attributes:
        N_components (int): Number of components extracted.
        Ntx (int): Number of samples (per trial, if any) of modality X during training.
        Nty (int): Number of samples (per trial, if any) of modality Y during training.
        Nkx (int): Number of trials of modality X during training (None if no trials).
        Nky (int): Number of trials of modality Y during training (None if no trials).
        N_samples (int): Total number of samples, pooled over trials (Ntx * Nkx).
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        N_features (int): Number of features in X and Y, i.e. the minimum of Nx and Ny.
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
            (featureX_name, latent_featureX_name)
        Wy (DataArray): Linear filters for dataset Y with dimensions
            (featureY_name, latent_featureY_name).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
    """

    def __init__(
        self,
        N_components: int = None,
        l2_reg: float | str | list = 0,
        n_units: int | None = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
    ):
        super(RidgeCCA, self).__init__(
            N_components=N_components,
            l1_reg=0,
            l2_reg=l2_reg,
            n_units=n_units,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
            pls=False,
            solver="whitening",
        )


class SparseCCA(ElasticNetCCA):
    """Perform Sparse CCA between two datasets X and Y with L1 regularization.

    This is a.k.a. sparse CCA, based on :cite:t:`parkhomenko_sparse_2009`.

    This algorithm is a particular case of the one implemented in the ElasticNetCCA
    class. See there for a detailed explanation of the algorithm.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between
            modalities.
        l1_reg (float or list of floats): list containing the relative L1 thresholds
            kappa_x and kappa_y, each in [0, 1). If a single float is provided, then
            kappa_x = kappa_y.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Relative tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance.
            Defaults to True.
        random_state (int, optional): Seed for the random initialization of the
            solver, making fits reproducible. Defaults to None (not reproducible).

    Attributes:
        N_components (int): Number of components extracted.
        Ntx (int): Number of samples (per trial, if any) of modality X during training.
        Nty (int): Number of samples (per trial, if any) of modality Y during training.
        Nkx (int): Number of trials of modality X during training (None if no trials).
        Nky (int): Number of trials of modality Y during training (None if no trials).
        N_samples (int): Total number of samples, pooled over trials (Ntx * Nkx).
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        N_features (int): Number of features in X and Y, i.e. the minimum of Nx and Ny.
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for X.
        latent_featureY_name (str): Label of transformed latent space dimension for Y.
        gamma_x (float): Shrinkage intensity of X, always 0 (no L2 regularization).
        gamma_y (float): Shrinkage intensity of Y, always 0 (no L2 regularization).
        Wx (DataArray): Linear filters for dataset X with dimensions
            (featureX_name, latent_featureX_name)
        Wy (DataArray): Linear filters for dataset Y with dimensions
            (featureY_name, latent_featureY_name).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
    """

    def __init__(
        self,
        N_components: int = None,
        l1_reg: float | list[float, float] = 0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
    ):
        super(SparseCCA, self).__init__(
            N_components=N_components,
            l1_reg=l1_reg,
            l2_reg=0,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
            pls=False,
        )


class CCA(ElasticNetCCA):
    """Perform CCA between two datasets X and Y.

    This algorithm is a particular case of the one implemented in the ElasticNetCCA
    class. See there for a detailed explanation of the algorithm. It uses the
    "whitening" solver, which without L1 gives the same filters as the "alternating"
    one, but is faster and robust to singular covariances.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between
            modalities.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Relative tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance.
            Defaults to True.
        random_state (int, optional): Seed for the random initialization of the
            solver, making fits reproducible. Defaults to None (not reproducible).

    Attributes:
        N_components (int): Number of components extracted.
        Ntx (int): Number of samples (per trial, if any) of modality X during training.
        Nty (int): Number of samples (per trial, if any) of modality Y during training.
        Nkx (int): Number of trials of modality X during training (None if no trials).
        Nky (int): Number of trials of modality Y during training (None if no trials).
        N_samples (int): Total number of samples, pooled over trials (Ntx * Nkx).
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        N_features (int): Number of features in X and Y, i.e. the minimum of Nx and Ny.
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for X.
        latent_featureY_name (str): Label of transformed latent space dimension for Y.
        gamma_x (float): Shrinkage intensity of X, always 0 (no L2 regularization).
        gamma_y (float): Shrinkage intensity of Y, always 0 (no L2 regularization).
        Wx (DataArray): Linear filters for dataset X with dimensions
            (featureX_name, latent_featureX_name)
        Wy (DataArray): Linear filters for dataset Y with dimensions
            (featureY_name, latent_featureY_name).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
    """

    def __init__(
        self,
        N_components: int = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
    ):
        super(CCA, self).__init__(
            N_components=N_components,
            l1_reg=0,
            l2_reg=0,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
            pls=False,
            solver="whitening",
        )


class SparsePLS(ElasticNetCCA):
    """Perform Partial Least Squares (PLS) between two datasets with L1 regularization.

    This is a.k.a. sparse PLS, based on a combination from
    :cite:t:`parkhomenko_sparse_2009` and :cite:t:`witten_penalized_2009`.

    In Witten's paper, the algorithm is presented as a particular case of their
    Penalized Matrix Decomposition (PMD) method, called PMD(L1, L1) or Sparse CCA.
    However, the latter name is misleading since in this problem we use identity
    matrices for the L2-norm constraints, rather than correlation matrices. That
    difference makes the method truly a SparsePLS one. Here, Witten's method is modified
    by adding normalization on each iteration and setting the L1 thresholds relative
    to the largest entry of the linear term, kappa * ||c||_inf.

    This algorithm is a particular case of the one implemented in the ElasticNetCCA
    class. See there for a detailed explanation of the algorithm.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between
            modalities.
        l1_reg (float or list of floats): list containing the relative L1 thresholds
            kappa_x and kappa_y, each in [0, 1). If a single float is provided, then
            kappa_x = kappa_y.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Relative tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance.
            Defaults to True.
        random_state (int, optional): Seed for the random initialization of the
            solver, making fits reproducible. Defaults to None (not reproducible).

    Attributes:
        N_components (int): Number of components extracted.
        Ntx (int): Number of samples (per trial, if any) of modality X during training.
        Nty (int): Number of samples (per trial, if any) of modality Y during training.
        Nkx (int): Number of trials of modality X during training (None if no trials).
        Nky (int): Number of trials of modality Y during training (None if no trials).
        N_samples (int): Total number of samples, pooled over trials (Ntx * Nkx).
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        N_features (int): Number of features in X and Y, i.e. the minimum of Nx and Ny.
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for X.
        latent_featureY_name (str): Label of transformed latent space dimension for Y.
        gamma_x (float): Shrinkage intensity of X, always None (Cx is not used in PLS).
        gamma_y (float): Shrinkage intensity of Y, always None (Cy is not used in PLS).
        Wx (DataArray): Linear filters for dataset X with dimensions
            (featureX_name, latent_featureX_name)
        Wy (DataArray): Linear filters for dataset Y with dimensions
            (featureY_name, latent_featureY_name).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
    """

    def __init__(
        self,
        N_components: int = None,
        l1_reg: float | list[float, float] = 0,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
    ):
        super(SparsePLS, self).__init__(
            N_components=N_components,
            l1_reg=l1_reg,
            l2_reg=0,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
            pls=True,
        )


class PLS(SparsePLS):
    """Perform PLS between two datasets X and Y.

    This algorithm is a particular case of the one implemented in the SparsePLS class
    when no penalty is imposed. See there for a detailed explanation of the algorithm.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between
            modalities.
        max_iter (int): Maximum number of iterations for the algorithm. Defaults to 1000
        tol (float): Relative tolerance for convergence. Defaults to 1e-6.
        scale (bool): Whether to scale the data during normalization to unit variance.
            Defaults to True.
        random_state (int, optional): Seed for the random initialization of the
            solver, making fits reproducible. Defaults to None (not reproducible).

    Attributes:
        N_components (int): Number of components extracted.
        Ntx (int): Number of samples (per trial, if any) of modality X during training.
        Nty (int): Number of samples (per trial, if any) of modality Y during training.
        Nkx (int): Number of trials of modality X during training (None if no trials).
        Nky (int): Number of trials of modality Y during training (None if no trials).
        N_samples (int): Total number of samples, pooled over trials (Ntx * Nkx).
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        N_features (int): Number of features in X and Y, i.e. the minimum of Nx and Ny.
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for X.
        latent_featureY_name (str): Label of transformed latent space dimension for Y.
        gamma_x (float): Shrinkage intensity of X, always None (Cx is not used in PLS).
        gamma_y (float): Shrinkage intensity of Y, always None (Cy is not used in PLS).
        Wx (DataArray): Linear filters for dataset X with dimensions
            (featureX_name, latent_featureX_name)
        Wy (DataArray): Linear filters for dataset Y with dimensions
            (featureY_name, latent_featureY_name).
        Ax (DataArray): Spatial patterns of Wx, with its same dimensions. Populated
            by build_spatial_patterns.
        Ay (DataArray): Spatial patterns of Wy, with its same dimensions. Populated
            by build_spatial_patterns.
    """

    def __init__(
        self,
        N_components: int = None,
        max_iter: int = 1000,
        tol: float = 1e-6,
        scale: bool = True,
        random_state: int | None = None,
    ):
        super(PLS, self).__init__(
            N_components=N_components,
            l1_reg=0,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            random_state=random_state,
        )

