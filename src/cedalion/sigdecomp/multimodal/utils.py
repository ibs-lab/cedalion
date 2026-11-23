"""Utility functions for multimodal models."""

import numpy as np
import xarray as xr

from cedalion import cite


def validate_dimension_labels(X : xr.DataArray, 
                              Y : xr.DataArray,
                              sample_name : str,
                              featureX_name : str,
                              featureY_name : str,
                              trial_name : str | None = None,
                              lag_name : str | None = None):
    """Validates that X and Y datasets contain the expected dimension labels.

    This method checks that the data arrays X and Y contain the same
    sample label and corresponding X and Y feature labels. Optionally, both can
    share a trial label, and X can have a lag label (for pre-embedded inputs).

    Args:
        X (DataArray): Input data for modality X. Expected to have dimensions
            ([lag_name], [trial_name], sample_name, featureX_name).
        Y (DataArray): Input data for modality Y. Expected to have dimensions
            ([trial_name], sample_name, featureY_name).
        sample_name (str): Name of the sample dimension.
        featureX_name (str): Name of the feature dimension for X.
        featureY_name (str): Name of the feature dimension for Y.
        trial_name (str, optional): Name of the trial dimension of X and Y.
        lag_name (str, optional): Name of the lag dimension of X.

    Raises:
        ValueError: If X or Y are missing the expected feature dimension labels.
    """

    # Catch missing feature dimensions
    X_dims = (sample_name, featureX_name)
    Y_dims = (sample_name, featureY_name)
    # Optional trial (both modalities) and lag (only X) dimensions
    if trial_name is not None:
        X_dims = (trial_name,) + X_dims
        Y_dims = (trial_name,) + Y_dims
    if lag_name is not None:
        X_dims = (lag_name,) + X_dims
    if set(X.dims) != set(X_dims):
        raise ValueError(f"X dimensions should be {X_dims} but got {X.dims}")
    if set(Y.dims) != set(Y_dims):
        raise ValueError(f"Y dimensions should be {Y_dims} but got {Y.dims}")

def validate_dimension_sizes(Ntx : int,
                             Nty : int,
                             N_features : int | None = None,
                             N_components: int | None = None,
                             Nkx : int | None = None,
                             Nky : int | None = None
                             ) -> tuple[int, int]:
        """Validates dimension sizes of multimodal data.

        Takes the sample dimension sizes corresponding to X and Y datasets, Ntx and Nty,
        the number of features and the number of components and corroborate the they are
        consistent among themselves. If X and Y come with trials, Ntx and Nty are the
        samples per trial, and the number of trials Nkx and Nky must match. The number
        of samples is not checked against the number of features, since with more
        features than samples the covariances can still be regularized (L2), and
        singular covariances are handled by the corresponding models.

        Args:
            Ntx (int): Number of samples in X (per trial, if any).
            Nty (int): Number of samples in Y (per trial, if any).
            N_features (int, optional): Number of features in X and Y.
            N_components (int, optional): Number of components to extract.
            Nkx (int, optional): Number of trials in X.
            Nky (int, optional): Number of trials in Y.

        Returns:
            tuple[int, int]: Updated number of components and total number of samples
                (pooled over trials, if any).

        Raises:
            ValueError: If X or Y do not have the expected dimensions, if the number
                of trials or samples between X and Y is inconsistent, or if the number
                of components exceeds the number of features.
        """

        # Catch inconsistent dimension size errors
        if Nkx != Nky:
            raise ValueError(f"Number of trials must be the same between X ({Nkx}) and Y ({Nky})")
        if Ntx != Nty:
            raise ValueError(f"Number of samples must be the same between X ({Ntx}) and Y ({Nty})")

        # Define total number of samples (pooled over trials if present)
        N_samples = Ntx if Nkx is None else Ntx * Nkx

        if N_components:
            if N_features and N_components > N_features:
                raise ValueError(f"Number of components {N_components} should be smaller than number of features {N_features}")
        else:
            N_components = N_features  # Catch None and 0 case

        return N_components, N_samples

def validate_sample_alignment(X : xr.DataArray,
                              Y : xr.DataArray,
                              sample_name : str
                              ) -> None:
        """Check that X and Y share the same sample coordinates, i.e. they are aligned.

        Samples of X and Y are paired by position, so their coordinates along the
        sample dimension (e.g. time points) must match. Numeric coordinates are compared
        up to a tolerance of 1e-6 times the smallest sampling step, independent of their
        magnitude (e.g. absolute time stamps), and any others (e.g. strings or
        datetimes) exactly. Without coordinates, both default to 0..N-1 and pass. Must
        be called after checking that Ntx == Nty (see validate_dimension_sizes).

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
            sample_name (str): Label of the sample dimension of X and Y.

        Raises:
            ValueError: If the sample coordinates of X and Y differ.
        """

        tx, ty = X[sample_name].values, Y[sample_name].values
        if np.issubdtype(tx.dtype, np.number) and np.issubdtype(ty.dtype, np.number):
            atol = 1e-6 * np.abs(np.diff(tx)).min() if len(tx) > 1 else 0
            aligned = np.allclose(tx, ty, rtol=0, atol=atol)
        else:
            aligned = np.array_equal(tx, ty)

        if not aligned:
            raise ValueError(
                f"X and Y must share the same {sample_name} coordinates, i.e. be "
                f"aligned along {sample_name}."
            )

def validate_time_coverage(X : xr.DataArray,
                           Y : xr.DataArray,
                           sample_name : str = "time"
                           ) -> None:
        """Check that X and Y span the same time domain, up to the sampling period of Y.

        For modalities with different sampling rates (e.g. mSPoC), sample alignment
        is replaced by a common time domain: no sample of either modality can be further
        than one sampling period of Y, dty, from its closest sample of the other
        modality. Since the time points are sorted, this only needs checking at the
        beginning and the end, where it reduces to |tx_0 - ty_0| <= dty and
        |tx_N - ty_N| <= dty. A tolerance of 1e-6 dty is used for floating point errors
        in the time coordinates.

        Args:
            X (DataArray): Input data for modality X, with the higher sampling rate.
            Y (DataArray): Input data for modality Y, with the lower sampling rate.
            sample_name (str, optional): Label of the time dimension of X and Y.

        Raises:
            ValueError: If the first or last time points of X and Y differ by more than
                one sampling period of Y.
        """

        tx, ty = X[sample_name].values, Y[sample_name].values
        dty = (ty[-1] - ty[0]) / (len(ty) - 1)

        # Distance between the first, and the last, time points of X and Y
        for edge, i in (("first", 0), ("last", -1)):
            if abs(tx[i] - ty[i]) > dty * (1 + 1e-6):
                raise ValueError(
                    f"X and Y must span the same time domain, but their {edge} time "
                    f"points ({tx[i]} and {ty[i]}) differ by more than one sampling "
                    f"period of Y ({dty})."
                )

def validate_l1_reg(l1_reg : float | list[float, float],
                    relative : bool = False
                    ) -> list[float, float]:
        """Check correct format of L1 regularization parameter.

        Args:
            l1_reg (float or list of floats): L1 regularization parameter(s) for sparsity. If a scalar,
                the same value is applied to both u and v; if a list of two values, the first is used for u
                and the second for v.
            relative (bool, optional): Whether the parameters are relative L1 thresholds,
                which must lie in [0, 1). Defaults to False (no range check).
        Returns:
            list of floats: List with L1 regularization parameters, the first is used for u
                and the second for v.
        """


        if isinstance(l1_reg, float) or isinstance(l1_reg, int):
            # Assume same parameters for both datasets
            l1_reg = [l1_reg, l1_reg]
        elif isinstance(l1_reg, list) and len(l1_reg)==2:
            l1_reg = l1_reg
        else:
            raise ValueError(f"Wrong format for l1_reg. Got {l1_reg}, but it should be a scalar, or a 2D list.")

        if relative:
            for kappa in l1_reg:
                if isinstance(kappa, bool) or not 0 <= kappa < 1:
                    raise ValueError(f"Wrong value for l1_reg. Got {kappa}, but it should be a number in [0, 1).")

        return l1_reg

def validate_l2_reg(l2_reg : float | str | list
                    ) -> list:
        """Check correct format of L2 regularization parameter.

        Args:
            l2_reg (float, str, or list): L2 regularization parameter(s), i.e. shrinkage
                intensities. Each is either a number in [0, 1], or "lw" / "auto" to estimate
                it with Ledoit-Wolf. If a single value, the same is applied to both X and Y;
                if a list of two values, the first is used for X and the second for Y.
        Returns:
            list: List with the two L2 regularization parameters (with "auto" mapped to
                "lw"), the first is used for X and the second for Y.
        """

        if not isinstance(l2_reg, list):
            # Assume same parameters for both datasets
            l2_reg = [l2_reg, l2_reg]
        elif len(l2_reg) != 2:
            raise ValueError(f"Wrong format for l2_reg. Got {l2_reg}, but it should be a single value, or a 2D list.")

        validated = []
        for gamma in l2_reg:
            if isinstance(gamma, str) and gamma.lower() in ("lw", "auto"):
                validated.append("lw")
            elif isinstance(gamma, (int, float, np.number)) and not isinstance(gamma, bool) and 0 <= gamma <= 1:
                validated.append(gamma)
            else:
                raise ValueError(f"Wrong value for l2_reg. Got {gamma}, but it should be a number in [0, 1], 'lw', or 'auto'.")

        return validated

def validate_solver(solver : str
                    ) -> str:
        """Check that the solver is either "alternating" or "whitening".

        Args:
            solver (str): Space where the one-filter sub-problems are solved.
        Returns:
            str: The validated solver.
        """

        if solver not in ("alternating", "whitening"):
            raise ValueError(f"Wrong solver. Got {solver}, but it should be 'alternating' or 'whitening'.")

        return solver

def validate_time_shifts(T : float,
                         time_shifts : np.ndarray
                         ) -> np.ndarray:
    """Corroborate that time shifts have the right format and are within the data domain.

    This method checks that the time shifts are positive and shorter than the duration
    of the data, and orders them in ascending order.

    Args:
        T (float): Duration of the data, i.e. last minus first time point.
        time_shifts (np.ndarray): Array of time shifts to consider.
    Returns:
        np.ndarray: Array of ordered, positive time shifts.
    """

    # Catch negative shifts
    if np.any(time_shifts < 0):
        raise ValueError('Time shifts must be positive.')
    
    # Sort time shifts in increasing order
    time_shifts = np.sort(time_shifts)
    tmax = time_shifts[-1]
    # Catch out of domain shift
    if tmax >= T:
        raise ValueError(f"Maximum time shift {tmax} must be smaller than the data duration {T}")

    return time_shifts

def standardize(x : xr.DataArray, 
                dim : str = 'time', 
                scale : bool = True
                ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:

    """Standardize x along dimension dim.

    It standardizes the input data x along the specified dimension dim by removing the mean value
    and scaling to unit variance (if scale=True).

    Args:
        x (DataArray): Input data to standardize.
        dim (str): Dimension to standardize along.
        scale (bool): Whether to scale the data to unit variance.

    Returns:
        tuple: A tuple (x_standard, mean, std) where:
            x_standard (DataArray): Standardized version of x.
            mean (DataArray): Mean value of x along dimension dim.
            std (DataArray): Standard deviation of x along dimension dim.
    """

    mean = x.mean(dim=dim)

    if scale:
        std = x.std(dim=dim)
        std = std.where(std != 0, 1)  # Avoid division by 0
    else:
        std = xr.ones_like(mean)
    
    x_standard = (x - mean)/std
        
    return x_standard, mean, std

def calculate_spatial_pattern(W : np.ndarray,
                              X : np.ndarray
                              ) -> np.ndarray:
    """Calculate the spatial patterns associated to a set of linear filters.

    Following :cite:t:`haufe_interpretation_2014`, the patterns A of filters W are
    A = Cov_x W Cov_s^{-1}, where Cov_x is the covariance of the data X and
    Cov_s = W^T Cov_x W is the covariance of the sources s = X W.

    Args:
        W (np.ndarray): Filters with shape (N_features, N_components).
        X (np.ndarray): Data with shape (N_samples, N_features), in the same space
            (e.g. normalization) in which the filters W were learned.

    Returns:
        np.ndarray: Spatial patterns with shape (N_features, N_components).
    """

    cov_x = np.cov(X, rowvar=False)
    cov_s = W.T @ cov_x @ W
    A = cov_x @ W @ np.linalg.inv(cov_s)
    
    return A

def filter_to_pattern(W : xr.DataArray,
                      X : xr.DataArray,
                      latent_name : str,
                      sample_name : str
                      ) -> xr.DataArray:
    """Compute the spatial patterns of filters W from data X, with the dims of W.

    All dims of W but the latent one are stacked into a single feature dim (e.g.
    lags and channels for tCCA), so the pattern is computed in the same flattened
    space in which W was learned, and then unstacked back.

    Args:
        W (DataArray): Filters with dims (..., latent_name).
        X (DataArray): Data in the filter space, with dims (sample_name, ...) and
            the same feature dims as W.
        latent_name (str): Label of the latent dim of W.
        sample_name (str): Label of the sample dim of X.

    Returns:
        DataArray: Spatial patterns with the same dims and coords as W.
    """
    feature_dims = [d for d in W.dims if d != latent_name]

    # Match the feature order of W before flattening
    X = X.sel({d: W[d].values for d in feature_dims})
    W_flat = W.stack(_feature=feature_dims).transpose("_feature", latent_name)
    X_flat = X.stack(_feature=feature_dims).transpose(sample_name, "_feature")

    # Compute the spatial patterns in the flattened space
    A = calculate_spatial_pattern(W_flat.data, X_flat.data)

    # Unstack and transpose back to the original dims of W
    A = W_flat.copy(data=A).unstack("_feature").transpose(*W.dims)

    return A

def ledoit_wolf_shrinkage(X : np.ndarray,
                          L : np.ndarray | None = None,
                          n_units : int | None = None
                          ) -> float:
    """Estimate the Ledoit-Wolf shrinkage intensity of the covariance of X.

    Following :cite:t:`ledoit_well-conditioned_2004`, the intensity is
    gamma = min(b2, d2) / d2, where d2 = ||C - m L||^2 is the distance between the
    covariance C and the target m L (m = trace(C) / p), and b2 = sum_k ||C_k - C||^2 / K^2
    is the variance of C estimated from K units with covariances C_k. Units are
    contiguous, (almost) equal-size blocks of samples, e.g. trials or epochs. With one
    unit per sample (default) this is the standard Ledoit-Wolf estimator.

    With the default L (identity) and n_units (one per sample), i.e. i.i.d. samples
    and an identity target, it matches sklearn.covariance.ledoit_wolf_shrinkage with
    assume_centered=True. It is implemented here since sklearn supports neither
    units (needed for autocorrelated samples) nor non-identity targets.

    Args:
        X (np.ndarray): Centered data with shape (N_samples, N_features).
        L (np.ndarray, optional): Shrinkage target structure with shape
            (N_features, N_features). Defaults to the identity.
        n_units (int, optional): Number of units the samples are split into.
            Defaults to N_samples (one unit per sample).

    Returns:
        float: Shrinkage intensity gamma in [0, 1].
    """

    N, p = X.shape
    L = np.eye(p) if L is None else L
    n_units = N if n_units is None else n_units
    if not 2 <= n_units <= N:
        raise ValueError(f"n_units must be between 2 and the number of samples {N}, but got {n_units}.")

    C = X.T @ X / N
    m = np.trace(C) / p
    d2 = np.sum((C - m * L) ** 2)

    # sum_k ||C_k||^2, using the smallest Gram matrix of each unit (same Frobenius norm)
    sq = 0.0
    for B in np.array_split(X, n_units):
        G = B @ B.T if len(B) < p else B.T @ B
        sq += np.sum(G**2) / len(B) ** 2

    # sum_k ||C_k - C||^2 = sum_k ||C_k||^2 - K ||C||^2, as C is the mean of the C_k
    b2 = (sq - n_units * np.sum(C**2)) / n_units**2

    # Clip gamma to [0, 1]
    b2 = min(b2, d2)

    gamma = b2 / d2

    return gamma

def shrink_cov(C : np.ndarray,
               gamma : float | str,
               L : np.ndarray | None = None,
               X : np.ndarray | None = None,
               n_units : int | None = None
               ) -> tuple[np.ndarray, float]:
    """Shrink a covariance matrix towards a scaled target, (1 - gamma) C + gamma m L.

    The target is the identity (or a structure L, e.g. a Laplacian) scaled by the mean
    variance m = trace(C) / p. The shrinkage intensity gamma is either given or
    estimated with Ledoit-Wolf from the data X (see ledoit_wolf_shrinkage).

    Args:
        C (np.ndarray): Covariance matrix with shape (N_features, N_features).
        gamma (float or str): Shrinkage intensity in [0, 1], or "lw" to estimate it.
        L (np.ndarray, optional): Shrinkage target structure. Defaults to the identity.
        X (np.ndarray, optional): Centered data from which C was computed, with shape
            (N_samples, N_features). Only needed if gamma is "lw".
        n_units (int, optional): Number of units for the Ledoit-Wolf estimation. Only
            used if gamma is "lw". Defaults to one unit per sample.

    Returns:
        tuple: A tuple (C_shrunk, gamma) with the shrunk covariance and the shrinkage
            intensity used.
    """

    p = len(C)
    L = np.eye(p) if L is None else L

    if gamma == "lw":
        cite("ledoit_well-conditioned_2004")
        gamma = ledoit_wolf_shrinkage(X, L, n_units)

    m = np.trace(C) / p
    C_shrunk = (1 - gamma) * C + gamma * m * L

    return C_shrunk, gamma

def inv_sqrt_cov(C : np.ndarray, 
                 eps : float = 1e-10
                 ) -> np.ndarray:
    """Compute the inverse square root of a covariance matrix C.

    Given a (symmetric) covariance matrix C, it computes C^{-1/2} = U Lambda^{-1/2} U^T
    using eigen-decomposition and clipping the inverse diagonal entries to eps to avoid
    division by zero and instabilities.

    Args:
        C (ndarray): Convariance matrix. Expected to be square and symmetric.
        eps (float, optional): Small value to avoid division by zero during inversion.

    Returns:
        Inverse square root of C of the same size as input matrix.
    """

    # Perform eigenvalue decomposition
    eigvals, U = np.linalg.eigh(C)
    eigvals_reg = np.clip(eigvals, eps, None)  # Avoid zero values

    # Inverse square root
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals_reg))
    C_inv_sqrt = U @ D_inv_sqrt @ U.T

    return C_inv_sqrt

def lag_to_samples(time: np.ndarray, lag: float) -> int:
    """Number of samples that a time lag spans on a time axis.

    It is the index of the first sample at or after time[0] + lag, i.e. lags between
    two samples are rounded up. A tolerance of 1e-6 sampling periods absorbs floating
    point errors in the lag and in the time coordinates (e.g. 3 * 0.1 > 3 / 10), which
    would otherwise shift a lag that lies on the time grid by one sample too many.

    Args:
        time (np.ndarray): Sorted time points.
        lag (float): Non-negative time lag, in the units of time.

    Returns:
        int: Number of samples spanned by the lag, between 0 and len(time).
    """

    time = np.asarray(time)
    tol = 1e-6 * (time[1] - time[0]) if len(time) > 1 else 0.0

    return int(np.searchsorted(time, time[0] + lag - tol))

def temporal_embedding(X: xr.DataArray, time_lags: np.ndarray) -> xr.DataArray:
    """Construct a time-embedded version of a matrix X.

    If X has shape T x N, the embedding matrix X_emb has shape P x T x N, where P is the
    number of time shifts in time_lags. The embedding matrix is built by concatenating
    time-shifted versions of the original matrix using the shifts inside time_lags. Zero
    padding is used at the beginning of each time-shifted copy to preserve the original
    length of time direction.

    Args:
        X (DataArray): Input data with at least a time dimension.
        time_lags (np.ndarray): Array with time lags to be used for temp. embedding.

    Returns:
        DataArray: Time-embedded version of X with dimensions (time_shift, *X.dims).
    """

    # Build temporal embeddings one by one (shifting along time, whatever its position
    # among the dims of X) and append to list
    time = X.time.values
    X_emb_list = []
    for shift in time_lags:
        start = lag_to_samples(time, shift)
        X_emb = X.shift(time=start, fill_value=0)
        X_emb_list.append(X_emb)

    # Stack the shifted copies along a new lag dim
    # (and assure dimensions are again in the right order)
    X_emb_xr = xr.concat(X_emb_list, dim="time_shift").assign_coords(
        {"time_shift": time_lags}
    )
    X_emb_xr = X_emb_xr.transpose("time_shift", ...)

    return X_emb_xr

def embed_lags(X : xr.DataArray,
               time_lags : np.ndarray,
               lag_name : str | None = None,
               new_lag_name : str = "time_shift"
               ) -> xr.DataArray:
    """Return a time-embedded X, with its lag dim (named new_lag_name) first.

    Without lag_name, X is embedded using time_lags (see temporal_embedding). Any extra
    dim (e.g. trials) is left untouched, so shifted copies never mix samples across it.
    With lag_name, X is already embedded and is only reordered to match time_lags,
    keeping the first copy of duplicated lags.

    Args:
        X (DataArray): Input data with a time dimension, and a lag dimension lag_name
            if already embedded.
        time_lags (np.ndarray): Validated time lags (sorted, without duplicates).
        lag_name (str, optional): Label of the lag dimension of a pre-embedded X.
        new_lag_name (str, optional): Label of the lag dimension of the output.

    Returns:
        DataArray: Time-embedded X with dims (new_lag_name, ...).
    """

    if lag_name is None:
        X = temporal_embedding(X, time_lags)
        X = X.rename({"time_shift": new_lag_name})
    # If X is already embedded, reorder it to match the time lags
    else:
        X = X.drop_duplicates(lag_name).sel({lag_name: time_lags})
        X = X.rename({lag_name: new_lag_name})

    return X.transpose(new_lag_name, ...)
