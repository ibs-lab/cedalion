"""Utility functions for multimodal models."""

import numpy as np
import xarray as xr


def validate_dimension_labels(X : xr.DataArray, 
                              Y : xr.DataArray,
                              sample_name : str,
                              featureX_name : str,
                              featureY_name : str):
    """Validates that X and Y datasets contain the expected dimension labels.

    This method checks that the data arrays X and Y contain the same 
    sample label and corresponding X and Y feature labels.

    Args:
        X (DataArray): Input data for modality X. Expected to have dimensions
            (sample_name, featureX_name).
        Y (DataArray): Input data for modality Y. Expected to have dimensions
            (sample_name, featureY_name).
        sample_name (str): Name of the sample dimension.
        featureX_name (str): Name of the feature dimension for X.
        featureY_name (str): Name of the feature dimension for Y.

    Raises:
        ValueError: If X or Y are missing the expected feature dimension labels.
    """

    # Catch missing feature dimensions
    X_dims = (sample_name, featureX_name)
    Y_dims = (sample_name, featureY_name)
    if set(X.dims) != set(X_dims):
        raise ValueError(f"X dimensions should be {X_dims} but got {X.dims}")
    if set(Y.dims) != set(Y_dims):
        raise ValueError(f"Y dimensions should be {Y_dims} but got {Y.dims}")

def validate_dimension_sizes(Ntx : int,
                             Nty : int,
                             N_features : int,
                             N_components: int
                             ) -> int:
        """Validates dimension sizes of multimodal data.

        Takes the sample dimension sizes corresponding to X and Y datasets, Ntx and Nty, 
        the number of features and the number of components and corroborate the they are
        consistent among themselves.

        Args:
            Ntx (int): Number of samples in X.
            Nty (int): Number of samples in Y.
            N_features (int): Number of features in X and Y.
            N_components (int): Number of components to extract.

        Returns:
            N (int): updated number of components.

        Raises:
            ValueError: If X or Y do not have the expected dimensions, if the number
                of samples between X and Y is inconsistent, or if the number of components
                exceeds the number of features.
        """
        
        # Catch inconsistent dimension size errors
        if Ntx != Nty:
            raise ValueError(f"Number of samples must be the same between X ({Ntx}) and Y ({Nty})")
        if Ntx < N_features:
            raise ValueError(f"Number of samples {Ntx} should be bigger than number of features {N_features}")
        if N_components:
            if N_components > N_features:
                raise ValueError(f"Number of components {N_components} should be smaller than number of features {N_features}")
        else:
            N_components = N_features  # Catch None and 0 case

        return N_components

def validate_l1_reg(l1_reg : float | list[float, float]
                    ) -> list[float, float]:
        """Check correct format of L1 regularization parameter.

        Args:
            l1_reg (float or list of floats): L1 regularization parameter(s) for sparsity. If a scalar, 
                the same value is applied to both u and v; if a list of two values, the first is used for u 
                and the second for v.
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

        return l1_reg

def validate_l2_reg(l2_reg : float | list[float, float]
                    ) -> list[float, float]:
        """Check correct format of L2 regularization parameter.

        Args:
            l2_reg (float or list of floats): L2 regularization parameter(s) for normalization. 
                If a scalar, the same value is applied to both u and v; if a list of two values, 
                the first is used for u and the second for v.
        Returns:
            list of floats: List with L2 regularization parameters, the first is used for u 
                and the second for v.
        """
        

        if isinstance(l2_reg, float) or isinstance(l2_reg, int):
            # Assume same parameters for both datasets
            l2_reg = [l2_reg, l2_reg]
        elif isinstance(l2_reg, list) and len(l2_reg)==2:
            l2_reg = l2_reg
        else:
            raise ValueError(f"Wrong format for l2_reg. Got {l2_reg}, but it should be a scalar, or a 2D list.")

        return l2_reg

def validate_time_shifts(T : float,
                         time_shifts : np.ndarray
                         ) -> np.ndarray:
    """Corroborate that time shifts have the right format and are within the data domain.

    This method checks that the time shifts are positive and within the data domain. It also
    order the shifts in ascending order and adds zero lag at the beginning of the series if not present.

    Args:
        T (float): Maximum time of the data.
        time_shifts (np.ndarray): Array of time shifts to consider.
    Returns:
        tuple: A tuple (time_shifts, N_shifts) where:
            time_shifts (np.ndarray): Array of ordered, positive time shifts.
    """

    # Catch negative shifts
    if np.any(time_shifts < 0):
        raise ValueError('Time shifts must be positive.')
    
    # Sort time shifts in increasing order
    time_shifts = np.sort(time_shifts)
    tmax = time_shifts[-1]
    tmin = time_shifts[0]
    # Catch out of domain shift
    if tmax >= T:
        raise ValueError(f"Maximum time shift {tmax} cannot be bigger than largest time in data {T}")
    # If not present, add tau=0 (no-shift) as first shift
    if tmin:
        time_shifts = np.insert(time_shifts, 0, 0.0)

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
        std[np.where(std==0)] = 1  # Avoid division by 0
    else:
        std = xr.ones_like(mean)
    
    x_standard = (x - mean)/std
        
    return x_standard, mean, std
