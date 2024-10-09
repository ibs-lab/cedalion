import numpy as np
import xarray as xr


def pseudo_inverse_stacked(Adot, alpha=0.01):
    """Pseudo-inverse of a stacked matrix.

    Args:
        Adot (xr.DataArray): Stacked matrix.
        alpha (float): Regularization parameter.

    Returns:
        xr.DataArray: Pseudo-inverse of the stacked matrix.
    """
    AA = Adot.values @ Adot.values.T
    highest_eigenvalue = np.linalg.eig(AA)[0][0].real

    B = Adot.values.T @ np.linalg.pinv(
        AA + alpha * highest_eigenvalue * np.eye(AA.shape[0])
    )
    B = xr.DataArray(B, dims=("flat_vertex", "flat_channel"))

    return B

