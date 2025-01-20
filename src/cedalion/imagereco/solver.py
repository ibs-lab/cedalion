"""Solver for the image reconstruction problem."""

import numpy as np
import pint
import xarray as xr
import cedalion.xrutils as xrutils


def pseudo_inverse_stacked(Adot, alpha=0.01):
    """Pseudo-inverse of a stacked matrix.

    Args:
        Adot (xr.DataArray): Stacked matrix.
        alpha (float): Regularization parameter.

    Returns:
        xr.DataArray: Pseudo-inverse of the stacked matrix.
    """

    if "units" in Adot.attrs:
        units = pint.Unit(Adot.attrs["units"])
        inv_units = (1/units).units
    else:
        inv_units = pint.Unit("1")

    AA = Adot.values @ Adot.values.T
    highest_eigenvalue = np.linalg.eig(AA)[0][0].real

    B = Adot.values.T @ np.linalg.pinv(
        AA + alpha * highest_eigenvalue * np.eye(AA.shape[0])
    )

    coords = xrutils.coords_from_other(Adot)

    # don't copy the MultiIndexes
    for k in ["flat_channel", "flat_vertex"]:
        if k in coords:
            del coords[k]

    B = xr.DataArray(
        B,
        dims=("flat_vertex", "flat_channel"),
        coords=coords,
        attrs={"units": str(inv_units)},
    )

    return B
