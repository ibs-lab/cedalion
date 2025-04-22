"""Solver for the image reconstruction problem."""

import numpy as np
import pint
import xarray as xr
from numpy.typing import ArrayLike

import cedalion.xrutils as xrutils


def pseudo_inverse_stacked(
    Adot : xr.DataArray,
    alpha: float = 0.01,
    Cmeas: ArrayLike | None = None,
    alpha_spatial: float | None = None,
):
    """Calculate the pseudo-inverse of a stacked sensitivity matrix.

    Args:
        Adot: Stacked matrix
        alpha: Thikonov regularization parameter
        Cmeas: Optional measurement regularization parameters. If specified this can
         be either a vector of size nchannel or a matrix of size nchannelxnchannel.
        alpha_spatial: Optional spatial regularization parameter. 
         Suggested default is 1e-3, or 1e-2 when spatial basis functions are used.

    Returns:
        xr.DataArray: Pseudo-inverse of the stacked matrix.
    """

    if "units" in Adot.attrs:
        units = pint.Unit(Adot.attrs["units"])
        inv_units = (1 / units).units
    elif Adot.pint.units is not None:
        inv_units = (1 / Adot.pint.units).units
        Adot = Adot.pint.dequantify()
    else:
        inv_units = pint.Unit("1")

    # do spatial regularization
    if alpha_spatial is not None:
        AAtdiag = np.sum((Adot.values**2), axis=0)

        b = AAtdiag.max()
        lambda_spatial = alpha_spatial * b

        L = np.sqrt(AAtdiag + lambda_spatial)
        Linv = 1 / L
        A_hat = Adot.values * Linv[np.newaxis, :]
        AAt = A_hat @ A_hat.T
        At = (Linv[:, np.newaxis]**2) * A_hat.T
    else:  # no spatial regularization
        AAt = Adot.values @ Adot.values.T
        AAt = Adot.values @ Adot.values.T
        At = Adot.values.T

    highest_eigenvalue = np.linalg.eig(AAt)[0][0].real
    lambda_meas = alpha * highest_eigenvalue
    if Cmeas is None:
        B = At @ np.linalg.pinv(AAt + lambda_meas * np.eye(AAt.shape[0]))
    elif len(Cmeas.shape) == 2:
        B = At @ np.linalg.inv(AAt + lambda_meas * Cmeas)
    else:
        B = At @ np.linalg.inv(AAt + lambda_meas * np.diag(Cmeas))

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
