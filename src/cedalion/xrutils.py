"""Utility functions for xarray objects."""

import numpy as np
import xarray as xr


def pinv(array: xr.DataArray) -> xr.DataArray:
    """Calculate the pseudoinverse of a 2D xr.DataArray.

    FIXME: handles unitless and quantified DataArrays but not
           DataArrays with units in their attrs.
    """
    if not array.ndim == 2:
        raise ValueError("array must have only 2 dimensions")

    # /!\ need to transpose dimensions when applying np.linalg.pinv
    dims = list(array.dims)
    transposed_dims = dims[::-1]

    units = array.pint.units
    inv_units = None

    # determine inverted units and dequantify
    if units is not None:
        q = 1 / units
        inv_units = q.units
        array = array.pint.dequantify()

    # apply numpy's pinv
    array_inv = xr.apply_ufunc(
        np.linalg.pinv,
        array,
        input_core_dims=[dims],
        output_core_dims=[transposed_dims],
    )

    # quantify if necessary
    if inv_units is not None:
        array_inv = array_inv.pint.quantify(inv_units)

    return array_inv


def norm(array: xr.DataArray, dim: str) -> xr.DataArray:
    """Calculate the vector norm along a given dimension."""
    if dim not in array.dims:
        raise ValueError(f"array does not have dimension '{dim}'")

    dim_index = array.dims.index(dim)

    if (units := array.pint.units) is not None:
        array = array.pint.dequantify()

    normed = xr.apply_ufunc(
        np.linalg.norm, array, input_core_dims=[[dim]], kwargs={"axis": dim_index}
    )

    if units is not None:
        normed = normed.pint.quantify(units)

    return normed


def mask(array: xr.DataArray, initval: bool) -> xr.DataArray:
    """Create a boolean mask array with the same shape as the input array."""
    return xr.full_like(array, initval, dtype=bool)
