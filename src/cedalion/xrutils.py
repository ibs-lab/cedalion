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


def apply_mask(data_array: xr.DataArray,
               mask: xr.DataArray, operator: str, dim_collapse: str) -> xr.DataArray:
    """Apply a boolean mask to a DataArray according to the defined "operator".

    INPUTS:
    data_array:     NDTimeSeries, input time series data xarray
    mask:           input boolean mask array with a subset of dimensions matching data_array
    operator:       operators to apply to the mask and data_array
        "nan":          inserts NaNs in the data_array where mask is False
        "drop":         drops value in the data_array where mask is False
    dim_collapse:   mask dimension to collapse to, merging boolean masks along all other
                    dimensions. can be skipped with "none".
                    Example: collapsing to "channel" dimension will drop or nan a channel
                    if it is "False" along any other dimensions

    OUTPUTS:
    masked_data_array:    input data_array with applied mask
    masked_elements:      list of elements in data_array that were masked (e.g. dropped or set to NaN)
    """
    flag_collapse = False

    # check if all dimensions in mask are dimensions of data_array
    if not all(dim in data_array.dims for dim in mask.dims):
        raise ValueError("mask dimensions must be a subset of data_array dimensions")
        # check if dim_collapse is a dimension of mask
    if dim_collapse.lower() != "none":
        if dim_collapse not in mask.dims:
            raise ValueError("dim_collapse must be a dimension of mask")
        else:
            # collapse to dimension given by "dim_collapse"
            flag_collapse = True
            dims2collapse = [dim for dim in mask.dims if dim != dim_collapse]
            mask = mask.all(dim=dims2collapse)
            print(f"mask collapsed to {dim_collapse} dimension")

    # apply the mask to data_array according to instructions from "operator" argument
    if operator.lower() == "nan":
        # inserts NaNs in the data_array where mask is False
        masked_data_array = data_array.where(mask, other=np.nan)
    elif operator.lower() == "drop":
        # drops value in the data_array where mask is False.
        # Note: values are only dropped if mask has "False" across the entire  relevant dimension
        masked_data_array = data_array.where(mask, drop=True)

    # return the masked elements if dimensions were collapsed
    if flag_collapse:
        masked_elements = mask.where(~mask, drop=True)[dim_collapse].values
    else:
        masked_elements = "N/A" #FIXME clean this up: return the masked elements as a list of indices

    return masked_data_array, masked_elements


def convolve(data_array: xr.DataArray, kernel: np.ndarray, dim: str) -> xr.DataArray:
    """Convolve a DataArray along a given dimension "dim" with a "kernel"."""

    if dim not in data_array.dims:
        raise ValueError(f"array does not have dimension '{dim}'")

    if (units := data_array.pint.units) is not None:
        data_array = data_array.pint.dequantify()

    convolved = xr.apply_ufunc(
        lambda x: np.convolve(x, kernel, mode='same'),
        data_array,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True)

    if units is not None:
        convolved = convolved.pint.quantify(units)

    return convolved
