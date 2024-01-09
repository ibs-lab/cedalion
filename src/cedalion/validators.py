from typing import List, Optional

import pint
import xarray as xr


def _assert_dims_and_coords(
    array: xr.DataArray, dimensions: List[str], coordinates: List[str]
):
    for dim in dimensions:
        if dim not in array.dims:
            raise AttributeError("Missing dimension '{dim}'")

    for coord in coordinates:
        if coord not in array.coords:
            raise AttributeError("Missing coorindate '{coord}'")


def has_time(array: xr.DataArray):
    _assert_dims_and_coords(array, ["time"], ["time"])


def has_wavelengths(array: xr.DataArray):
    _assert_dims_and_coords(array, ["wavelength"], ["wavelength"])


def has_channel(array: xr.DataArray):
    _assert_dims_and_coords(array, ["channel"], ["channel", "source", "detector"])


def has_positions(array: xr.DataArray, npos: Optional[int] = None):
    _assert_dims_and_coords(array, ["pos"], [])

    if npos is not None:
        axis = array.get_axis_num("pos")
        npos_found = array.shape[axis]
        if npos_found != npos:
            raise AttributeError(
                f"Expected geometry with {npos} dimensions but found {npos_found}"
            )


def is_quantified(array: xr.DataArray):
    return isinstance(array.variable.data, pint.Quantity)


def check_dimensionality(name: str, q: pint.Quantity, dim: str):
    if not q.check(dim):
        raise ValueError(f"quantity '{name}' does not have dimensionality '{dim}'")
