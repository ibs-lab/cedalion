"""Data array schemas and utilities to build labeled data arrays."""

import functools
import inspect
import typing
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import pint
import xarray as xr
from numpy.typing import ArrayLike

import cedalion.dataclasses as cdc


class ValidationError(Exception):
    pass


@dataclass(frozen=True)
class DataArraySchema:
    dims: tuple[str]
    coords: tuple[tuple[str, tuple[str]]]

    def validate(self, data_array: xr.DataArray):
        if not isinstance(data_array, xr.DataArray):
            raise ValidationError("object is not a xr.DataArray")

        for dim in self.dims:
            if dim not in data_array.dims:
                raise ValidationError(f"dimension '{dim}' not found in data array.")

        for dim, coordinate_names in self.coords:
            for name in coordinate_names:
                if name not in data_array.coords:
                    raise ValidationError(
                        f"coordinate '{name}' missing for " f"dimension '{dim}'"
                    )
                coords = data_array.coords[name]
                actual_dim = coords.dims[0]

                if not actual_dim == dim:
                    raise ValidationError(
                        f"coordinate '{name}' belongs to dimension "
                        f"'{actual_dim}' instead of '{dim}'"
                    )


# FIXME better location?
def validate_schemas(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        ba = inspect.signature(func).bind(*args, **kwargs)
        ba.apply_defaults()

        hints = typing.get_type_hints(func, include_extras=True)
        for arg_name, hint in hints.items():
            if not isinstance(hint, typing._AnnotatedAlias):
                continue

            if arg_name == "return":
                continue

            for md in hint.__metadata__:
                if isinstance(md, DataArraySchema):
                    md.validate(ba.arguments[arg_name])

        return func(*args, **kwargs)

    return wrapper


# schemas describe the minimum requirements. LabeledPointClouds have an additional
# dimension that's name denote the coordinate system and that is not enforced yet.
# FIXME support wildcards in dims?

LabeledPointCloudSchema = DataArraySchema(
    dims=("label",), coords=(("label", ("label", "type")),)
)


NDTimeSeriesSchema = DataArraySchema(
    dims=("channel", "time"),
    coords=(
        ("time", ("time", "samples")),
        ("channel", ("channel",)),
    ),
)


# FIXME better location?
def build_timeseries(
    data: ArrayLike,
    dims: List[str],
    time: ArrayLike,
    channel: List[str],
    value_units: str,
    time_units: str,
    other_coords: dict[str, ArrayLike] = {},
):
    """Build a labeled time series data array.

    Args:
        data (ArrayLike): The data values.
        dims (List[str]): The dimension names.
        time (ArrayLike): The time values.
        channel (List[str]): The channel names.
        value_units (str): The units of the data values.
        time_units (str): The units of the time values.
        other_coords (dict[str, ArrayLike]): Additional coordinates.

    Returns:
        da (xr.DataArray): The labeled time series data array.
    """
    assert len(dims) == data.ndim
    assert "time" in dims
    assert "channel" in dims
    assert data.shape[dims.index("time")] == len(time)
    assert data.shape[dims.index("channel")] == len(channel)

    samples = np.arange(len(time))

    coords = {
        "time": ("time", time),
        "samples": ("time", samples),
        "channel": ("channel", channel),
    }
    coords.update(other_coords)

    da = xr.DataArray(
        data,
        dims=dims,
        coords=coords,
    )
    da = da.pint.quantify(value_units)
    da = da.pint.quantify({"time": time_units})

    return da


def build_labeled_points(
    coordinates: ArrayLike | None = None,
    crs: str = "pos",
    units: Optional[pint.Unit | str] = "1",
    labels: Optional[list[str]] = None,
    types: Optional[list[str]] = None,
):
    """Build a labeled point cloud data array.

    Args:
        coordinates (ArrayLike, optional): The coordinates of the points. Default: None.
        crs (str, optional): The coordinate system. Defaults to "pos".
        units (Optional[pint.Unit | str], optional): The units of the coordinates.
            Defaults to "1".
        labels (Optional[list[str]], optional): The labels of the points. Defaults to
            None.
        types (Optional[list[str]], optional): The types of the points. Defaults to
            None.

    Returns:
        xr.DataArray: The labeled point cloud data array.
    """
    if coordinates is None:
        coordinates = np.zeros((0, 3), dtype=float)
    else:
        coordinates = np.asarray(coordinates)
        assert coordinates.ndim == 2
    npoints = len(coordinates)

    if labels is None:
        # generate labels "0..0" ... "0..<npoints>" with a dynamic amount of 0-padding
        template = "%0" + str(int(np.ceil(np.log10(npoints + 1)))) + "d"
        labels = [template % i for i in range(npoints)]

    if types is None:
        types = [cdc.PointType.UNKNOWN] * npoints

    labeled_points = xr.DataArray(
        coordinates,
        dims=["label", crs],
        coords={"label": ("label", labels), "type": ("label", types)},
    ).pint.quantify(units)

    return labeled_points


def validate_stim_schema(df: pd.DataFrame):
    for column_name in ["onset", "duration", "value", "trial_type"]:
        if column_name not in df:
            raise ValidationError(f"DataFrame misses required column '{column_name}'.")


def build_stim_dataframe():
    columns = ["onset", "duration", "value", "trial_type"]
    return pd.DataFrame(columns=columns)
