import functools
import inspect
import typing
from dataclasses import dataclass
from typing import List
import numpy as np
from numpy.typing import ArrayLike
import xarray as xr
import pint
from typing import Optional
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
    data: np.ndarray,
    dims: List[str],
    time: np.ndarray,
    channel: List[str],
    value_units: str,
    time_units: str,
):
    assert len(dims) == data.ndim
    assert "time" in dims
    assert "channel" in dims
    assert data.shape[dims.index("time")] == len(time)
    assert data.shape[dims.index("channel")] == len(channel)

    samples = np.arange(len(time))

    da = xr.DataArray(
        data,
        dims=dims,
        coords={
            "time": ("time", time),
            "samples": ("time", samples),
            "channel": ("channel", channel),
        },
    )
    da = da.pint.quantify(value_units)
    da = da.pint.quantify({"time": time_units})

    return da


def build_labeled_points(
    coordinates: ArrayLike,
    crs: str,
    units: Optional[pint.Unit | str] = "1",
    labels: Optional[list[str]] = None,
    types: Optional[list[str]] = None,
):
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
