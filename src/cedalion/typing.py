"""Type aliases for Cedalion dataclasses.

Cedalion relies as much as possible on generic data types (like xarray DataArrays).
We then use type aliases and annotations to augment these data types with additional
information about the data they carry. For DataArrays there is a basic mechanism
to specify and validate data schemas that specify dimension and coordinate names.
This way we can distinguish between time series DataArrays (NDTimeSeries) and DataArrays
representing points in space (LabeledPointCloud). By using these aliases in type hints
we indicate to user which kind of DataArray is expected.

Parameters with physical units are represented by cedalion.Quantity. Aliases are defined
to indicate the dimensionality of quantities.
"""

from __future__ import annotations
from typing import Annotated, TypeAlias

import xarray as xr

from cedalion.dataclasses.schemas import LabeledPointCloudSchema, NDTimeSeriesSchema
from cedalion import Quantity

#: DataArrays representing labeled points in space.
LabeledPointCloud: TypeAlias = Annotated[xr.DataArray, LabeledPointCloudSchema]

#: DataArrays representing time series.
NDTimeSeries: TypeAlias = Annotated[xr.DataArray, NDTimeSeriesSchema]

#: 4x4 DataArrays representing affine transformations.
AffineTransform: TypeAlias = xr.DataArray

#: Quantities with units of time
QTime : TypeAlias = Annotated[Quantity, "[time]"]

#: Quantities with units of length
QLength : TypeAlias = Annotated[Quantity, "[length]"]

#: Quantities with units of frequency
QFrequency : TypeAlias = Annotated[Quantity, "[frequency]"]
