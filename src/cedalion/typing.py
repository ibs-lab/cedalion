from __future__ import annotations
from typing import Annotated, TypeAlias

import xarray as xr

from cedalion.dataclasses.xrschemas import LabeledPointCloudSchema, NDTimeSeriesSchema

LabeledPointCloud: TypeAlias = Annotated[xr.DataArray, LabeledPointCloudSchema]
NDTimeSeries: TypeAlias = Annotated[xr.DataArray, NDTimeSeriesSchema]

AffineTransform: TypeAlias = xr.DataArray
