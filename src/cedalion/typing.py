from typing import Annotated, TypeAlias

import xarray as xr

from cedalion.dataclasses.xrschemas import LabeledPointCloudSchema, NDTimeSeriesSchema

LabeledPointCloud = Annotated[xr.DataArray, LabeledPointCloudSchema]
NDTimeSeries = Annotated[xr.DataArray, NDTimeSeriesSchema]

AffineTransform: TypeAlias = xr.DataArray
