from typing import Annotated, TypeAlias

import xarray as xr

from cedalion.dataclasses.xrschemas import LabeledPointCloudSchema, NDTimeSeriesSchema, NDDataSetSchema

LabeledPointCloud = Annotated[xr.DataArray, LabeledPointCloudSchema]
NDTimeSeries = Annotated[xr.DataArray, NDTimeSeriesSchema]
NDDataSet = Annotated[xr.Dataset, NDDataSetSchema]

AffineTransform: TypeAlias = xr.DataArray
