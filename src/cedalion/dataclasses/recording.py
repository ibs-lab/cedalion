from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import xarray as xr

import cedalion
import cedalion.dataclasses as cdc
from cedalion.typing import LabeledPointCloud, NDTimeSeries


@dataclass
class Recording:
    """Main container for analysis objects.

    The `Recording` class holds timeseries adjunct objects in ordered dictionaries.
    It maps to the NirsElement in the snirf format but it also holds additional
    attributes (masks, headmodel, aux_obj) for which there is no corresponding
    entity in the snirf format.
    """

    timeseries: OrderedDict[str, NDTimeSeries] = field(default_factory=OrderedDict)
    masks: OrderedDict[str, xr.DataArray] = field(default_factory=OrderedDict)
    geo3d: LabeledPointCloud = field(default_factory=cdc.build_labeled_points)
    stim: pd.DataFrame = field(default_factory=cdc.build_stim_dataframe)
    aux_ts: OrderedDict[str, NDTimeSeries] = field(default_factory=OrderedDict)
    aux_obj: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    head_model: Optional[Any] = None
    meta_data: OrderedDict[str, Any] = field(default_factory=OrderedDict)

    # these are the loaded ML from the snirf file.
    _measurement_lists: OrderedDict[str, pd.DataFrame] = field(
        default_factory=OrderedDict
    )

    def __repr__(self):
        return (
            f"<Recording | "
            f" timeseries: {list(self.timeseries.keys())}, "
            f" masks: {list(self.masks.keys())}, "
            f" stim: {list(self.trial_types)}, "
            f" aux_ts: {list(self.aux_ts.keys())}, "
            f" aux_obj: {list(self.aux_obj.keys())}>"
        )

    def get_timeseries(self, key: Optional[str] = None) -> NDTimeSeries:
        if not self.timeseries:
            raise ValueError("timeseries dict is empty.")

        if key:
            return self.timeseries[key]
        else:
            last_key = list(self.timeseries.keys())[-1]

            return self.timeseries[last_key]

    # The main objects of interest are timeseries. Make them conveniently
    # accessible. rec[key] is a shortcut for rec.timeseries[key]
    def __getitem__(self, key):
        return self.get_timeseries(key)

    def __setitem__(self, key, value):
        return self.set_timeseries(key, value, overwrite=True)

    def set_timeseries(self, key: str, value: NDTimeSeries, overwrite: bool = False):
        if (overwrite is False) and (key in self.timeseries):
            raise ValueError(f"a timeseries with key '{key}' already exists!")

        self.timeseries[key] = value

    def get_mask(self, key: Optional[str] = None) -> xr.DataArray:
        if not self.masks:
            raise ValueError("masks dict is empty.")

        if key:
            return self.masks[key]
        else:
            last_key = list(self.masks.keys())[-1]

            return self.masks[last_key]

    def set_mask(self, key: str, value: xr.DataArray, overwrite: bool = False):
        if (overwrite is False) and (key in self.masks):
            raise ValueError(f"a mask with key '{key}' already exists!")

        self.mask[key] = value

    def get_timeseries_type(self, key):
        if key not in self.timeseries:
            raise KeyError(f"unknown timeseries '{key}'")

        ts = self.timeseries[key]

        q = cedalion.Quantity(1, ts.pint.units)

        if key == "amp" or key.startswith("amp_"):
            return "amplitude"
        elif key == "od" or key.startswith("od_"):
            return "od"
        elif key == "hrf" or key.startswith("hrf_"):
            return "hrf"
        elif key == "conc" or key.startswith("conc_") or (q.check(["concentration"])):
            return "concentration"
        else:
            raise ValueError(f"could not infer data type of timeseries '{key}'")

    @property
    def source_labels(self):
        labels = [
            ts.source.values for ts in self.timeseries.values() if "source" in ts.coords
        ]
        return list(np.unique(np.hstack(labels)))

    @property
    def detector_labels(self):
        labels = [
            ts.detector.values
            for ts in self.timeseries.values()
            if "detector" in ts.coords
        ]
        return list(np.unique(np.hstack(labels)))

    @property
    def wavelengths(self):
        wl = [
            ts.wavelength.values
            for ts in self.timeseries.values()
            if "wavelength" in ts.coords
        ]
        return list(np.unique(np.hstack(wl)))

    @property
    def trial_types(self):
        return list(self.stim["trial_type"].drop_duplicates())
