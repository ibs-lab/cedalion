"""Recording class definition for timeseries data."""

from __future__ import annotations
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
    """A dictionary of timeseries objects. The keys are the names of the timeseries."""

    masks: OrderedDict[str, xr.DataArray] = field(default_factory=OrderedDict)
    """A dictionary of masks. The keys are the names of the masks."""

    geo3d: LabeledPointCloud = field(default_factory=cdc.build_labeled_points)
    """A labeled point cloud representing the 3D geometry of the recording."""

    geo2d: LabeledPointCloud = field(default_factory=cdc.build_labeled_points)
    """A labeled point cloud representing the 2D geometry of the recording."""

    stim: pd.DataFrame = field(default_factory=cdc.build_stim_dataframe)
    """A dataframe containing the stimulus information."""

    aux_ts: OrderedDict[str, NDTimeSeries] = field(default_factory=OrderedDict)
    """A dictionary of auxiliary timeseries objects."""

    aux_obj: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    """A dictionary of auxiliary objects."""

    head_model: Optional[Any] = None
    """A head model object."""

    meta_data: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    """A dictionary of meta data."""

    # these are the loaded ML from the snirf file.
    _measurement_lists: OrderedDict[str, pd.DataFrame] = field(
        default_factory=OrderedDict
    )

    def __repr__(self):
        """Return a string representation of the Recording object."""
        return (
            f"<Recording | "
            f" timeseries: {list(self.timeseries.keys())}, "
            f" masks: {list(self.masks.keys())}, "
            f" stim: {list(self.trial_types)}, "
            f" aux_ts: {list(self.aux_ts.keys())}, "
            f" aux_obj: {list(self.aux_obj.keys())}>"
        )

    def get_timeseries(self, key: Optional[str] = None) -> NDTimeSeries:
        """Get a timeseries object by key.

        Args:
            key: The key of the timeseries to retrieve. If None, the
                last timeseries is returned.

        Returns:
            NDTimeSeries: The requested timeseries object.
        """
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
        """Get a mask by key.

        Args:
            key: The key of the mask to retrieve. If None, the last
                mask is returned.

        Returns:
            xr.DataArray: The requested mask.
        """
        if not self.masks:
            raise ValueError("masks dict is empty.")

        if key:
            return self.masks[key]
        else:
            last_key = list(self.masks.keys())[-1]

            return self.masks[last_key]

    def set_mask(self, key: str, value: xr.DataArray, overwrite: bool = False):
        """Set a mask.

        Args:
            key: The key of the mask to set.
            value: The mask to set.
            overwrite: Whether to overwrite an existing mask with the same key.
                Defaults to False.
        """
        if (overwrite is False) and (key in self.masks):
            raise ValueError(f"a mask with key '{key}' already exists!")

        self.masks[key] = value

    def get_timeseries_type(self, key):
        """Get the type of a timeseries.

        Args:
            key: The key of the timeseries.

        Returns:
            str: The type of the timeseries.
        """
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
        """Get the unique source labels from the timeseries.

        Returns:
            list: A list of unique source labels.
        """
        labels = [
            ts.source.values for ts in self.timeseries.values() if "source" in ts.coords
        ]
        return list(np.unique(np.hstack(labels)))

    @property
    def detector_labels(self):
        """Get the unique detector labels from the timeseries.

        Returns:
            list: A list of unique detector labels.
        """
        labels = [
            ts.detector.values
            for ts in self.timeseries.values()
            if "detector" in ts.coords
        ]
        return list(np.unique(np.hstack(labels)))

    @property
    def wavelengths(self):
        """Get the unique wavelengths from the timeseries.

        Returns:
            list: A list of unique wavelengths.
        """
        wl = [
            ts.wavelength.values
            for ts in self.timeseries.values()
            if "wavelength" in ts.coords
        ]
        return list(np.unique(np.hstack(wl)))

    @property
    def trial_types(self):
        """Get the unique trial types from the stimulus dataframe.

        Returns:
            list: A list of unique trial types.
        """
        return list(self.stim["trial_type"].drop_duplicates())
