"""Extract epochs from a time series based on stimulus events."""

from __future__ import annotations
import logging

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity

from .frequency import sampling_rate

logger = logging.getLogger("cedalion")


def to_epochs(
    ts: cdt.NDTimeSeries,
    df_stim: pd.DataFrame,
    trial_types: list[str],
    before: cdt.QTime,
    after: cdt.QTime,
):
    """Extract epochs from the time series based on stimulus events.

    Args:
        ts: the time series
        df_stim: DataFrame containing stimulus events.
        trial_types: List of trial types to include in the epochs.
        before: Time before stimulus event to include in epoch.
        after: Time after stimulus event to include in epoch.

    Returns:
        xarray.DataArray: Array containing the extracted epochs.
    """

    if not isinstance(before, Quantity):
        raise ValueError("please specify 'before' as a Quantity with time units.")
    if not isinstance(after, Quantity):
        raise ValueError("please specify 'before' as a Quantity with time units.")

    # check if user-selected trial types are available
    available_trial_types = set(df_stim.trial_type)
    for trial_type in trial_types:
        if trial_type not in available_trial_types:
            raise ValueError(f"df_stim does not contain trial_type '{trial_type}'")

    # reduce df_stim to only the selected trial types
    df_stim = df_stim[df_stim.trial_type.isin(trial_types)]

    # get time axis in seconds
    if ts.time.pint.units is not None:
        time = ts.time.to("s").pint.dequantify().values
    else:
        # assume time coords are already in seconds
        time = ts.time.values

    before = before.to("s").magnitude.item()
    after = after.to("s").magnitude.item()
    fs = sampling_rate(ts).to("Hz")

    # the time stamps of the sampled time series and the events can have different
    # precision. Be explicit about how timestamps are assigned to samples in ts.
    # For samples i-1,  i , i+1 in ts with timestamps t[i-1], t[i], t[i+1] we say
    # that sample i range from 0.5 * (t[i-1] + t[i]) till 0.5 * (t[i] + t[i+1])
    # (exclusive), i.e. the time stamp is centered in the bin.
    first_edge = time[0] - 0.5 * (time[1] - time[0])
    last_edge = time[-1] + 0.5 * (time[-1] - time[-2])
    bin_edges_between_timepoints = 0.5 * (time[:-1] + time[1:])
    sample_bin_edges = np.r_[first_edge, bin_edges_between_timepoints, last_edge]

    onset_indices = np.digitize(df_stim.onset, sample_bin_edges) - 1

    before_samples = int(np.ceil((before * fs).magnitude))
    after_samples = int(np.ceil((after * fs).magnitude))
    start_indices = np.clip(onset_indices - before_samples, -1, len(time))
    stop_indices = np.clip(onset_indices + after_samples, -1, len(time))

    # Define time axis relative to onset. This time axis depends only on before
    # after and the time series' sampling_rate. We round 1/fs to millisecond precision.
    # This way, epochs from different datasets with slightly different sampling rates
    # will have the same reltime axis and can be concatenated together.
    fs = fs.magnitude.item()
    dT = np.round(1 / fs, 3)  # millisecond precision
    reltime = np.arange(-before_samples, after_samples + 1) * dT

    units = ts.pint.units
    ts = ts.pint.dequantify()

    interpolator = interp1d(
        ts.time.values, ts.values, axis=ts.dims.index("time"), fill_value="extrapolate"
    )

    epochs = []

    # dimensions of the epoch arrays. rename time to reltime
    dims = list(ts.dims)
    dims[dims.index("time")] = "reltime"

    for onset, trial_type, start, stop in zip(
        df_stim.onset, df_stim.trial_type, start_indices, stop_indices
    ):
        if start < 0:
            # start is outside time range -> skip this event
            continue

        if stop == len(time):
            # end is outside time range -> skip this event
            continue

        coords = xrutils.coords_from_other(
            ts, dims=dims, reltime=reltime, trial_type=trial_type
        )

        # Extract this epoch from ts. The timestamps in reltime and ts.time do
        # not have to agree. Hence, we linearly interpolate the original ts to query
        # ts between time samples.
        interpolated = interpolator(onset + reltime)

        epochs.append(xr.DataArray(interpolated, dims=dims, coords=coords))

    if not epochs:
        shape = list(ts.shape)
        shape[dims.index("reltime")] = len(reltime)
        shape = tuple([0] + shape)

        return xr.DataArray(
            np.zeros(shape),
            dims=["epoch"] + dims,
            coords=xrutils.coords_from_other(ts, dims=dims, reltime=reltime),
        )

    # concatenate an create epoch dimension
    epochs = xr.concat(epochs, dim="epoch")

    # if there is only one epoch or multiple epochs with the same trial_type
    # the coord 'trial_type' remains scalar. Tranform it into an array.
    if epochs.trial_type.values.shape == tuple():
        epochs = epochs.assign_coords(
            {
                "trial_type": (
                    "epoch",
                    [epochs.trial_type.item()] * epochs.sizes["epoch"],
                )
            }
        )

    # add units back
    epochs = epochs.pint.quantify(units)

    return epochs
