import xarray as xr
import cedalion.typing as cdt
import pandas as pd
import numpy as np
from typing import Protocol
import cedalion.nirs as nirs
import cedalion.xrutils as xrutils

TIMING_COLUMNS = ["onset", "duration", "trial_type", "value", "channel"]

########################################################################################
# Add artifact functions here

# Artifact functions must take args (time, onset_time, duration) and should create an
# artifact with amplitude 1.
########################################################################################

class ArtifactFunction(Protocol):
    def __call__(
        self, time: xr.DataArray, onset_time: float, duration: float
    ) -> xr.DataArray:
        pass


def gen_spike(time: xr.DataArray, onset_time: float, duration: float) -> xr.DataArray:
    """Generate a basic spike artifact.

    Shape is a Gaussian centered at onset_time with amplitude = 1 and standard deviation
    = duration.

    Args:
        time: Time axis to which the spike will be added.
        onset_time: Center of the spike.
        duration: Standard deviation of the spike.

    Returns:
        DataFrame with event timing data (columns onset_time, duration, trial_type,
        value, channel).
    """

    duration = 5 if duration > 5 else duration
    return xr.DataArray(
        np.exp(-0.5 * ((time - onset_time) / duration) ** 2),
        dims="time",
        coords={"time": time},
    )


def gen_bl_shift(
    time: xr.DataArray, onset_time: float, duration: float = 0
) -> xr.DataArray:
    """Generate a baseline shift artifact.

    Args:
        time: Time axis to which the baseline shift will be added.
        onset_time: Onset of the baseline shift.
        duration: Duration of the baseline shift (has no effect).

    Returns:
        DataFrame with event timing data (columns onset_time, duration, trial_type,
        value, channel).
    """

    return xr.DataArray(
        np.heaviside(time - onset_time, 1), dims="time", coords={"time": time}
    )


########################################################################################


def add_event_timing(
    events: list[tuple[float, float]],
    type: str,
    channels: list[str] | None = None,
    timing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add event data to the timing DataFrame, or creates a new one if not provided.

    Args:
        events: List of tuples in format (onset, duration)
        type: Type of the event batch.
        channels: List of channels to which the event batch applies.
        timing: DataFrame of events.

    Returns:
        Updated timing DataFrame.
    """

    new_rows = pd.DataFrame(events, columns=["onset", "duration"])
    new_rows["trial_type"] = type
    new_rows["value"] = 1
    if channels:
        new_rows["channel"] = [channels] * len(new_rows)
    else:
        new_rows["channel"] = None

    if timing is None:
        timing = pd.DataFrame(columns=TIMING_COLUMNS)

    timing = pd.concat([timing, new_rows], ignore_index=True)

    return timing


def sel_chans_by_opt(optodes: list[str], ts: cdt.NDTimeSeries) -> list[str]:
    """Returns list of channels involving selected optodes."""

    sel_chan = []
    for opt in optodes:
        sel_chan.extend(ts.sel(channel=ts.source == opt).channel.values)
        sel_chan.extend(ts.sel(channel=ts.detector == opt).channel.values)
    sel_chan = np.unique(sel_chan).tolist()
    return sel_chan


def random_events_num(
    time: xr.DataArray,
    num_events: int,
    types: list[str],
    channels: list[str] | None = None,
):
    """Generates timing data for random events.

    Events are randomly selected from the types list and assigned random onset/duration
    within the time series.

    Args:
        time: Time series to which the events are added.
        num_events: Number of events to generate.
        types: List of event types.
        channels: List of channels to which the events apply.

    Returns:
        DataFrame with event timing data (columns onset_time, duration, trial_type,
        value, channel).
    """

    timing = pd.DataFrame(columns=TIMING_COLUMNS)

    for i in range(num_events):
        onset_time = np.random.uniform(time[0], time[-1])
        duration = np.random.uniform(0.1, time[-1] - onset_time)
        type = np.random.choice(types)
        timing = add_event_timing([(onset_time, duration)], type, channels, timing)

    return timing


def random_events_perc(
    time: xr.DataArray,
    perc_events: float,
    types: list[str],
    channels: list[str] | None = None,
    min_dur: float = 0.1,
    max_dur: float = 0.4,
    timing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generates timing data for random events. Mainly intended for spike artifacts.

    Events are randomly selected from the types list and assigned random onset/duration
    within the time series.

    Args:
        time: Time series to which the events are added.
        perc_events: Percentage of time series to cover with events.
        types: List of event types.
        channels: List of channels to which the events apply.
        min_dur: Minimum duration of the events.
        max_dur: Maximum duration of the events.
        timing (optional): DataFrame with existing event timing data.

    Returns:
        DataFrame with event timing data (columns onset_time, duration, trial_type,
        value, channel).
    """

    if timing is None:
        timing = pd.DataFrame(columns=TIMING_COLUMNS)

    event_time = 0
    total_time = time[-1].item() - time[0].item()

    while event_time < total_time * perc_events:
        onset_time = np.random.uniform(time[0], time[-1])
        duration = np.random.uniform(min_dur, max_dur)
        type = np.random.choice(types)
        if onset_time + duration < time[-1]:
            timing = add_event_timing([(onset_time, duration)], type, channels, timing)
            event_time += duration
    return timing

def add_artifact_direct(
    ts: cdt.NDTimeSeries, timing: tuple[float, float], artifact_func, scale: float = 1.0
):
    """Add a single artifact to a timeseries with direct scaling (mainly for testing).

    Args:
        ts: timeseries data
        timing: Tuple with onset time, duration of the artifact.
        artifact_func: Artifact function. Artifact function must take args (time).
        scale: scale for artifact.

    Returns:
        Timeseries data with added artifact.
    """

    ts_copy = ts.copy()
    unit = ts_copy.pint.units if ts_copy.pint.units else 1

    artifact = artifact_func(ts_copy.time, timing[0], timing[1])
    ts_copy += artifact * scale * unit
    return ts_copy


def add_artifacts(
    ts: cdt.NDTimeSeries,
    timing: pd.DataFrame,
    artifacts: dict[str, ArtifactFunction],
    mode: str = "auto",
    scale: float = 1.0,
    window_size: float = 120
) -> cdt.NDTimeSeries:
    """Add scaled artifacts to timeseries data.

    Supports timeseries with channel and either wavelength or chromophore dimension.
    Currently assumes that artifacts affect both wavelengths/chromophores equally.

    Args:
        ts: fNIRS timeseries data (i.e. amp or OD).
        timing: DataFrame with event timing data (columns onset_time, duration,
            trial_type, value, channel).
        artifacts: Dictionary of artifact functions. Artifact functions must take args
            (time, onset_time, duration). Keys correspond to the trial_type in the
            timing DataFrame.
        mode: 'auto' or 'manual'. If 'auto', artifacts are scaled using the alpha
            parameter (median of median of sliding windows) AND the scale parameter. If 
            'manual', artifacts are scaled only by the scale parameter.
        scale: scaling parameter for artifacts
        window_size: size of sliding window for alpha computation

    Returns:
        Amplitude data with added artifacts.
    """

    ts_copy = ts.copy()
    unit = ts_copy.pint.units if ts_copy.pint.units else 1

    time_start = ts_copy["time"][0].item()
    time_end = ts_copy["time"][-1].item()

    # Detect dimension for chromophore or wavelength
    try:
        dim_name = xrutils.other_dim(ts_copy, "time", "channel")
    except ValueError:
        raise ValueError("No wavelength or chromophore dimension found.")
    dim_values = ts_copy[dim_name].values

    # set parameters for computing alpha
    step_size = window_size // 2
    channels = ts_copy.channel.values

    # generate alpha for each channel/wavelength or channel/chromo
    if mode == "auto":
        windows = ts_copy.rolling(time=window_size, center=True).construct(
            "window", stride=step_size
        )
        amplitudes = windows.reduce(np.max, dim="window") - windows.reduce(
            np.min, dim="window"
        )
        alphas = amplitudes.median(dim="time").pint.dequantify()
    elif mode == "manual":
        alphas = {
            (channel, dim_value): 1
            for dim_value in dim_values
            for channel in channels
        }
    else:
        raise ValueError("Invalid mode. Must be 'auto' or 'manual'.")

    # make sure events are within bounds of timeseries
    valid_events = timing[
        (timing["onset"] >= time_start)
        & (timing["onset"] + timing["duration"] <= time_end)
    ]

    # add artifacts to timeseries
    for index, row in valid_events.iterrows():
        onset_time = row["onset"]
        duration = row["duration"]
        type = row["trial_type"]
        sel_channels = row["channel"] if row["channel"] else channels
        if type in artifacts.keys():
            artifact = artifacts[type](ts_copy.time, onset_time, duration)
            for channel in list(set(channels) & set(sel_channels)):
                for dim_value in dim_values:
                    alpha = alphas.sel(channel=channel, **{dim_name: dim_value}).item()
                    ts_copy.loc[dict(channel=channel, **{dim_name: dim_value})] += (
                        artifact * alpha * scale * unit
                    )
        else:
            raise ValueError(f"Unknown artifact type {type}")

    return ts_copy


def add_chromo_artifacts_2_od(
    od: cdt.NDTimeSeries,
    timing: pd.DataFrame,
    artifacts,
    geo3d,
    dpf,
    scale: float = 1.0,
    window_size: float = 120,
):
    """Scale artifacts by chromo amplitudes and add to OD data."""

    conc = nirs.od2conc(od, geo3d, dpf)
    conc = add_artifacts(conc, timing, artifacts, mode="auto", scale=scale,
                         window_size=window_size)
    return nirs.conc2od(conc, geo3d, dpf)
