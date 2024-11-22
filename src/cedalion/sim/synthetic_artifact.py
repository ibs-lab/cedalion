import xarray as xr
import cedalion.typing as cdt
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from cedalion import units
import cedalion.nirs as nirs

####################################################################################################
# Add artifact functions here

# Artifact functions must take args (time, onset_time, duration) and should create an
# artifact with amplitude 1.
####################################################################################################

def gen_spike(time: xr.DataArray, onset_time, duration):
    """Generate a spike artifact.

    Args:
        time: Time series to which the spike is added.
        onset_time: Time of the spike.
        duration: Duration of the spike.

    Returns:
        DataFrame with event timing data (columns onset_time, duration, trial_type, value, channel).
    """
    duration = 5 if duration > 5 else duration
    return xr.DataArray(np.exp(-0.5*((time-onset_time)/duration)**2), dims="time", coords={"time":time})

def gen_bl_shift(time: xr.DataArray, onset_time, duration=0):
    """Generate a baseline shift artifact.

    Args:
        time: Time series to which the baseline shift is added.
        onset_time: Time of the baseline shift.
        duration: Duration of the baseline shift (has no effect).

    Returns:
        DataFrame with event timing data (columns onset_time, duration, trial_type, value, channel).
    """
    return xr.DataArray(np.heaviside(time-onset_time, 1), dims="time", coords={"time":time})

####################################################################################################

def add_event_timing(
        events: List[Tuple],
        type: str,
        channels: Union[List[str], None] = None,
        timing: pd.DataFrame = pd.DataFrame(columns = ['onset', 'duration', 'trial_type', 'value', 'channel']), 
):
    """Add event data to the timing DataFrame, or creates a new one if not provided.

    Args:
        events: List of tuples in format (onset, duration)
        type: Type of the event batch.
        channels: List of channels to which the event batch applies.
        timing: DataFrame of events.

    Returns:
        Updated timing DataFrame.
    """
    new_rows = pd.DataFrame(events, columns=['onset', 'duration'])
    new_rows['trial_type'] = type
    new_rows['value'] = 1
    if channels:
        new_rows['channel'] = [channels] * len(new_rows)
    else:
        new_rows['channel'] = None

    timing = pd.concat([timing, new_rows], ignore_index=True)

    return timing

def sel_chans_by_opt(optodes, ts):
    """Returns list of channels involving selected optodes."""
    sel_chan = []
    for opt in optodes:
        sel_chan.extend(ts.sel(channel = ts.source == opt).channel.values)
        sel_chan.extend(ts.sel(channel = ts.detector == opt).channel.values)
    sel_chan = np.unique(sel_chan).tolist()
    return sel_chan

def random_events_num(
        time: xr.DataArray,
        num_events: int,
        types: List[str],
        channels: Union[List[str], None] = None,
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
        DataFrame with event timing data (columns onset_time, duration, trial_type, value, channel).
    """
    timing = pd.DataFrame(columns = ['onset', 'duration', 'trial_type', 'value', 'channel'])
    for i in range(num_events):
        onset_time = np.random.uniform(time[0], time[-1])
        duration = np.random.uniform(0.1, time[-1] - onset_time)
        type = np.random.choice(types)
        timing = add_event_timing([(onset_time, duration)], type, channels, timing)

    return timing

def random_events_perc(
        time: xr.DataArray,
        perc_events: float,
        types: List[str],
        channels: Union[List[str], None] = None,
        min_dur: float = 0.1,
        max_dur: float = 0.4,
        timing: pd.DataFrame = pd.DataFrame(columns = ['onset', 'duration', 'trial_type', 'value', 'channel']),
):
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
        DataFrame with event timing data (columns onset_time, duration, trial_type, value, channel).
    """
    event_time = 0
    while event_time < time[-1] * perc_events:
        onset_time = np.random.uniform(time[0], time[-1])
        duration = np.random.uniform(min_dur, max_dur)
        type = np.random.choice(types)
        if onset_time + duration < time[-1]:
            timing = add_event_timing([(onset_time, duration)], type, channels, timing)
            event_time += duration
    return timing

def sliding_window(ts: cdt.NDTimeSeries, window_size, step_size):
    """Generates overlapping windows from the input timeseries.

    Args:
        ts: fNIRS timeseries data (i.e. amp or OD).
        window_size: Size of the window in s.
        step_size: Step size in s.

    Returns:
        List of windows.
    """
    num_windows = int((ts["time"][-1].item() - window_size) // step_size + 1)
    windows = [ts.sel(time=slice(i*step_size, i*step_size + window_size)) for i in range(num_windows)]
    return windows

def calculate_amplitudes(windows):
    """Calculates the amplitude (max-min) for each window."""
    amplitudes = [np.max(window.pint.dequantify()) - np.min(window.pint.dequantify()) for window in windows]
    return np.array(amplitudes)

def compute_alpha(ts: cdt.NDTimeSeries, window_size, step_size):
    """Computes the Alpha parameter using the sliding window approach.

    Args:
        ts: fNIRS timeseries data (i.e. amp or OD).
        window_size: Size of the window in s.
        step_size: Step size in s.

    Returns:
        Median amplitude of the windows.
    """
    windows = sliding_window(ts, window_size, step_size)
    amplitudes = calculate_amplitudes(windows)
    return np.median(amplitudes)

def add_artifact_direct(ts: cdt.NDTimeSeries, timing: Tuple, artifact_func, scale: float = 1.0):
    """Add a single artifact to timeseries data with direct scaling (mainly for testing).

    Args:
        ts: timeseries data
        timing: Tuple with onset time, duration of the artifact.
        artifact: Artifact function. Artifact function must take args (time).
        scale: scale for artifact.

    Returns:
        Timeseries data with added artifact.
    """
    ts_copy = ts.copy()
    unit = ts_copy.pint.units if ts_copy.pint.units else 1

    artifact = artifact_func(ts_copy.time, timing[0], timing[1])
    ts_copy += artifact*scale*unit
    return ts_copy

def add_artifacts(ts: cdt.NDTimeSeries, timing: pd.DataFrame, artifacts, mode: str = "auto", scale: float = 1.0):
    """Add scaled artifacts to timeseries data.

    Supports timeseries with channel and either wavelength or chromophore dimension.
    Currently assumes that artifacts affect both wavelengths/chromophores equally.

    Args:
        ts: fNIRS timeseries data (i.e. amp or OD).
        timing: DataFrame with event timing data (columns onset_time, duration, trial_type, value, channel).
        artifacts (Dict[str, function]): Dictionary of artifact functions. Artifact
            functions must take args (time, onset_time, duration). Keys correspond to
            the trial_type in the timing DataFrame.
        mode: 'auto' or 'manual'. If 'auto', artifacts are scaled using the alpha
            parameter (median of median of sliding windows). If 'manual', artifacts
            are scaled only by the scale parameter.
        scale: scaling parameter for artifacts

    Returns:
        Amplitude data with added artifacts.
    """
    ts_copy = ts.copy()
    unit = ts_copy.pint.units if ts_copy.pint.units else 1

    time_start = ts_copy["time"][0].item()
    time_end = ts_copy["time"][-1].item()

    # set parameters for compute_alpha
    window_size = ts_copy.time.size // 100
    step_size = window_size // 2
    channels = ts_copy.channel.values

    # Detect dimension for chromophore or wavelength
    if 'wavelength' in ts_copy.dims:
        dim_name = 'wavelength'
    elif 'chromo' in ts_copy.dims:
        dim_name = 'chromo'
    else:
        raise ValueError("No wavelength or chromophore dimension found.")
    dim_values = ts_copy[dim_name].values

    # generate alpha for each channel/wavelength or channel/chromo
    if mode == "auto":
        windows = ts_copy.rolling(time=window_size, center=True).construct("window", stride=step_size)
        amplitudes = windows.reduce(np.max, dim="window") - windows.reduce(np.min, dim="window")
        alphas = amplitudes.median(dim="time").pint.dequantify()
    elif mode == "manual":
        alphas = {
            (channel, dim_value): scale 
            for dim_value in dim_values 
            for channel in channels
        }
    else:
        raise ValueError("Invalid mode. Must be 'auto' or 'manual'.")

    # make sure events are within bounds of timeseries
    valid_events = timing[(timing['onset'] >= time_start) & (timing['onset'] + timing['duration'] <= time_end)]

    for index, row in valid_events.iterrows():
        onset_time = row['onset']
        duration = row['duration']
        type = row['trial_type']
        sel_channels = row['channel'] if row['channel'] else channels
        if type in artifacts.keys():
            artifact = artifacts[type](ts_copy.time, onset_time, duration)
            for channel in list(set(channels) & set(sel_channels)):
                for dim_value in dim_values:
                    alpha = alphas.sel(channel=channel, **{dim_name: dim_value}).item()
                    ts_copy.loc[dict(channel=channel, **{dim_name: dim_value})] += artifact * alpha * scale * unit
        else:
            print(f"Unknown artifact type {type}")
    return ts_copy

def add_chromo_artifacts_2_od(od: cdt.NDTimeSeries, timing: pd.DataFrame, artifacts, geo3d, dpf, scale: float = 1.0):
    """Scale artifacts by chromo amplitudes and add to OD data."""
    conc = nirs.od2conc(od, geo3d, dpf)
    conc = add_artifacts(conc, timing, artifacts, mode="auto", scale=scale)
    return nirs.conc2od(conc, geo3d, dpf)
