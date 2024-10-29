import xarray as xr
import cedalion.typing as cdt
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from cedalion import units

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
    windows = [ts.sel(time=slice(i*step_size, i*step_size + window_size)) for i in range(0, num_windows)]
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

def add_artifacts(ts: cdt.NDTimeSeries, timing: pd.DataFrame, artifacts, scale=False):
    """Add scaled artifacts to timeseries data.

    Artifacts are scaled for each channel and wavelength based on the parameter 
    generated by the function compute_alpha(...)

    Args:
        ts: fNIRS timeseries data (i.e. amp or OD).
        timing: DataFrame with event timing data (columns onset_time, duration, trial_type, value, channel).
        artifacts (Dict[str, function]): Dictionary of artifact functions. Artifact
            functions must take args (time, onset_time, duration). Keys correspond to
            the trial_type in the timing DataFrame.
        scale: scale for artifacts, or omit for automatic scaling

    Returns:
        Amplitude data with added artifacts.
    """
    ts_copy = ts.copy()
    unit = ts_copy.pint.units if ts_copy.pint.units else 1

    time_start = ts_copy["time"][0].item()
    time_end = ts_copy["time"][-1].item()

    # set parameters for compute_alpha
    window_size = (time_end - time_start) // 100
    step_size = window_size // 2
    channels = ts_copy.channel.values
    wavelengths = ts_copy.wavelength.values

    # generate alpha for each channel/wavelength
    if not scale:
        alphas = {
            (channel, wavelength): compute_alpha(ts_copy.loc[dict(channel=channel, wavelength=wavelength)], window_size, step_size) 
            for wavelength in wavelengths 
            for channel in channels
        }
    else:
        alphas = {
            (channel, wavelength): scale 
            for wavelength in wavelengths 
            for channel in channels
        }

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
                for wavelength in wavelengths:
                    scale = alphas[(channel, wavelength)]
                    # print(f"Adding {type} at {onset_time} for {duration} to {channel} {wavelength} with scale {scale}")
                    ts_copy.loc[dict(channel=channel, wavelength=wavelength)] += artifact*scale*unit

        else:
            print(f"Unknown artifact type {type}")
    return ts_copy
