import xarray as xr
import cedalion.typing as cdt
from typing import List, Tuple, Union
import pandas as pd
import numpy as np
from cedalion import units

ARTIFACTS = ["spike", "bl_shift"]

def add_event_timing(
        events: List[Tuple],
        type: str,
        channels: Union[List[str], None] = None,
        timing: pd.DataFrame = pd.DataFrame(columns = ['onset', 'duration', 'trial_type', 'value', 'channel']), 
):
    """Add event data to the timing DataFrame.

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

def gen_random_events(
        time: cdt.NDTimeSeries,
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



def add_artifacts_amp(amp: cdt.NDTimeSeries, timing: pd.DataFrame, scale: float = 1):
    """Add (scaled) artifacts to amplitude data.

    Args:
        amp: Amplitude data.
        timing: DataFrame with event timing data (columns onset_time, duration, trial_type, value, channel).
        scale: Scaling factor for the artifacts.

    Returns:
        Amplitude data with added artifacts.
    """
    amp_copy = amp.copy()
    for index, row in timing.iterrows():
        onset_time = row['onset']
        duration = row['duration']
        type = row['trial_type']
        channels = row['channel']

        if type in ARTIFACTS:
            print(f"Adding {type} at {onset_time} to {channels if channels else 'all channels'}")
            if type == "spike":
                spike = xr.DataArray(np.exp(-0.5*((amp.time-onset_time)/3)**2), dims="time", coords={"time":amp.time})
                amp_copy.loc[dict(channel=channels if channels else slice(None))] += spike*scale*0.1*units.volt
            elif type == "bl_shift":
                amp_copy.loc[dict(channel=channels if channels else slice(None), time=slice(onset_time,None))] += 0.1*scale*units.volt
        else:
            print(f"Unknown artifact type {type}")
    return amp_copy
