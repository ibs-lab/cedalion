import pandas as pd
import numpy as np
from typing import Optional, List
import xarray as xr
import random
from cedalion import units
import cedalion.typing as cdt

TIMING_COLUMNS = ["onset", "duration", "trial_type", "value", "channel"]

def add_event_timing(
    events: list[tuple[float, float]] | list[tuple[float, float, float]],
    type: str,
    channels: list[str] | None = None,
    timing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add event data to the timing DataFrame, or creates a new one if not provided.

    Args:
        events: List of tuples in format (onset, duration) or (onset, duration, value).
        type: Type of the event batch.
        channels: List of channels to which the event batch applies.
        timing: DataFrame of events.

    Returns:
        Updated timing DataFrame.
    """

    if len(events[0]) == 3:
        new_rows = pd.DataFrame(events, columns=["onset", "duration", "value"])
    elif len(events[0]) == 2:
        new_rows = pd.DataFrame(events, columns=["onset", "duration"])
        new_rows["value"] = 1
    else:
        raise ValueError("Events must be tuples of length 2 or 3.")

    new_rows["trial_type"] = type

    if channels:
        new_rows["channel"] = [channels] * len(new_rows)
    else:
        new_rows["channel"] = None

    if timing is None:
        timing = pd.DataFrame(columns=TIMING_COLUMNS)

    timing = pd.concat([timing, new_rows], ignore_index=True)

    return timing


def build_event_df(
    time_axis: xr.DataArray,
    trial_types: List[str],
    num_events: Optional[int] = None,
    perc_events: Optional[float] = None,
    min_dur: cdt.QTime = 10 * units.seconds,
    max_dur: cdt.QTime = 10 * units.seconds,
    min_interval: cdt.QTime = None,
    min_value: float = 1.0,
    max_value: float = 1.0,
    order: str = "random",
    channels: Optional[List[str]] = None,
    max_attempts: int = 10000,
) -> pd.DataFrame:
    """Build a DataFrame of events given a time axis and event generation parameters.

    This function generalizes event generation for both HRF-like and artifact-like
    scenarios.

    Args:
        time_axis: The time axis of the data.
        trial_types: List of trial types to draw from.
        num_events: Number of events to generate.
        perc_events: Percentage of total time to cover with events.
        min_dur: Minimum event duration.
        max_dur: Maximum event duration.
        min_interval: Minimum interval between events.
        min_value: Minimum event amplitude.
        max_value: Maximum event amplitude.
        order: Order of types ('alternating', 'random', or 'random balanced').
            Alternating will cycle through trial types.
            Random will randomly assign trial types.
            Random balanced will randomly assign trial types, but each type will be
            assigned the same number of times (if possible).
        channels: List of channel names to add events to.
        max_attempts: Maximum number of attempts to place events.

    Returns:
        DataFrame containing stimulus metadata. Columns are:
            - onset: Event onset time.
            - duration: Event duration.
            - value: Event amplitude.
            - trial_type: Event trial type.
            - channel: Event channel (if provided).

    """

    # Validate parameters
    if (num_events is None and perc_events is None):
        raise ValueError("At least one of num_events or perc_events must be provided.")

    if order not in ["alternating", "random", "random balanced"]:
        raise ValueError("order must be 'alternating', 'random' or 'random balanced'.")

    time_unit = time_axis.units

    # Convert all time-related quantities
    min_dur = (min_dur / time_unit).to_base_units().magnitude
    max_dur = (max_dur / time_unit).to_base_units().magnitude
    if min_interval is not None:
        min_interval = (min_interval / time_unit).to_base_units().magnitude

    allow_overlap = False
    if min_interval is None or min_interval <= 0:
        min_interval = 0.0
        allow_overlap = True

    start_time = float(time_axis[0].item())
    end_time = float(time_axis[-1].item())
    total_time = end_time - start_time

    events = pd.DataFrame(columns=TIMING_COLUMNS)
    covered_time = 0.0
    attempt_count = 0
    event_count = 0

    coverage_target = total_time * perc_events if perc_events is not None else None

    perc_condition = lambda: perc_events is not None and covered_time < coverage_target  # noqa: E731
    num_condition = lambda: num_events is not None and event_count < num_events  # noqa: E731
    loop_condition = lambda: (perc_condition() or num_condition())  # noqa: E731

    while loop_condition():

        attempt_count += 1
        if attempt_count > max_attempts:
            print(
                f"Could not place events within {max_attempts} attempts. "
                f"Try smaller/no interval or reducing event count/percentage."
            )
            break

        dur = np.random.uniform(min_dur, max_dur)
        onset = np.random.uniform(
            start_time + min_interval, end_time - dur - min_interval
        )
        val = np.random.uniform(min_value, max_value)

        if not allow_overlap and overlaps(onset, dur, min_interval, events):
            # Overlap not allowed, try again
            continue

        # Event can be placed
        events = add_event_timing([(onset, dur, val)], "", channels, events)
        event_count += 1
        covered_time += dur + min_interval

    # sort events by onset
    events = events.sort_values(by="onset")
    # add trial_types
    events["trial_type"] = pick_trial_types(events, trial_types, order)

    return events


def overlaps(onset, dur, min_interval, existing_events):
    new_start = onset
    new_end = onset + dur + min_interval
    for e in existing_events:
        es = e[0]  # onset
        ee = e[0] + e[1] + min_interval  # onset + duration + interval
        # check overlap
        if not (new_end <= es or new_start >= ee):
            return True
    return False


def pick_trial_types(
    df: pd.DataFrame, trial_types: list[str], order: str,
) -> list[str]:

    num_events = len(df)
    trial_type_column = []

    if order == "alternating":
        for index in range(num_events):
            trial_type_column.append(trial_types[index % len(trial_types)])

    elif order.startswith("random"):
        if order == "random balanced":
            num_events_per_type = num_events // len(trial_types)
            stims_left = {trial_type: num_events_per_type for trial_type in trial_types}
            while any(stims_left.values()):
                trial_type = random.choices(
                    list(trial_types),
                    weights=[stims_left[tt] for tt in trial_types],
                )[0]
                trial_type_column.append(trial_type)
                stims_left[trial_type] -= 1
        # fill with random choices
        while len(trial_type_column) < num_events:
            trial_type_column.append(random.choice(trial_types))

    return trial_type_column
