import numpy as np
import pandas as pd

import cedalion.dataclasses as cdc
import cedalion.sim.synthetic_artifact as synthetic_artifact
import cedalion.sim.synthetic_utils as synthetic_utils
from cedalion import units


def _ts():
    fs = 10.0
    t = np.arange(0, 300, 1 / fs)
    return cdc.build_timeseries(
        np.ones((2, 2, len(t))),
        dims=["channel", "wavelength", "time"],
        time=t,
        channel=["S1D1", "S1D2"],
        value_units="V",
        time_units="s",
        other_coords={
            "wavelength": [760.0, 850.0],
            "source": ("channel", ["S1", "S1"]),
            "detector": ("channel", ["D1", "D2"]),
        },
    )


def test_build_event_df_min_interval():
    """Events generated with min_interval do not overlap and keep the interval.

    Regression test: the overlap check iterated over the column names of the
    DataFrame and raised a TypeError whenever min_interval was given.
    """
    ts = _ts()
    events = synthetic_utils.build_event_df(
        ts.time,
        ["a"],
        num_events=8,
        min_dur=5 * units.s,
        max_dur=5 * units.s,
        min_interval=10 * units.s,
    )

    assert len(events) == 8
    onsets = events.sort_values("onset")["onset"].values
    durations = events.sort_values("onset")["duration"].values
    gaps = onsets[1:] - (onsets[:-1] + durations[:-1])
    assert (gaps >= 10 - 1e-9).all()


def test_add_artifacts_manual_mode():
    """mode='manual' adds the artifact with unit amplitude times scale.

    Regression test: alphas was a dict and `.sel` raised an AttributeError.
    """
    ts = _ts()
    timing = pd.DataFrame(
        {
            "onset": [100.0],
            "duration": [2.0],
            "trial_type": ["spike"],
            "channel": [None],
        }
    )
    out = synthetic_artifact.add_artifacts(
        ts, timing, {"spike": synthetic_artifact.gen_spike}, mode="manual", scale=0.5
    )

    diff = (out - ts).pint.dequantify()
    assert float(diff.max()) > 0.4
    # the Gaussian spike (std = duration) is confined to the event
    assert float(diff.sel(time=slice(0, 80)).max()) < 1e-6
