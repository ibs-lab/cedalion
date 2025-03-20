import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import cedalion.dataclasses as cdc
import cedalion.datasets
from cedalion.sigproc.epochs import to_epochs
from cedalion import units


@pytest.fixture
def rec():
    rec = cedalion.datasets.get_fingertapping()
    rec.stim.cd.rename_events(
        {"1.0": "control", "2.0": "Tapping/Left", "3.0": "Tapping/Right", "15.0": "end"}
    )

    rec["od"] = cedalion.nirs.int2od(rec["amp"])

    # differential pathlenght factors
    dpf = xr.DataArray(
        [6, 6],
        dims="wavelength",
        coords={"wavelength": rec["amp"].wavelength},
    )

    rec["conc"] = cedalion.nirs.od2conc(rec["od"], rec.geo3d, dpf, spectrum="prahl")

    return rec


@pytest.mark.parametrize(
    "tsname, dim",
    [
        ("amp", "wavelength"),
        ("od", "wavelength"),
        ("conc", "chromo"),
    ],
)
def test_to_epochs_dims_and_coordinates(rec, tsname, dim):
    """Make sure that epochs maintains coordinates and units."""

    epochs = to_epochs(
        rec[tsname],
        rec.stim,
        ["Tapping/Left", "Tapping/Right"],
        before=2 * units.s,
        after=30 * units.s,
    )

    assert set(epochs.dims) == {"epoch", "channel", dim, "reltime"}
    for c in ["reltime", "trial_type"]:
        assert c in epochs.coords

    for c in ["channel", "source", "detector", dim]:
        assert c in epochs.coords
        assert (epochs.coords[c] == rec[tsname].coords[c]).all()

    assert epochs.pint.units == rec[tsname].pint.units


def test_to_epochs_missing_trial_types(rec):
    with pytest.raises(ValueError):
        to_epochs(
            rec["amp"],
            rec.stim,
            ["Tapping/Both"],
            before=2 * units.s,
            after=30 * units.s,
        )


@pytest.fixture
def timeseries():
    # construct a sawtooth waveform with three peaks (_◿_◿_◿_)
    # fs = 0.1 Hz. Ramp up over 1 second with 0.5 s pause before and after.
    # onsets at 0.5, 2.5, and 4.5 s.

    return cdc.build_timeseries(
        np.tile(np.r_[np.zeros(5), np.arange(10), np.zeros(5)], reps=3)[None, :],
        dims=("channel", "time"),
        time=np.arange(60) * 0.1,
        channel=["S1D1"],
        value_units=None,
        time_units="s",
    )


def test_to_epochs_reltime(timeseries):
    """Stable reltime for trial onsets between timestamps of the timeseries."""

    reltime = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for onset in np.arange(0.5, 0.701, 0.01):
        df_stim = pd.DataFrame(
            {"onset": [onset], "duration": [1.0], "value": [1.0], "trial_type": ["A"]}
        )

        epochs = to_epochs(
            timeseries, df_stim, ["A"], before=0.3 * units.s, after=1 * units.s
        )

    assert_allclose(epochs.reltime, reltime)


def test_to_epochs_interpolation(timeseries):
    """Test linear interpolation for onsets between timestamps"""

    df_stim = pd.DataFrame(
        {"onset": [0.5], "duration": [1.0], "value": [1.0], "trial_type": ["A"]}
    )

    epochs = to_epochs(
        timeseries, df_stim, ["A"], before=0.3 * units.s, after=1 * units.s
    )

    reltime = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    waveform = [0.0, 0.0, 0.0, 0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

    assert_allclose(epochs.reltime, reltime)
    assert_allclose(epochs[0, 0, :], waveform)

    # shift onset by half a sample -> the timeseries needs to be evaluated between
    # samples. Linear interpolation is used.

    df_stim = pd.DataFrame(
        {"onset": [0.55], "duration": [1.0], "value": [1.0], "trial_type": ["A"]}
    )

    epochs = to_epochs(
        timeseries, df_stim, ["A"], before=0.3 * units.s, after=1 * units.s
    )

    reltime = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    waveform = [0.0, 0.0, 0.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 4.5, 0]

    assert_allclose(epochs.reltime, reltime)
    assert_allclose(epochs[0, 0, :], waveform)


def test_to_epochs_multiple_epochs(timeseries):
    """Stable reltime for trial onsets between timestamps of the timeseries."""

    df_stim = pd.DataFrame(
        {
            "onset": [0.5, 2.5, 4.5],
            "duration": [1.0, 1.0, 1.0],
            "value": [1.0, 1.0, 1.0],
            "trial_type": ["A", "A", "B"],
        }
    )

    epochs = to_epochs(
        timeseries, df_stim, ["A", "B"], before=0.3 * units.s, after=1 * units.s
    )

    assert epochs.sizes["epoch"] == 3

    waveform = [0.0, 0.0, 0.0, 0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    for i in range(3):
        assert_allclose(epochs[i, 0, :], waveform)


def test_to_epochs_trial_type_selection(timeseries):
    """Stable reltime for trial onsets between timestamps of the timeseries."""

    df_stim = pd.DataFrame(
        {
            "onset": [0.5, 2.5, 4.5],
            "duration": [1.0, 1.0, 1.0],
            "value": [1.0, 1.0, 1.0],
            "trial_type": ["A", "A", "B"],
        }
    )

    epochs = to_epochs(
        timeseries, df_stim, ["A"], before=0.3 * units.s, after=1 * units.s
    )

    assert epochs.sizes["epoch"] == 2
    assert all(epochs.trial_type == ["A", "A"])

    epochs = to_epochs(
        timeseries, df_stim, ["B"], before=0.3 * units.s, after=1 * units.s
    )

    assert epochs.sizes["epoch"] == 1
    assert all(epochs.trial_type == ["B"])


def test_to_epochs_robustness_to_jitter(timeseries):
    """Stable reltime for timeseries with jitter."""

    np.random.seed(42)

    ts1 = timeseries.copy()
    ts2 = timeseries.copy()

    df_stim1 = pd.DataFrame(
        {
            "onset": [0.5, 2.5, 4.5],
            "duration": [1.0, 1.0, 1.0],
            "value": [1.0, 1.0, 1.0],
            "trial_type": ["A", "A", "B"],
        }
    )

    df_stim2 = pd.DataFrame(
        {
            "onset": [0.4, 2.2, 4.1],
            "duration": [1.0, 1.0, 1.0],
            "value": [1.0, 1.0, 1.0],
            "trial_type": ["A", "B", "A"],
        }
    )

    reltime = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for jitter in [0.005, 0.01, 0.02, 0.03]:
        # add jitter to time stamps
        time1 = timeseries.time + np.random.normal(0, jitter, len(timeseries.time))
        time2 = timeseries.time + np.random.normal(0, jitter, len(timeseries.time))
        time1.attrs["units"] = "s"
        time2.attrs["units"] = "s"
        ts1 = ts1.assign_coords({"time": time1})
        ts2 = ts2.assign_coords({"time": time2})

        epochs1 = to_epochs(
            ts1, df_stim1, ["A", "B"], before=0.3 * units.s, after=1 * units.s
        )

        epochs2 = to_epochs(
            ts2, df_stim2, ["A", "B"], before=0.3 * units.s, after=1 * units.s
        )

        epochs = xr.concat((epochs1, epochs2), dim="epoch")

        if jitter <= 0.02:
            # tolerate small jitter on the time stamps (up to 20% of 1/fs)
            assert epochs1.sizes["epoch"] == epochs2.sizes["epoch"] == 3
            assert epochs1.sizes["reltime"] == epochs2.sizes["reltime"] == 14
            assert epochs.sizes["epoch"] == 6
            assert all(epochs.trial_type == ["A", "A", "B", "A", "B", "A"])

            assert_allclose(epochs1.reltime, reltime)
            assert_allclose(epochs2.reltime, reltime)
            assert_allclose(epochs.reltime, reltime)
        else:
            # but fail for larger values
            assert epochs.sizes["epoch"] == 6
            assert all(epochs.trial_type == ["A", "A", "B", "A", "B", "A"])
            assert epochs1.sizes["epoch"] == epochs2.sizes["epoch"] == 3

            with pytest.raises(AssertionError):
                assert epochs1.sizes["reltime"] == epochs2.sizes["reltime"] == 14


def test_to_epochs_nostim(timeseries):
    """Stable reltime for trial onsets between timestamps of the timeseries."""

    df_stim = pd.DataFrame(
        {
            "onset": [0.0, 10.0],  # epochs not contained in time range
            "duration": [1.0, 1.0],
            "value": [1.0, 1.0],
            "trial_type": ["A", "B"],
        }
    )

    epochs = to_epochs(
        timeseries, df_stim, ["A", "B"], before=0.3 * units.s, after=1 * units.s
    )

    assert epochs.sizes["epoch"] == 0


def test_to_epochs_quantitycheck(timeseries):
    df_stim = pd.DataFrame(
        {"onset": [1.0], "duration": [1.0], "value": [1.0], "trial_type": ["A"]}
    )

    with pytest.raises(ValueError):
        to_epochs(timeseries, df_stim, ["A"], before=1, after=1 * units.s)

    with pytest.raises(ValueError):
        to_epochs(timeseries, df_stim, ["A"], before=1 * units.s, after=1)


def test_to_epochs_accessor(timeseries):
    df_stim = pd.DataFrame(
        {"onset": [1.0], "duration": [1.0], "value": [1.0], "trial_type": ["A"]}
    )

    epochs1 = to_epochs(
        timeseries, df_stim, ["A"], before=0.2 * units.s, after=1 * units.s
    )

    epochs2 = timeseries.cd.to_epochs(
        df_stim, ["A"], before=0.2 * units.s, after=1 * units.s
    )

    assert (epochs1 == epochs2).all()


def test_to_epochs_dimension_independence():
    """to_epochs should work for every timeseries that has a time domain."""

    ts_channel_chromo = cdc.build_timeseries(
        np.random.random((20, 100, 2)),
        dims=["channel", "time", "chromo"],
        time=np.arange(100),
        channel=[f"C{i:04d}" for i in np.arange(20)],
        value_units="uM",
        time_units="s",
    )

    df_stim = pd.DataFrame(
        {
            "onset": [10.0, 20.0, 30.0],
            "duration": [10.0, 10.0, 10.0],
            "value": [1.0, 1.0, 1.0],
            "trial_type": ["a", "a", "a"],
        }
    )

    kwargs = {
        "df_stim": df_stim,
        "trial_types": ["a"],
        "before": 5 * cedalion.units.s,
        "after": 30 * cedalion.units.s,
    }

    to_epochs(ts_channel_chromo, **kwargs)

    ts_vertex_chromo = ts_channel_chromo.rename({"channel": "vertex"})

    to_epochs(ts_vertex_chromo, **kwargs)

    ts_vertex = ts_vertex_chromo[:, :, 0]

    to_epochs(ts_vertex, **kwargs)
