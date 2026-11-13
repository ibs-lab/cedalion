"""Pre-embedded trials can be shorter than the largest time lag.

Lags of a pre-embedded X are built on the continuous recording, before epoching, so
every lagged copy holds real signal history, whatever the length of the trials.
"""

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from cedalion.sigdecomp.multimodal.mspoc import mSPoC
from cedalion.sigdecomp.multimodal.tcca import tCCA
from cedalion.sigdecomp.multimodal.utils import temporal_embedding

TIME_LAGS = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
TRUE_LAG = 3.0
TRIAL_LENGTH = 2.0  # seconds, shorter than the largest lag


def cut_trials(ts: xr.DataArray, onsets: np.ndarray, fs: float) -> xr.DataArray:
    """Cut trials of TRIAL_LENGTH seconds, with a time axis relative to the onset."""
    n = int(TRIAL_LENGTH * fs)
    trials = []
    for onset in onsets:
        start = int(round(onset * fs))
        trial = ts.isel(time=slice(start, start + n))
        trials.append(trial.assign_coords(time=np.arange(n) / fs))
    return xr.concat(trials, dim="epoch")


@pytest.fixture
def lagged_data():
    """Continuous X power and Y at 4 Hz, with Y following X by TRUE_LAG seconds."""
    rng = np.random.default_rng(0)
    fs, duration = 4.0, 400.0
    n = int(duration * fs)
    time = np.arange(n) / fs

    source = gaussian_filter1d(rng.standard_normal(n), sigma=2.0)
    source /= source.std()
    delayed = np.roll(source, int(TRUE_LAG * fs))

    X = xr.DataArray(
        np.outer(source, [1.0, -0.5, 0.8]) + 0.3 * rng.standard_normal((n, 3)),
        dims=("time", "channel"),
        coords={"time": time, "channel": ["x0", "x1", "x2"]},
    )
    Y = xr.DataArray(
        np.outer(delayed, [0.7, 1.0]) + 0.3 * rng.standard_normal((n, 2)),
        dims=("time", "channel"),
        coords={"time": time, "channel": ["y0", "y1"]},
    )
    onsets = np.arange(10.0, duration - 10.0, 5.0)
    return X, Y, onsets, fs


def test_tcca_preembedded_trials_shorter_than_largest_lag(lagged_data):
    X, Y, onsets, fs = lagged_data
    X_trials = cut_trials(temporal_embedding(X, TIME_LAGS), onsets, fs)
    Y_trials = cut_trials(Y, onsets, fs)
    assert X_trials.time[-1] - X_trials.time[0] < TIME_LAGS.max()

    tcca = tCCA(N_components=1, random_state=0)
    tcca.fit(X_trials, Y_trials, trial_name="epoch", lag_name="time_shift")
    Sx, Sy = tcca.transform(
        X_trials, Y_trials, trial_name="epoch", lag_name="time_shift"
    )

    np.testing.assert_array_equal(tcca.time_lags, TIME_LAGS)
    assert tcca.optimal_shift[0] == TRUE_LAG
    corr = np.corrcoef(Sx.values.ravel(), Sy.values.ravel())[0, 1]
    assert corr > 0.8


def test_tcca_model_embedded_lags_must_still_fit_in_the_trial(lagged_data):
    """Lags embedded by the model are zero padded, so the duration check stays."""
    X, Y, onsets, fs = lagged_data
    X_trials = cut_trials(X, onsets, fs)
    Y_trials = cut_trials(Y, onsets, fs)

    tcca = tCCA(N_components=1, time_lags=TIME_LAGS, random_state=0)
    with pytest.raises(ValueError, match="must be smaller than the data duration"):
        tcca.fit(X_trials, Y_trials, trial_name="epoch")


def test_mspoc_preembedded_trials_shorter_than_largest_lag():
    rng = np.random.default_rng(1)
    fs_x, fs_y, duration = 40.0, 4.0, 120.0
    tx = np.arange(int(duration * fs_x)) / fs_x
    ty = np.arange(int(duration * fs_y)) / fs_y
    X = xr.DataArray(
        rng.standard_normal((len(tx), 3)),
        dims=("time", "channel"),
        coords={"time": tx, "channel": ["x0", "x1", "x2"]},
    )
    Y = xr.DataArray(
        rng.standard_normal((len(ty), 2)),
        dims=("time", "channel"),
        coords={"time": ty, "channel": ["y0", "y1"]},
    )
    onsets = np.arange(10.0, duration - 10.0, 5.0)
    X_trials = cut_trials(temporal_embedding(X, TIME_LAGS), onsets, fs_x)
    Y_trials = cut_trials(Y, onsets, fs_y)

    mspoc = mSPoC(N_components=1, random_state=0, max_iter=20)
    mspoc.fit(X_trials, Y_trials, trial_name="epoch", lag_name="time_shift")
    Sx, Sy = mspoc.transform(
        X_trials, Y_trials, trial_name="epoch", lag_name="time_shift"
    )

    np.testing.assert_array_equal(mspoc.time_lags, TIME_LAGS)
    assert mspoc.Wt.sizes["time_shift"] == len(TIME_LAGS)
    assert Sx.shape == Sy.shape == (len(onsets), int(TRIAL_LENGTH * fs_y), 1)
    assert np.all(np.isfinite(Sx)) and np.all(np.isfinite(Sy))
