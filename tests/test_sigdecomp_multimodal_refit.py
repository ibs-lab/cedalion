"""Refitting a model must resolve its defaults again, from the new data."""

import numpy as np
import pytest
import xarray as xr

from cedalion.sigdecomp.multimodal.mspoc import mSPoC

FS_X = 40.0
DURATION = 60.0


def make_ts(fs: float, n_features: int, seed: int) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    n = int(DURATION * fs)
    return xr.DataArray(
        rng.standard_normal((n, n_features)),
        dims=("time", "channel"),
        coords={
            "time": np.arange(n) / fs,
            "channel": [f"c{seed}{i}" for i in range(n_features)],
        },
    )


@pytest.fixture
def X():
    return make_ts(FS_X, 3, seed=0)


def test_default_power_window_follows_y_on_refit(X):
    """The default window is one sampling period of the Y of the current fit."""
    mspoc = mSPoC(N_components=1, time_lags=[0.0, 1.0], random_state=0, max_iter=10)

    mspoc.fit(X, make_ts(5.0, 2, seed=1))
    assert mspoc.power_window == pytest.approx(0.2)

    # Slower Y: the window of the first fit (0.2 s) is shorter than its period
    mspoc.fit(X, make_ts(2.5, 2, seed=1))
    assert mspoc.power_window == pytest.approx(0.4)

    # Faster Y: the window of an earlier fit must not be silently reused
    mspoc.fit(X, make_ts(10.0, 2, seed=1))
    assert mspoc.power_window == pytest.approx(0.1)


def test_refit_equals_fresh_fit(X):
    """A refitted model gives the same filters as a new one."""
    Y_first, Y_second = make_ts(10.0, 2, seed=1), make_ts(5.0, 2, seed=2)
    kwargs = dict(N_components=1, time_lags=[0.0, 1.0], random_state=0, max_iter=10)

    refitted = mSPoC(**kwargs)
    refitted.fit(X, Y_first)
    refitted.fit(X, Y_second)
    fresh = mSPoC(**kwargs)
    fresh.fit(X, Y_second)

    assert refitted.power_window == pytest.approx(fresh.power_window)
    np.testing.assert_allclose(refitted.Wx.values, fresh.Wx.values)
    np.testing.assert_allclose(refitted.Wt.values, fresh.Wt.values)


def test_given_power_window_is_kept(X):
    mspoc = mSPoC(
        N_components=1,
        time_lags=[0.0, 1.0],
        power_window=1.0,
        random_state=0,
        max_iter=10,
    )

    mspoc.fit(X, make_ts(5.0, 2, seed=1))
    mspoc.fit(X, make_ts(2.5, 2, seed=1))

    assert mspoc.power_window == 1.0
    assert mspoc.init_power_window == 1.0
