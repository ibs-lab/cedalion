"""Time lags on the sampling grid must be embedded with the exact number of samples.

Lags built as multiples of a float step (np.arange(0, 1, 0.1)) are not exactly equal
to the time points of the data (np.arange(N) / fs): 3 * 0.1 > 3 / 10. Looking such a
lag up on the time axis must not skip to the next sample.
"""

import numpy as np
import pytest
import xarray as xr

from cedalion.sigdecomp.multimodal.mspoc import mSPoC
from cedalion.sigdecomp.multimodal.tcca import ElasticNetTCCA
from cedalion.sigdecomp.multimodal.utils import lag_to_samples, temporal_embedding

FS = 10.0
TIME_LAGS = np.arange(0, 1.0, 0.1)  # 0.30000000000000004, 0.6000000000000001, ...


def make_ts(n: int, n_features: int, seed: int, fs: float = FS) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.standard_normal((n, n_features)),
        dims=("time", "channel"),
        coords={
            "time": np.arange(n) / fs,
            "channel": [f"c{seed}{i}" for i in range(n_features)],
        },
    )


@pytest.mark.parametrize(
    "fs, step", [(10.0, 0.1), (8.0, 0.125), (100.0, 0.01), (7.8125, 0.128)]
)
def test_lag_to_samples_on_grid(fs, step):
    time = np.arange(500) / fs
    lags = np.arange(0, 12) * step

    assert [lag_to_samples(time, lag) for lag in lags] == list(range(12))


def test_lag_to_samples_with_time_offset():
    time = 1234.5 + np.arange(500) / FS

    assert [lag_to_samples(time, lag) for lag in TIME_LAGS] == list(range(10))


def test_lag_to_samples_between_grid_points_rounds_up():
    time = np.arange(100) / FS

    assert lag_to_samples(time, 0.25) == 3
    assert lag_to_samples(time, 1e6) == len(time)


def test_temporal_embedding_shifts_grid_lags_exactly():
    X = make_ts(200, 2, seed=0)

    X_emb = temporal_embedding(X, TIME_LAGS)

    for i in range(len(TIME_LAGS)):
        copy = X_emb.isel(time_shift=i).transpose("time", "channel").values
        np.testing.assert_array_equal(copy[:i], 0.0)
        np.testing.assert_array_equal(copy[i:], X.values[: len(X.time) - i])


def test_tcca_with_float_step_lags():
    """Duplicated lag copies made the covariance of the embedded X singular."""
    X = make_ts(2000, 2, seed=1)
    Y = make_ts(2000, 2, seed=2)

    tcca = ElasticNetTCCA(N_components=1, time_lags=TIME_LAGS, random_state=0)
    tcca.fit(X, Y)

    assert tcca.Wx.sizes["time_shift"] == len(TIME_LAGS)
    assert np.all(np.isfinite(tcca.Wx))


def test_mspoc_crops_exactly_the_padded_samples():
    """The first source sample follows the largest lag without an extra gap."""
    X = make_ts(1200, 2, seed=3, fs=10.0)
    Y = make_ts(600, 2, seed=4, fs=5.0)

    mspoc = mSPoC(
        N_components=1, time_lags=[0.0, 3 * 0.1], random_state=0, max_iter=10
    )
    mspoc.fit(X, Y)
    _, Sy = mspoc.transform(X, Y)

    # Epochs span +-0.1 s around each Y sample, the padding of X ends at 0.3 s
    assert Sy.time.values[0] == pytest.approx(0.4)
