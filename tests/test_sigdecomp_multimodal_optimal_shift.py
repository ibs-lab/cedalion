"""Optimal time shift of tCCA models, estimated from the source correlations."""

import warnings

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import gaussian_filter1d

from cedalion.sigdecomp.multimodal.tcca import ElasticNetTCCA

FS = 4.0
TIME_LAGS = np.array([0.0, 1.0, 2.0, 3.0])
TRUE_LAG = 2.0


@pytest.fixture
def lagged_data():
    """X and Y (time, channel) sharing one source, delayed by TRUE_LAG in Y."""
    rng = np.random.default_rng(0)
    n = 2000
    time = np.arange(n) / FS
    source = gaussian_filter1d(rng.standard_normal(n), sigma=1.5)
    source /= source.std()
    delayed = np.roll(source, int(TRUE_LAG * FS))

    X = xr.DataArray(
        np.outer(source, [1.0, 0.8, -0.6]) + 0.2 * rng.standard_normal((n, 3)),
        dims=("time", "channel"),
        coords={"time": time, "channel": ["x0", "x1", "x2"]},
    )
    Y = xr.DataArray(
        np.outer(delayed, [1.0, 0.5]) + 0.2 * rng.standard_normal((n, 2)),
        dims=("time", "channel"),
        coords={"time": time, "channel": ["y0", "y1"]},
    )
    return X, Y


def correlation_shift(model: ElasticNetTCCA, X: xr.DataArray, Y: xr.DataArray):
    X_emb, Y_norm = model.to_filter_space(X, Y)
    model.estimate_optimal_shift(X_emb, Y_norm, mode="correlation")
    return model.optimal_shift


def test_correlation_mode_recovers_the_lag(lagged_data):
    X, Y = lagged_data
    tcca = ElasticNetTCCA(N_components=1, time_lags=TIME_LAGS, random_state=0)
    tcca.fit(X, Y)

    assert correlation_shift(tcca, X, Y)[0] == TRUE_LAG


def test_correlation_mode_ignores_lags_with_zero_filters(lagged_data):
    """With L1 sparsity whole lags are zeroed: they must not win as NaN."""
    X, Y = lagged_data
    tcca = ElasticNetTCCA(
        N_components=1,
        time_lags=TIME_LAGS,
        l1_reg=[0.6, 0.0],
        l2_reg=0.1,
        random_state=0,
    )
    tcca.fit(X, Y)
    zero_lags = (tcca.Wx == 0).all("channel").values[:, 0]
    assert zero_lags[0] and not zero_lags.all()  # lag 0 is sparsified away

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "invalid value encountered" either
        shift = correlation_shift(tcca, X, Y)

    assert shift[0] == TRUE_LAG
    assert not zero_lags[list(TIME_LAGS).index(shift[0])]
