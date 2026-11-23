"""Tests for the Ledoit-Wolf shrinkage used by the multimodal decomposition models."""

import numpy as np
import pytest
import xarray as xr
from sklearn.covariance import ledoit_wolf_shrinkage as sklearn_lw

from cedalion.sigdecomp.multimodal.cca import ElasticNetCCA
from cedalion.sigdecomp.multimodal.mspoc import mSPoC
from cedalion.sigdecomp.multimodal.utils import ledoit_wolf_shrinkage, shrink_cov


def test_ledoit_wolf_matches_sklearn():
    """One unit per sample and an identity target is the standard estimator."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 6)) @ rng.standard_normal((6, 6))
    X = X - X.mean(axis=0)

    assert ledoit_wolf_shrinkage(X) == pytest.approx(
        sklearn_lw(X, assume_centered=True)
    )


def test_ledoit_wolf_single_feature_is_zero():
    """A 1x1 covariance equals its target, so the intensity is 0 (was NaN)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 1))

    gamma = ledoit_wolf_shrinkage(X)

    assert gamma == 0.0
    C = X.T @ X / len(X)
    C_shrunk, gamma = shrink_cov(C, "lw", X=X)
    assert gamma == 0.0
    np.testing.assert_allclose(C_shrunk, C)


def test_ledoit_wolf_covariance_equal_to_target_is_zero():
    """A covariance exactly proportional to the identity needs no shrinkage."""
    X = np.vstack([np.eye(4), -np.eye(4)])

    assert ledoit_wolf_shrinkage(X) == 0.0


def test_ledoit_wolf_unequal_units_stays_in_unit_interval():
    """Units of unequal size must not give a negative intensity."""
    # 2 units of 3 and 2 samples: without clipping, sum_k ||C_k||^2 < K ||C||^2
    X = np.array([[1.1, 0.0]] * 3 + [[1.0, 0.0]] * 2)
    X[:, 1] = [0.3, -0.2, 0.1, -0.3, 0.1]

    gamma = ledoit_wolf_shrinkage(X, n_units=2)

    assert 0.0 <= gamma <= 1.0


@pytest.fixture
def coupled_data():
    """X (time, channel) and a univariate Y (time, channel) sharing one source."""
    rng = np.random.default_rng(1)
    n = 600
    s = rng.standard_normal(n)
    time = np.arange(n) / 10.0
    X = xr.DataArray(
        np.outer(s, rng.standard_normal(4)) + 0.5 * rng.standard_normal((n, 4)),
        dims=("time", "channel"),
        coords={"time": time, "channel": [f"x{i}" for i in range(4)]},
    )
    Y = xr.DataArray(
        (s + 0.5 * rng.standard_normal(n))[:, None],
        dims=("time", "channel"),
        coords={"time": time, "channel": ["y0"]},
    )
    return X, Y, s


def test_cca_ledoit_wolf_with_univariate_y(coupled_data):
    """l2_reg="lw" must work when one modality has a single feature."""
    X, Y, s = coupled_data

    cca = ElasticNetCCA(N_components=1, l2_reg="lw", random_state=0)
    cca.fit(X, Y)
    Sx, _ = cca.transform(X, Y)

    assert cca.gamma_y == 0.0
    assert 0.0 <= cca.gamma_x <= 1.0
    assert np.all(np.isfinite(cca.Wx)) and np.all(np.isfinite(cca.Wy))
    assert abs(np.corrcoef(Sx.values[:, 0], s)[0, 1]) > 0.9


def test_mspoc_ledoit_wolf_with_univariate_y():
    """Fit mSPoC with l2_reg="lw" and a single Y feature (e.g. one fNIRS signal)."""
    rng = np.random.default_rng(2)
    fs_x, fs_y, duration = 50, 5, 60
    tx = np.arange(duration * fs_x) / fs_x
    ty = np.arange(duration * fs_y) / fs_y
    X = xr.DataArray(
        rng.standard_normal((len(tx), 3)),
        dims=("time", "channel"),
        coords={"time": tx, "channel": ["a", "b", "c"]},
    )
    Y = xr.DataArray(
        rng.standard_normal((len(ty), 1)),
        dims=("time", "channel"),
        coords={"time": ty, "channel": ["y0"]},
    )

    mspoc = mSPoC(
        N_components=1, time_lags=[0.0, 1.0], l2_reg="lw", random_state=0, max_iter=20
    )
    mspoc.fit(X, Y)

    assert mspoc.gamma_y == 0.0
    assert np.all(np.isfinite(mspoc.Wx)) and np.all(np.isfinite(mspoc.Wy))
