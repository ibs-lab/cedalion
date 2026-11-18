"""The "whitening" solver must not depend on the scale (units) of the data."""

import numpy as np
import pytest
import xarray as xr

from cedalion.sigdecomp.multimodal.cca import CCA
from cedalion.sigdecomp.multimodal.utils import inv_sqrt_cov


def random_covariance(p: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((4 * p, p))
    return Q.T @ Q / (4 * p)


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-12, 1e-16, 1e8])
def test_inv_sqrt_cov_is_scale_equivariant(scale):
    """Rescaling C by a factor a rescales C^(-1/2) by 1 / sqrt(a)."""
    C = random_covariance(5, seed=0)

    W = inv_sqrt_cov(scale * C)

    np.testing.assert_allclose(W * np.sqrt(scale), inv_sqrt_cov(C), rtol=1e-8)
    np.testing.assert_allclose(W @ (scale * C) @ W, np.eye(5), atol=1e-8)


def test_inv_sqrt_cov_of_singular_covariance_is_finite():
    """Null directions are floored relative to the largest eigenvalue."""
    C = random_covariance(4, seed=1)
    C[:, 3] = C[:, 0]
    C[3, :] = C[0, :]  # features 0 and 3 identical -> rank 3

    for scale in (1.0, 1e-12):
        W = inv_sqrt_cov(scale * C)
        assert np.all(np.isfinite(W))
        # whitened covariance: identity on the range of C, 0 on its null space
        eigvals = np.linalg.eigvalsh(W @ (scale * C) @ W)
        np.testing.assert_allclose(eigvals, [0, 1, 1, 1], atol=1e-6)


def test_inv_sqrt_cov_of_zero_matrix_is_finite():
    assert np.all(np.isfinite(inv_sqrt_cov(np.zeros((3, 3)))))


@pytest.fixture
def coupled_data():
    """X and Y (time, channel) sharing two sources, X with heterogeneous scales."""
    rng = np.random.default_rng(1)
    n = 3000
    S = rng.standard_normal((n, 2))
    X = S @ rng.standard_normal((2, 5)) + 0.5 * rng.standard_normal((n, 5))
    Y = S @ rng.standard_normal((2, 4)) + 0.5 * rng.standard_normal((n, 4))
    X = X * np.array([1.0, 2.0, 5.0, 10.0, 20.0])
    time = np.arange(n) / 10.0
    X = xr.DataArray(
        X, dims=("time", "channel"), coords={"time": time, "channel": list("abcde")}
    )
    Y = xr.DataArray(
        Y, dims=("time", "channel"), coords={"time": time, "channel": list("fghi")}
    )
    return X, Y


def canonical_correlations(X: xr.DataArray, Y: xr.DataArray) -> np.ndarray:
    cca = CCA(N_components=2, scale=False, random_state=0)
    cca.fit(X, Y)
    Sx, Sy = cca.transform(X, Y)
    return np.array(
        [np.corrcoef(Sx.values[:, i], Sy.values[:, i])[0, 1] for i in range(2)]
    )


@pytest.mark.parametrize("unit", [1e-3, 1e-6, 1e-9])
def test_cca_without_scaling_is_invariant_to_units(coupled_data, unit):
    """E.g. concentrations in M instead of uM must give the same correlations."""
    X, Y = coupled_data

    np.testing.assert_allclose(
        canonical_correlations(X * unit, Y * unit),
        canonical_correlations(X, Y),
        rtol=1e-6,
    )
