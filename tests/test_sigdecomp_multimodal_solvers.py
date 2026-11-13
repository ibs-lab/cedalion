"""Tests for the single-filter solvers of the multimodal decomposition models."""

import numpy as np
import pytest
import xarray as xr
from scipy.linalg import eigh
from scipy.optimize import minimize

from cedalion.sigdecomp.multimodal import solvers
from cedalion.sigdecomp.multimodal.mspoc import mSPoC


def random_metric(rng: np.random.Generator, p: int) -> np.ndarray:
    """Return a random positive definite (p, p) matrix."""
    Q = rng.standard_normal((3 * p, p))
    return Q.T @ Q / (3 * p)


def l1_objective(w: np.ndarray, c: np.ndarray, B: np.ndarray, lam: float) -> float:
    """Evaluate c'w - (lam/2) ||w||_1 for w rescaled onto the ellipsoid w'Bw = 1."""
    w = w / np.sqrt(w @ B @ w)
    return c @ w - 0.5 * lam * np.abs(w).sum()


def test_linear_closed_form():
    rng = np.random.default_rng(0)
    B, c = random_metric(rng, 6), rng.standard_normal(6)

    w = solvers.solve_filter(c=c, B=B)

    expected = np.linalg.solve(B, c)
    expected /= np.sqrt(expected @ B @ expected)
    np.testing.assert_allclose(w, expected)


def test_quadratic_is_leading_generalized_eigenvector():
    rng = np.random.default_rng(1)
    B = random_metric(rng, 6)
    A = rng.standard_normal((6, 6))
    A = A + A.T

    w = solvers.solve_filter(A=A, B=B)

    assert w @ B @ w == pytest.approx(1.0)
    assert w @ A @ w == pytest.approx(eigh(A, B, eigvals_only=True)[-1])


@pytest.mark.parametrize("kappa", [0.1, 0.5, 0.9])
def test_linear_l1_is_optimal_and_sparse(kappa):
    """FISTA reaches the optimum of the constrained problem (vs. a generic solver)."""
    rng = np.random.default_rng(2)
    p = 5
    B, c = random_metric(rng, p), rng.standard_normal(p)
    lam = kappa * 2 * np.abs(c).max()

    w = solvers.solve_filter(c=c, B=B, kappa=kappa)

    assert w @ B @ w == pytest.approx(1.0)
    best = max(
        -minimize(
            lambda v: -l1_objective(v, c, B, lam),
            rng.standard_normal(p),
            method="Powell",
            options=dict(xtol=1e-10, ftol=1e-12),
        ).fun
        for _ in range(20)
    )
    assert l1_objective(w, c, B, lam) >= best - 1e-6


@pytest.mark.parametrize("factor", [1.0, 1.3, 3.0])
def test_linear_l1_threshold_above_linear_term_gives_one_sparse_optimum(factor):
    """With lam >= 2 ||c||_inf the filter is the best 1-sparse one, not zero."""
    rng = np.random.default_rng(3)
    p = 5
    B, c = random_metric(rng, p), rng.standard_normal(p)
    lam = factor * 2 * np.abs(c).max()

    w = solvers.solve_linear_l1(c, B, kappa=0.0, lam=lam)

    assert np.count_nonzero(w) == 1
    assert w @ B @ w == pytest.approx(1.0)
    best = max(
        -minimize(
            lambda v: -l1_objective(v, c, B, lam),
            rng.standard_normal(p),
            method="Powell",
            options=dict(xtol=1e-10, ftol=1e-12),
        ).fun
        for _ in range(30)
    )
    assert l1_objective(w, c, B, lam) >= best - 1e-6


def test_linear_l1_is_continuous_at_the_threshold_bound():
    """Just below lam = 2 ||c||_inf FISTA gives the same 1-sparse filter."""
    rng = np.random.default_rng(4)
    B, c = random_metric(rng, 8), rng.standard_normal(8)
    bound = 2 * np.abs(c).max()

    w_below = solvers.solve_linear_l1(c, B, kappa=0.0, lam=(1 - 1e-6) * bound)
    w_at = solvers.solve_linear_l1(c, B, kappa=0.0, lam=bound)

    np.testing.assert_allclose(w_below, w_at, atol=1e-6)


@pytest.mark.parametrize("kappa", [0.3, 0.7, 0.9, 0.99])
def test_quadratic_l1_never_zeroes_the_filter(kappa):
    """Every kappa in [0, 1) must return a feasible, non-zero filter."""
    rng = np.random.default_rng(5)
    for _ in range(40):
        p = int(rng.integers(3, 20))
        B = random_metric(rng, p)
        A = rng.standard_normal((p, p))
        A = A + A.T

        w = solvers.solve_filter(A=A, B=B, kappa=kappa)

        assert np.all(np.isfinite(w))
        assert np.count_nonzero(w) >= 1
        assert w @ B @ w == pytest.approx(1.0)


def test_linear_l1_zero_linear_term_still_raises():
    with pytest.raises(AssertionError, match="linear term c is zero"):
        solvers.solve_filter(c=np.zeros(4), B=np.eye(4) * 2.0, kappa=0.2)


@pytest.mark.parametrize("l1_reg", [0.7, 0.95])
def test_mspoc_with_strong_l1(l1_reg):
    """Fit mSPoC with a large, but valid, relative L1 threshold."""
    rng = np.random.default_rng(6)
    fs_x, fs_y, duration = 50, 5, 120
    tx = np.arange(duration * fs_x) / fs_x
    ty = np.arange(duration * fs_y) / fs_y
    X = xr.DataArray(
        rng.standard_normal((len(tx), 6)) * np.array([1, 2, 3, 1, 2, 3]),
        dims=("time", "channel"),
        coords={"time": tx, "channel": [f"x{i}" for i in range(6)]},
    )
    Y = xr.DataArray(
        rng.standard_normal((len(ty), 4)),
        dims=("time", "channel"),
        coords={"time": ty, "channel": [f"y{i}" for i in range(4)]},
    )

    mspoc = mSPoC(
        N_components=2,
        time_lags=[0.0, 1.0],
        l1_reg=l1_reg,
        l2_reg=0.05,
        deflation="schur",
        random_state=0,
        max_iter=30,
    )
    mspoc.fit(X, Y)

    assert np.all(np.isfinite(mspoc.Wx)) and np.all(np.isfinite(mspoc.Wy))
    assert np.all((mspoc.Wx != 0).sum("channel") >= 1)
    assert np.all((mspoc.Wy != 0).sum("channel") >= 1)
