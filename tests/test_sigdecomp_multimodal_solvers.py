"""Metric factorization and FISTA convergence in the shared multimodal solvers."""

import numpy as np
import pytest

from cedalion.sigdecomp.multimodal.solvers import factorize_metric, solve_filter

P = 60


def spd(rng, p=P, cond=1e3):
    """Symmetric positive definite matrix with a prescribed condition number."""
    Q, _ = np.linalg.qr(rng.standard_normal((p, p)))
    return Q @ np.diag(np.logspace(0, -np.log10(cond), p)) @ Q.T


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((P, P))
    return spd(rng), rng.standard_normal(P), (A + A.T) / 2


# ==================== FACTORIZED METRIC ====================

@pytest.mark.parametrize("kw", [{}, {"kappa": 0.3}])
def test_metric_matches_B_for_linear_problems(problem, kw):
    """Passing the factorized metric must not change the filter at all."""
    B, c, _ = problem
    w_B = solve_filter(c=c, B=B, **kw)
    w_metric = solve_filter(c=c, metric=factorize_metric(B), **kw)

    np.testing.assert_allclose(w_metric, w_B, rtol=0, atol=1e-12)


def test_metric_matches_B_for_the_quadratic_problem(problem):
    B, _, A = problem
    w_B = solve_filter(A=A, B=B, kappa=0.3)
    w_metric = solve_filter(A=A, metric=factorize_metric(B), kappa=0.3)

    np.testing.assert_allclose(w_metric, w_B, rtol=0, atol=1e-12)


def test_factorize_metric_recognizes_the_identity():
    """None and the identity both mean "no metric", so nothing is factorized."""
    for B in (None, np.eye(P)):
        metric = factorize_metric(B)
        assert metric["B_is_eye"] and metric["L_B"] is None and metric["lip"] is None


def test_factorize_metric_precomputes_for_a_real_metric():
    B = spd(np.random.default_rng(1))
    metric = factorize_metric(B)

    assert not metric["B_is_eye"]
    np.testing.assert_allclose(metric["L_B"] @ metric["L_B"].T, B, atol=1e-10)
    assert metric["lip"] == pytest.approx(2.0 * np.linalg.eigvalsh(B)[-1])


def test_B_and_metric_are_mutually_exclusive(problem):
    """Checked unconditionally: with checks off, metric would silently win over B."""
    B, c, _ = problem
    with pytest.raises(AssertionError, match="either B or metric"):
        solve_filter(c=c, B=B, metric=factorize_metric(B), checks=False)


# ==================== FISTA CONVERGENCE ====================

@pytest.mark.parametrize("cond, atol", [(1e4, 1e-5), (1e6, 1e-2)])
def test_fista_converges_within_the_default_budget(cond, atol):
    """Without the adaptive restart the momentum oscillates and max_iter decides.

    A converged solver gives the same filter however long it is allowed to run, so
    comparing the default budget with a long one detects a solver that merely ran
    out. Measured here: 3e-3 (cond 1e4) and 3e-1 (1e6) before the restart, 0 and
    2e-3 after. The looser bound at 1e6 is honest -- even with the restart the
    default budget does not fully converge there, so a metric that ill conditioned
    wants a larger max_iter or more L2 shrinkage.
    """
    rng = np.random.default_rng(2)
    B, c = spd(rng, cond=cond), rng.standard_normal(P)

    w_default = solve_filter(c=c, B=B, kappa=0.3)
    w_long = solve_filter(c=c, B=B, kappa=0.3, max_iter=20000)

    np.testing.assert_allclose(w_default, w_long, rtol=0, atol=atol)


def test_fista_reaches_the_optimum_of_the_penalized_problem():
    """At the optimum no feasible direction improves the objective."""
    rng = np.random.default_rng(3)
    B, c = spd(rng, cond=1e5), rng.standard_normal(P)

    w = solve_filter(c=c, B=B, kappa=0.3, max_iter=20000)
    lam = 0.3 * 2.0 * np.abs(c).max()

    def objective(v):
        return c @ v - lam / 2 * np.abs(v).sum()

    best = objective(w)
    for _ in range(200):
        v = w + 1e-4 * rng.standard_normal(P)
        v /= np.sqrt(v @ B @ v)                      # back onto the constraint surface
        assert objective(v) <= best + 1e-9
