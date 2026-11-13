"""Unified entry point for the single-filter sub-problems arising in CCA, tCCA,
CSP and mSPoC (and their regularized variants), when solved one filter at a time.

Every sub-problem has the form

    max_w  f(w) - lam * ||w||_1     s.t.   w' B w = 1,   B > 0

where f is EITHER linear (f = c'w) OR quadratic (f = w'A w), never both, and
B is the within-view metric (a covariance, already shrunk if L2 is wanted).
"""  # noqa: D205

import numpy as np
from scipy.linalg import cho_solve, eigh, eigvalsh

# ==== PROBLEM TAXONOMY =======================================================
# tag      objective              lam    B        solver strategy
# -------  ---------------------  -----  -------  ------------------------------
# 'A'      c'w                    = 0    any      closed form: B^-1 c, normalized
# 'B'      c'w - lam|w|_1         > 0    any      FISTA on the composite problem
# 'B_eye'  c'w - lam|w|_1         > 0    I        closed form: one soft-threshold
# 'B_one'  c'w - lam|w|_1         >= 2|c|_inf  any  closed form: best 1-sparse filter
# 'C0'     w'A w                  = 0    any      GEVD: eigh(A, B), leading vector
# 'C1'     w'A w - lam|w|_1       > 0    any      shift A by rho*B, then MM -> 'B'
#
# Only 'A', 'B', 'B_eye', and 'C0' are globally optimal; 'C1' reaches a
# stationary point only, so the caller is responsible for random restarts.

# ==== CONSTANTS ==============================================================

SYM_TOL = 1e-8      # relative tolerance when checking symmetry of A and B
EYE_TOL = 1e-10     # tolerance when deciding whether B is the identity


# ==== CONSISTENCY CHECKS =====================================================

def _check_square_symmetric(M, name, p):
    """M must be square (p,p) and symmetric. Returns the symmetrized copy."""
    assert M.shape == (p, p), f"{name} must have shape ({p},{p}), got {M.shape}"
    assert np.all(np.isfinite(M)), f"{name} contains NaN or inf"
    asym = np.abs(M - M.T).max()
    assert asym <= SYM_TOL * max(np.abs(M).max(), 1.0), f"{name} is not symmetric (max asymmetry {asym:.2e})"
    return 0.5 * (M + M.T)   # kill the residual numerical asymmetry

def _check_metric_pd(B, name):
    """B must be positive definite.

    Cholesky is the cheapest exact test, and the
    factor is reusable by the solver, so we hand it back instead of discarding it.
    """

    try:
        L = np.linalg.cholesky(B)
    except np.linalg.LinAlgError:
        raise ValueError(f"{name} is not positive definite. If it is a sample covariance "
                         f"with p > n, apply shrinkage (L2) before calling this solver.")
    return L


# ==== DISPATCHER =============================================================

def solve_filter(A=None, c=None, B=None, kappa=0.0, w_init=None, checks=True, **solver_kwargs):
    """Solve one single-filter sub-problem and return the filter w.

    Parameters
    ----------
    A : (p,p) ndarray or None
        Objective matrix for a QUADRATIC sub-problem (e.g. mSPoC's C^{s_y},
        CSP's class covariance). May be indefinite. Mutually exclusive with c.
    c : (p,) ndarray or None
        Objective vector for a LINEAR sub-problem (a cross-covariance against a
        fixed target, e.g. CCA's C_xy w_y). Mutually exclusive with A.
    B : (p,p) ndarray or None
        The constraint metric, w'Bw = 1. Must be positive definite, so pass the
        SHRUNK covariance if L2 regularization is wanted. None means identity.
    kappa : float in [0, 1)
        Scale-free sparsity knob. 0 disables L1. The absolute threshold is
        derived inside each solver as lam = kappa * 2*||c||_inf, where c is the
        linear term CURRENTLY in play -- which is why the knob is relative:
        in the quadratic case that term is rebuilt at every MM iteration.
    w_init : (p,) ndarray or None
        Warm start. Ignored by the closed-form solvers, used by FISTA and MM.
    checks : bool
        Run the consistency checks. Set False inside hot loops once the caller
        knows the inputs are well formed.
    **solver_kwargs
        Passed straight through (max_iter, tol, n_mm_steps, ...).

    Returns
    -------
    w : (p,) ndarray, normalized so that w' B w = 1.
    """

    # ---- identify the objective type ----
    assert (A is None) != (c is None), "provide exactly one of A (quadratic) or c (linear)"
    quadratic = A is not None
    p = A.shape[0] if quadratic else c.shape[0]

    # ---- validate inputs ----
    if checks:
        assert 0.0 <= kappa < 1.0, f"kappa must lie in [0,1), got {kappa}"
        if quadratic:
            A = _check_square_symmetric(A, 'A', p)      # A may be indefinite, but must be symmetric
        else:
            assert c.shape == (p,), f"c must have shape ({p},), got {c.shape}"
            assert np.all(np.isfinite(c)), "c contains NaN or inf"
        if w_init is not None:
            assert w_init.shape == (p,), f"w_init must have shape ({p},), got {w_init.shape}"

    # ---- resolve the metric: B = None means identity --> closed-form solution ----
    B_is_eye = B is None or (B.shape == (p, p) and np.abs(B - np.eye(p)).max() <= EYE_TOL)
    L_B = None
    if not B_is_eye:
        if checks:
            B = _check_square_symmetric(B, 'B', p)
            L_B = _check_metric_pd(B, 'B')              # also gives the solvers a free factorization

    sparse = kappa > 0.0

    # ---- route to the specialized solver ----
    if not quadratic and not sparse:
        # Problem A: convex, closed form.
        return solve_linear(c, B, L_B=L_B, B_is_eye=B_is_eye, **solver_kwargs)

    if not quadratic and sparse:
        # Problem B: convex but non-smooth -> proximal gradient (FISTA), or a
        # single soft-threshold when B = I (the solver branches internally).
        return solve_linear_l1(c, B, kappa, w_init=w_init, L_B=L_B, B_is_eye=B_is_eye, 
                               **solver_kwargs)

    if quadratic and not sparse:
        # Problem C with lam = 0: non-convex, but closed GEVD solution.
        return solve_quadratic(A, B, B_is_eye=B_is_eye, **solver_kwargs)

    # Problem C with lam > 0: non-convex AND non-smooth. Shift A by rho*B to make
    # it PSD (a constant offset on the constraint surface), then minorize the
    # quadratic by its tangent, which turns each MM step into Problem B.
    return solve_quadratic_l1(A, B, kappa, w_init=w_init, L_B=L_B, B_is_eye=B_is_eye, 
                              **solver_kwargs)


# ==== SHARED PROX ============================================================

def soft_threshold(v, delta):
    """Elementwise soft-thresholding operator of :cite:t:`donoho_ideal_1994`.

    The flat region at |v|<=delta is what produces exact zeros.
    """
    return np.sign(v) * np.maximum(np.abs(v) - delta, 0.0)


# ==== PROBLEM A: max c'w  s.t.  w'Bw = 1 =====================================

def solve_linear(c, B, L_B=None, B_is_eye=False, **kw):
    """Closed form. Stationarity of c'w - mu(w'Bw - 1) gives w ∝ B^-1 c."""

    # B = I: just the direction of c
    if B_is_eye:
        return c / np.linalg.norm(c)

    # Reuse the dispatcher's Cholesky factor when it was computed by the PD check.
    w = cho_solve((L_B, True), c) if L_B is not None else np.linalg.solve(B, c)

    # Normalization shortcut: w'Bw = c'B^-1 c = c'w for the unnormalized w,
    # so no second quadratic form is needed. The value is > 0 since B > 0.
    return w / np.sqrt(c @ w)


# ==== PROBLEM B: max c'w - lam||w||_1  s.t.  w'Bw <= 1 =======================

def solve_linear_l1(c, B, kappa, w_init=None, L_B=None, lam=None, B_is_eye=False,
                    max_iter=2000, tol=1e-10, **kw):
    """Convex but non-smooth -> FISTA :cite:t:`beck_fast_2009`.

    FISTA on the penalized form:
        min_w  w'Bw - 2c'w + lam||w||_1

    followed by renormalization. This is equivalent to the constrained problem:
    along a ray w = t*v with v'Bv = 1 the penalized objective is t^2 - 2t*g(v)
    with g(v) = c'v - (lam/2)||v||_1, minimized at t = g(v) with value -g(v)^2.
    So minimizing it is the same as maximizing g over the ellipsoid -- i.e. the
    constrained problem with penalty lam/2.

    `lam` overrides the kappa-derived threshold. Callers that invoke this inside
    an outer loop MUST pass a fixed lam: re-deriving it from ||c||_inf every
    iteration makes the penalty chase its own tail (c grows as w sparsifies).

    With the kappa-derived threshold, lam < 2||c||_inf for every kappa in [0, 1).
    A fixed lam, however, can reach 2||c||_inf when c changes between calls (the
    MM iterations of solve_quadratic_l1). The penalized form then thresholds every
    entry and returns w = 0, which is not on the constraint surface w'Bw = 1, so
    that case is solved exactly by solve_linear_l1_one_sparse instead.
    """

    # scale-free threshold
    if lam is None:
        lam = kappa * 2.0 * np.abs(c).max()

    # Penalty at or above every entry of the linear term: the filter cannot be zero
    # (w'Bw = 1), and the optimum on the constraint surface is 1-sparse.
    c_max = np.abs(c).max()
    if c_max > 0.0 and lam >= 2.0 * c_max:
        return solve_linear_l1_one_sparse(c, B, lam, B_is_eye=B_is_eye)

    # special case: B = I -> smooth part = ||w - c||^2 -> no gradient loop needed.
    if B_is_eye:
        w = soft_threshold(c, lam / 2.0)
    else:
        lip = 2.0 * np.linalg.eigvalsh(B)[-1]        # Lipschitz constant of grad(w'Bw - 2c'w)
        w = np.zeros_like(c) if w_init is None else w_init.copy()
        u, t = w.copy(), 1.0                         # u = momentum point, t = Nesterov sequence

        for _ in range(max_iter):
            grad = 2.0 * (B @ u - c)                 # gradient of the smooth part only
            w_new = soft_threshold(u - grad / lip, lam / lip)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            u = w_new + ((t - 1.0) / t_new) * (w_new - w)
            converged = np.linalg.norm(w_new - w) <= tol * max(np.linalg.norm(w), 1e-12)
            w, t = w_new, t_new
            if converged:
                break

    nrm = np.linalg.norm(w) if B_is_eye else np.sqrt(w @ B @ w)
    if nrm <= 1e-12:
        # Everything was thresholded although lam < 2||c||_inf (lam within rounding
        # of the bound): take the 1-sparse limit. Only c = 0 has no solution.
        assert c_max > 0.0, "solve_linear_l1 has no solution: the linear term c is zero"
        return solve_linear_l1_one_sparse(c, B, lam, B_is_eye=B_is_eye)
    w /= nrm

    return w


def solve_linear_l1_one_sparse(c, B, lam, B_is_eye=False):
    """Problem B when lam >= 2||c||_inf: the optimum is a 1-sparse filter.

    With a_i = lam/2 - |c_i| >= 0, the objective obeys

        c'w - (lam/2)||w||_1  <=  -sum_i a_i |w_i|  =:  -h(w),

    with equality when every non-zero w_i has the sign of c_i. Minimizing the
    weighted L1 norm h over the ellipsoid w'Bw = 1 is the same as maximizing the
    convex function w'Bw over the polytope h(w) <= 1, whose maximum is attained at a
    vertex, i.e. at a 1-sparse w = +-e_j / sqrt(B_jj). There h = a_j / sqrt(B_jj), and
    the bound above is attained with the sign of c_j, so the solution is

        j = argmax_j (|c_j| - lam/2) / sqrt(B_jj),    w = sign(c_j) e_j / sqrt(B_jj).

    This is the limit of the FISTA solution for lam -> 2||c||_inf from below (only
    the largest entry of c survives the threshold), so the solver is continuous in
    lam and never returns the infeasible w = 0.
    """
    b = np.ones(len(c)) if B_is_eye else np.diag(B)
    j = np.argmax((np.abs(c) - lam / 2.0) / np.sqrt(b))

    w = np.zeros(len(c))
    w[j] = (1.0 if c[j] >= 0 else -1.0) / np.sqrt(b[j])

    return w


# ==== PROBLEM C with lam = 0: max w'Aw  s.t.  w'Bw = 1 =======================

def solve_quadratic(A, B, B_is_eye=False, return_all=False, **kw):
    """Non-convex, but exactly solvable via GEVD.

    The stationarity condition is the generalized eigenvalue problem:
    A w = mu B w,  and feasibility gives w'Aw = mu, so the LARGEST 
    eigenvalue is the maximum.

    A need NOT be PSD, only symmetric: B^-1/2 A B^-1/2 is symmetric for any such
    A, so the spectrum is real and the eigenvectors B-orthogonal. Indefinite A is
    the normal case when A is a signed-weighted sum of PSD matrices.
    """
    if B_is_eye:
        mu, V = np.linalg.eigh(A)
    else:
        mu, V = eigh(A, B)

    # all filters, best first (for deflation)
    if return_all:
        return V[:, ::-1], mu[::-1]
    else:
        return V[:, -1]


# ==== PROBLEM C with lam > 0: max w'Aw - lam||w||_1  s.t.  w'Bw = 1 ==========

def solve_quadratic_l1(A, B, kappa, w_init=None, L_B=None, B_is_eye=False,
                       max_iter=100, tol=1e-6, **kw):
    """Non-convex AND non-smooth :cite:t:`sriperumbudur_majorization-minimization_2011`.

    Two reductions:

      1. SHIFT.    On the constraint surface, w'Aw = w'(A + rho B)w - rho, so adding
                   rho*B changes the objective by a CONSTANT. Pick rho so that the
                   shifted matrix is PSD; this removes the indefiniteness of A.
      2. MINORIZE. A PSD quadratic lies above its tangent, so  w'Ãw >= 2(Ãw_k)'w + const
                   is a valid MM minorant -- and it is LINEAR in w. Each step is
                   therefore Problem B with c = Ã w_k.

    Monotone ascent to a stationary point only; the caller supplies restarts.
    """
    p = A.shape[0]
    eye = B_is_eye

    # ---- step 1: shift A to PSD using the generalized spectrum of (A, B) ----
    mu = np.linalg.eigvalsh(A) if eye else eigvalsh(A, B)
    rho = max(0.0, -mu[0]) + 1e-8 * max(np.abs(mu[-1]), 1.0)
    A_shift = A + rho * (np.eye(p) if eye else B)

    # Warm start from the lam = 0 solution, which the spectrum above nearly gives us.
    w = solve_quadratic(A, B, B_is_eye=eye) if w_init is None else w_init.copy()

    # ---- fix the penalty ONCE, from the warm start ----
    # kappa stays the scale-free knob, but the absolute threshold must be frozen:
    # inside MM, ||c||_inf grows as w sparsifies, so a re-derived lam would ratchet
    # the sparsity upward every iteration and never reach a fixed point.
    lam = kappa * 2.0 * np.abs(A_shift @ w).max()

    # ---- step 2: MM, each iteration being one Problem B solve ----
    for _ in range(max_iter):
        # tangent of the shifted quadratic at w
        c = A_shift @ w
        w_new = solve_linear_l1(c, B, kappa, w_init=w, L_B=L_B, lam=lam, B_is_eye=eye, 
                                **kw)
        # Sign is free (both the objective and the constraint are even in w),
        # so align before measuring the step, or a flip reads as non-convergence.
        if w_new @ w < 0:
            w_new = -w_new
        delta = np.linalg.norm(w_new - w)
        w = w_new
        if delta <= tol:
            break

    return w
