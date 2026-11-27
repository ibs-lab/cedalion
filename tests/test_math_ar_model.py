import numpy as np
import pandas as pd
import pytest

import cedalion.dataclasses as cdc
from cedalion.math.ar_irls import ar_irls_GLM
from cedalion.math.ar_model import bic_arfit


def _ar_process(coefs: dict[int, float], n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.zeros(n)
    pmax = max(coefs)
    for t in range(pmax, n):
        x[t] = e[t] + sum(c * x[t - lag] for lag, c in coefs.items())
    return x


def _ar_order(result) -> int:
    # params include the intercept
    return len(result.params) - 1


def test_bic_arfit_selects_global_bic_minimum():
    """An AR(2) process with a vanishing lag-1 coefficient must yield order 2.

    Regression test: the order search stopped at the first increase of the BIC,
    which for this process is at order 1, and returned order 0 (no whitening).
    """
    x = pd.Series(_ar_process({2: -0.8}, 4000))
    assert _ar_order(bic_arfit(x, pmax=6)) == 2


def test_bic_arfit_returns_pmax_when_bic_still_decreases():
    """If the minimum is at pmax, pmax is returned (not pmax - 1) with a warning."""
    x = pd.Series(_ar_process({4: -0.8}, 4000))
    with pytest.warns(UserWarning):
        result = bic_arfit(x, pmax=4)
    assert _ar_order(result) == 4


def test_ar_irls_glm_default_pmax():
    """pmax=None derives the maximum order from the sampling rate."""
    fs = 10.0
    t = np.arange(0, 400, 1 / fs)
    regressor = np.sin(2 * np.pi * 0.05 * t)
    y = 2.0 * regressor + _ar_process({1: 0.7}, len(t))

    ts = cdc.build_timeseries(
        y[None, :],
        dims=["channel", "time"],
        time=t,
        channel=["S1D1"],
        value_units="uM",
        time_units="s",
        other_coords={"source": ("channel", ["S1"]), "detector": ("channel", ["D1"])},
    ).isel(channel=0)

    x = pd.DataFrame({"hrf": regressor, "const": np.ones(len(t))})

    result = ar_irls_GLM(ts, x, pmax=None)
    assert abs(result.params["hrf"] - 2.0) < 0.3
