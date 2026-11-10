import itertools
import functools
import operator

import numpy as np
import pytest
from pint.testsuite.helpers import assert_quantity_almost_equal as assert_approx
import pandas as pd

import cedalion
import cedalion.dataclasses as cdc
import cedalion.models.glm as glm
import cedalion.models.glm.basis_functions as bfuncs
import cedalion.models.glm.design_matrix as dm
import cedalion.nirs as nirs
from cedalion import units


def test_avg_short_channel(rec):
    ts_long, ts_short = nirs.split_long_short_channels(
        rec["conc"], rec.geo3d, distance_threshold=1.5 * units.cm
    )

    dms = dm.average_short_channel_regressor(ts_short)
    regressor = dms.common

    assert regressor.dims == ("time", "regressor", "chromo")

    mean_hbo_0 = ts_short.sel(chromo="HbO", time=0).mean(dim="channel").item()
    mean_hbr_0 = ts_short.sel(chromo="HbR", time=0).mean(dim="channel").item()

    assert_approx(regressor.sel(chromo="HbO", time="0").item(), mean_hbo_0.magnitude)
    assert_approx(regressor.sel(chromo="HbR", time="0").item(), mean_hbr_0.magnitude)


def test_make_design_matrix_channel_only(rec):
    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
        rec["conc"], rec.geo3d, distance_threshold=1.5 * units.cm
    )

    base = (
        dm.hrf_regressors(
            ts_long,
            rec.stim,
            glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=3 * units.s),
        )
        & dm.drift_regressors(ts_long, drift_order=1)
    )

    _ = base & dm.closest_short_channel_regressor(ts_long, ts_short, rec.geo3d)
    _ = base & dm.max_corr_short_channel_regressor(ts_long, ts_short)
    _ = base & dm.average_short_channel_regressor(ts_short)


def test_short_channel_regressors_raise_in_parcel_space(rec):
    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
        rec["conc"], rec.geo3d, distance_threshold=1.5 * units.cm
    )

    ts_long_parcel = ts_long.copy().rename({"channel": "parcel"})
    ts_short_parcel = ts_short.copy().rename({"channel": "parcel"})

    with pytest.raises((AssertionError, ValueError)):
        dm.closest_short_channel_regressor(ts_long_parcel, ts_short_parcel, rec.geo3d)

    with pytest.raises((AssertionError, ValueError)):
        dm.max_corr_short_channel_regressor(ts_long_parcel, ts_short_parcel)

    with pytest.raises((AssertionError, ValueError)):
        dm.average_short_channel_regressor(ts_short_parcel)




@pytest.mark.parametrize(
    "ts_key, spectral_dim, spatial_dim",
    [
        ("conc", "chromo", "channel"),
        ("conc", "chromo", "parcel"),
        ("conc", "chromo", "vertex"),
        ("od", "wavelength", "channel"),
        ("od", "wavelength", "parcel"),
        ("od", "wavelength", "vertex"),
    ],
)
def test_make_design_matrix_combinations(rec, ts_key, spectral_dim, spatial_dim):
    ts = rec[ts_key].copy().rename({"channel": spatial_dim})

    stim = pd.DataFrame(
        {
            "onset": [3.0, 6.0],
            "duration": [2.0, 2.0],
            "value": [1.0, 1.0],
            "trial_type": ["A", "B"],
        }
    )

    basis_functions = [
        bfuncs.GaussianKernels(
            t_pre=5 * units.s,
            t_post=30 * units.s,
            t_delta=3 * units.s,
            t_std=3 * units.s,
        ),
        bfuncs.GaussianKernelsWithTails(
            t_pre=5 * units.s,
            t_post=30 * units.s,
            t_delta=3 * units.s,
            t_std=3 * units.s,
        ),
        bfuncs.Gamma(tau=0 * units.s, sigma=3 * units.s),
        bfuncs.GammaDeriv(tau=2 * units.s, sigma=2 * units.s),
        bfuncs.AFNIGamma(p=1, q=0.7 * units.s),
        bfuncs.DiracDelta()
    ]

    hrf_regressors = [dm.hrf_regressors(ts, stim, bf) for bf in basis_functions]

    nuisance_regressors = [
        dm.drift_regressors(ts, 1),
        dm.drift_regressors(ts, 2),
        dm.drift_regressors(ts, 3),
        dm.drift_legendre_regressors(ts, 1),
        dm.drift_legendre_regressors(ts, 2),
        dm.drift_legendre_regressors(ts, 3),
        dm.drift_cosine_regressors(ts, 0.02 * cedalion.units.Hz),
        dm.global_mean_regressor(ts),
    ]

    for reg_combo in itertools.product(hrf_regressors, nuisance_regressors):

        dms = functools.reduce(operator.and_, reg_combo)

        assert "time" in dms.common.dims
        assert "regressor" in dms.common.dims
        assert spectral_dim in dms.common.dims




def _hrf_test_recording(t0: float):
    fs = 10.0
    t = t0 + np.arange(0, 80, 1 / fs)
    ts = cdc.build_timeseries(
        np.zeros((1, 2, len(t))),
        dims=["channel", "chromo", "time"],
        time=t,
        channel=["S1D1"],
        value_units="uM",
        time_units="s",
        other_coords={
            "chromo": ["HbO", "HbR"],
            "source": ("channel", ["S1"]),
            "detector": ("channel", ["D1"]),
        },
    )
    stim = pd.DataFrame(
        {
            "onset": [t0 + 2.0, t0 + 40.0],
            "duration": [5.0, 5.0],
            "value": [1.0, 1.0],
            "trial_type": ["A", "A"],
        }
    )
    return ts, stim


@pytest.mark.parametrize(
    "make_basis",
    [
        lambda: glm.Gamma(tau=2 * units.s, sigma=3 * units.s, T=0 * units.s),
        lambda: glm.GaussianKernels(
            t_pre=5 * units.s,
            t_post=20 * units.s,
            t_delta=3 * units.s,
            t_std=3 * units.s,
        ),
    ],
)
def test_hrf_regressors_independent_of_time_axis_origin(make_basis):
    """HRF regressors of a recording starting at t=100 s equal those starting at 0 s.

    Regression test: when the earliest (basis-shifted) onset preceded time[0], the
    padded time axis was built relative to 0 instead of time[0], which placed the
    stimulus at time[0] instead of its true position.
    """
    regressors = {}
    for t0 in [0.0, 100.0]:
        ts, stim = _hrf_test_recording(t0)
        regressors[t0] = dm.hrf_regressors(ts, stim, make_basis()).common

    np.testing.assert_allclose(
        regressors[0.0].values, regressors[100.0].values, atol=1e-10
    )
    # the stimulus response must not be zero at the beginning of the recording
    assert np.abs(regressors[100.0].values[:60]).max() > 0
