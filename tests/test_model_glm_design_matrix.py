import itertools
import functools
import operator

import pytest
from pint.testsuite.helpers import assert_quantity_almost_equal as assert_approx
import numpy as np
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


def _split(rec):
    return nirs.split_long_short_channels(
        rec["conc"], rec.geo3d, distance_threshold=1.5 * units.cm
    )


def test_max_corr_short_channel_skips_nans(rec):
    ts_long, ts_short = _split(rec)
    ts_short = ts_short.copy()

    # all short channels are NaN at one time point, which used to raise
    ts_short[{"time": 10}] = np.nan * units.uM
    dm.max_corr_short_channel_regressor(ts_long, ts_short)

    # a short channel that equals a long channel is selected for it, even if it has
    # a NaN segment. Before, a NaN made its correlations NaN and it was never chosen.
    same = ts_long[{"channel": 0}].transpose(*ts_short[{"channel": 0}].dims)
    ts_short[{"channel": 0}] = same.pint.to(ts_short.pint.units)
    ts_short[{"channel": 0, "time": slice(20, 30)}] = np.nan * units.uM
    result = dm.max_corr_short_channel_regressor(ts_long, ts_short)

    selected = result.channel_wise[0].short_channel.values[0]
    assert selected == ts_short.channel.values[0]


def _long_short(long, short):
    """Build long and short time series with one chromophore from 2D arrays."""

    def build(values, prefix):
        values = np.asarray(values, dtype=float)
        return cdc.build_timeseries(
            values[:, :, None],
            dims=["time", "channel", "chromo"],
            time=np.arange(len(values)) / 10.0,
            channel=[f"{prefix}{i}" for i in range(values.shape[1])],
            value_units=units.uM,
            time_units=units.s,
            other_coords={"chromo": ["HbO"]},
        )

    return build(long, "L"), build(short, "S")


def _selected(ts_long, ts_short):
    result = dm.max_corr_short_channel_regressor(ts_long, ts_short)
    return list(result.channel_wise[0].short_channel.values)


def test_max_corr_short_channel_correlates_over_common_samples():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=1000)

    # S0 equals L0 where it is finite. L0 has an offset where S0 is missing, which
    # must not lower their correlation. S1 has the offset, too, but is noisier.
    offset = np.where(np.arange(1000) < 500, 5.0, 0.0)
    long = signal + offset
    s0 = np.where(offset > 0, np.nan, signal)
    s1 = signal + offset + rng.normal(size=1000)
    ts_long, ts_short = _long_short(long[:, None], np.stack([s0, s1], axis=1))

    assert _selected(ts_long, ts_short) == ["S0"]


def test_max_corr_short_channel_skips_infs():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=1000)
    s0 = signal.copy()
    s0[100] = np.inf
    s1 = signal + rng.normal(size=1000)
    ts_long, ts_short = _long_short(signal[:, None], np.stack([s0, s1], axis=1))

    assert _selected(ts_long, ts_short) == ["S0"]


def test_max_corr_short_channel_raises_for_all_nan_long_channel():
    rng = np.random.default_rng(0)
    long = rng.normal(size=(1000, 2))
    long[:, 1] = np.nan
    ts_long, ts_short = _long_short(long, rng.normal(size=(1000, 2)))

    with pytest.raises(ValueError, match="undefined: L1"):
        dm.max_corr_short_channel_regressor(ts_long, ts_short)


def test_fill_nonfinite_masks_channel_wise_nans(rec):
    ts_long, ts_short = _split(rec)
    ts_short = ts_short.copy()
    ts_short[{"channel": 0, "chromo": 0, "time": slice(10, 20)}] = np.nan * units.uM

    dms = dm.drift_regressors(ts_long, drift_order=1) & (
        dm.closest_short_channel_regressor(ts_long, ts_short, rec.geo3d)
    )
    filled, mask = dms.fill_nonfinite(ts_long)

    regressor = dms.channel_wise[0]
    expected = np.isfinite(regressor.pint.dequantify()).all("regressor")
    assert not expected.all()  # the NaNs reached the regressor of some channels
    assert mask.dims == ts_long.dims
    np.testing.assert_array_equal(
        mask.transpose(*expected.dims).values, expected.values
    )

    assert np.isfinite(filled.channel_wise[0].pint.dequantify()).all()
    assert filled.channel_wise[0].pint.units == regressor.pint.units
    np.testing.assert_array_equal(filled.common.values, dms.common.values)

    # the finite design matrix and the mask can be passed to glm.fit
    result = glm.fit(
        ts_long, filled, "ols", max_jobs=1, mask=np.isfinite(ts_long) & mask
    )
    assert np.isfinite(result.sm.params).all()


def test_fill_nonfinite_masks_common_nans_in_all_channels(rec):
    ts_long, ts_short = _split(rec)
    ts_short = ts_short.copy()
    ts_short.loc[{"chromo": "HbO", "time": ts_short.time[10]}] = np.nan * units.uM

    dms = dm.average_short_channel_regressor(ts_short)
    filled, mask = dms.fill_nonfinite(ts_long)

    assert np.isfinite(filled.common).all()
    masked = ~mask
    assert masked.sel(chromo="HbO")[{"time": 10}].all()
    assert masked.sum() == ts_long.sizes["channel"]


def test_fill_nonfinite_raises_for_mismatching_time(rec):
    ts_long, ts_short = _split(rec)
    dms = dm.average_short_channel_regressor(ts_short)

    with pytest.raises(ValueError, match="coordinates"):
        dms.fill_nonfinite(ts_long.isel(time=slice(1, None)))
