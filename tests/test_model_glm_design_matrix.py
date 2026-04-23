import pytest
from pint.testsuite.helpers import assert_quantity_almost_equal as assert_approx

import cedalion
import cedalion.data
import cedalion.models.glm as glm
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


@pytest.mark.parametrize("ts_key, spectral_dim", [("conc", "chromo"), ("od", "wavelength")])
def test_make_design_matrix_parcel(rec, ts_key, spectral_dim):
    ts_parcel = rec[ts_key].copy().rename({"channel": "parcel"})

    dms = (
        dm.hrf_regressors(
            ts_parcel,
            rec.stim,
            glm.Gamma(tau=0 * units.s, sigma=3 * units.s, T=3 * units.s),
        )
        & dm.drift_regressors(ts_parcel, drift_order=1)
    )

    assert "time" in dms.common.dims
    assert "regressor" in dms.common.dims
    assert spectral_dim in dms.common.dims


