import pytest
import xarray as xr
from pint.testsuite.helpers import assert_quantity_almost_equal as assert_approx

import cedalion
import cedalion.datasets
import cedalion.models.glm as glm
import cedalion.models.glm.design_matrix as dm


@pytest.fixture
def rec():
    rec = cedalion.datasets.get_snirf_test_data()[0]
    rec["od"] = cedalion.nirs.int2od(rec["amp"])

    # differential pathlenght factors
    dpf = xr.DataArray(
        [6, 6],
        dims="wavelength",
        coords={"wavelength": rec["amp"].wavelength},
    )

    rec["conc"] = cedalion.nirs.od2conc(rec["od"], rec.geo3d, dpf, spectrum="prahl")

    return rec


def test_avg_short_channel(rec):
    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
        rec["conc"], rec.geo3d, distance_threshold=1.5 * cedalion.units.cm
    )

    regressor = dm.average_short_channel(ts_short)

    assert regressor.dims == ("time", "regressor", "chromo")

    mean_hbo_0 = ts_short.sel(chromo="HbO", time=0).mean().item()
    mean_hbr_0 = ts_short.sel(chromo="HbR", time=0).mean().item()

    assert_approx(regressor.sel(chromo="HbO", time="0").item(), mean_hbo_0.magnitude)
    assert_approx(regressor.sel(chromo="HbR", time="0").item(), mean_hbr_0.magnitude)


def test_make_design_matrix(rec):
    # split time series into two based on channel distance
    ts_long, ts_short = cedalion.nirs.split_long_short_channels(
        rec["conc"], rec.geo3d, distance_threshold=1.5 * cedalion.units.cm
    )

    for short_channel_method in [None, "closest", "max_corr", "mean"]:
        design_matrix, channel_wise_regressors = dm.make_design_matrix(
            ts_long,
            ts_short,
            rec.stim,
            rec.geo3d,
            basis_function=glm.Gamma(
                tau=0 * cedalion.units.s,
                sigma=3 * cedalion.units.s,
                T=3 * cedalion.units.s,
            ),
            drift_order=1,
            short_channel_method=short_channel_method,
        )

    # FIXME only checks that methods run.
