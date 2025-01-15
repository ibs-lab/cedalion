import numpy as np
import pint
import pytest
import xarray as xr
from numpy.testing import assert_allclose

import cedalion.dataclasses as cdc
import cedalion.datasets
import cedalion.nirs


def test_get_extinction_coefficients_notexistant():
    wavelengths = [750, 850]

    with pytest.raises(ValueError):
        cedalion.nirs.get_extinction_coefficients("nonsupported spectrum", wavelengths)


def test_get_extinction_coefficients_prahl():
    wavelengths = [750, 850]
    E = cedalion.nirs.get_extinction_coefficients("prahl", wavelengths)

    assert "chromo" in E.dims
    assert "wavelength" in E.dims
    assert E.pint.units == pint.Unit("mm^-1 / M")

    assert (E.wavelength.values == wavelengths).all()


@pytest.fixture
def ts():
    return cdc.build_timeseries(
        np.asarray([[[10, 20, 30], [30, 20, 10]]]),
        dims=["channel", "wavelength", "time"],
        time=[1, 2, 3],
        channel=["S1D1"],
        value_units="mV",
        time_units="s",
        other_coords={"wavelength": [760.0, 850.0]},
    )


def test_int2od(ts):
    od = cedalion.nirs.int2od(ts)
    assert od.pint.units == 1
    od = od.pint.dequantify()
    ch = "S1D1"
    assert_allclose(od.loc[ch, 760.0, :], [-np.log(0.5), -np.log(1.0), -np.log(1.5)])
    assert_allclose(od.loc[ch, 850.0, :], [-np.log(1.5), -np.log(1.0), -np.log(0.5)])


def test_od2conc2od():
    rec = cedalion.datasets.get_snirf_test_data()[0]

    for wl1,wl2 in [(760., 850.), (700, 900), (810, 820)]:
        amp = rec["amp"].copy()
        amp.wavelength.values[:] = [wl1, wl2]

        dpf = xr.DataArray(
            [6, 6], dims="wavelength", coords={"wavelength": [wl1, wl2]}
        )

        od1 = cedalion.nirs.int2od(rec["amp"])
        conc = cedalion.nirs.od2conc(od1, rec.geo3d, dpf, "prahl")
        od2 = cedalion.nirs.conc2od(conc, rec.geo3d, dpf, "prahl")

    assert od1.pint.units == od2.pint.units
    od1 = od1.pint.dequantify()
    od2 = od2.pint.dequantify()

    assert_allclose(
        od1.transpose("channel", "wavelength", "time"),
        od2.transpose("channel", "wavelength", "time"),
    )
