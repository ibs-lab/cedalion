
import pytest
import xarray as xr
import cedalion
import cedalion.data

@pytest.fixture
def rec():
    rec = cedalion.data.get_snirf_test_data()[0]
    rec["od"] = cedalion.nirs.cw.int2od(rec["amp"])

    # differential pathlenght factors
    dpf = xr.DataArray(
        [6, 6],
        dims="wavelength",
        coords={"wavelength": rec["amp"].wavelength},
    )

    rec["conc"] = cedalion.nirs.cw.od2conc(rec["od"], rec.geo3d, dpf, spectrum="prahl")

    return rec
