
import pytest
import xarray as xr
# import numpy as np
# import pandas as pd

import cedalion
import cedalion.data
# import cedalion.dataclasses as cdc
# import cedalion.models.glm as glm

# from cedalion import units


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