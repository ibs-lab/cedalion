import numpy as np
import xarray as xr

import cedalion
import cedalion.io.utils as ioutils


def test_netcdf_quantified_round_trip(tmp_path):
    """Units and dim names survive save_/load_dataarray_quantified."""
    array = xr.DataArray(np.eye(4), dims=["aligned", "ijk"]).pint.quantify("mm")

    path = str(tmp_path / "transform.nc")
    ioutils.save_dataarray_quantified(array, path)
    loaded = ioutils.load_dataarray_quantified(path)

    assert loaded.dims == array.dims
    assert loaded.pint.units == array.pint.units
    np.testing.assert_array_equal(
        loaded.pint.dequantify().values, array.pint.dequantify().values
    )


def test_load_dataarray_quantified_fallback(tmp_path):
    """Files without a units attribute fall back to the given units."""
    array = xr.DataArray(np.eye(4), dims=["aligned", "ijk"])

    path = str(tmp_path / "transform.nc")
    array.to_netcdf(path)

    assert ioutils.load_dataarray_quantified(path).pint.units is None
    assert ioutils.load_dataarray_quantified(
        path, fallback_units="cm"
    ).pint.units == cedalion.units.Unit("cm")


def test_save_dataarray_quantified_accepts_unquantified(tmp_path):
    """An unquantified array is written unchanged, units in attrs preserved."""
    array = xr.DataArray(np.eye(4), dims=["aligned", "ijk"], attrs={"units": "mm"})

    path = str(tmp_path / "transform.nc")
    ioutils.save_dataarray_quantified(array, path)

    assert ioutils.load_dataarray_quantified(path).pint.units == cedalion.units.Unit(
        "mm"
    )
