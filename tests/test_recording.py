"""Tests for the Recording data container."""

import numpy as np
import xarray as xr

from cedalion.dataclasses import Recording


def test_wavelengths_from_wavelength_timeseries():
    """Return unique wavelengths when wavelength data are present."""
    rec = Recording()

    rec.timeseries["amp"] = xr.DataArray(
        np.zeros((2, 2, 3)),
        dims=("channel", "wavelength", "time"),
        coords={
            "channel": ["S1D1", "S1D2"],
            "wavelength": [760.0, 850.0],
            "time": [0.0, 1.0, 2.0],
        },
    )

    assert rec.wavelengths == [760.0, 850.0]


def test_wavelengths_empty_without_wavelength_timeseries():
    """Return an empty list when no time series has wavelengths."""
    rec = Recording()

    rec.timeseries["conc"] = xr.DataArray(
        np.zeros((2, 2, 3)),
        dims=("channel", "chromo", "time"),
        coords={
            "channel": ["S1D1", "S1D2"],
            "chromo": ["HbO", "HbR"],
            "time": [0.0, 1.0, 2.0],
        },
    )

    assert rec.wavelengths == []
