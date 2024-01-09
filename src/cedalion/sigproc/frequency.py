"""Frequency-related signal processing methods."""

import numpy as np
import scipy.signal
import xarray as xr
import cedalion.typing as cdt
from cedalion import Quantity, units
from cedalion.validators import check_dimensionality
import cedalion.dataclasses as cdc


@cdc.validate_schemas
def sampling_rate(timeseries: cdt.NDTimeSeries) -> Quantity:
    """Estimate the sampling rate of the timeseries.

    This functions assumes uniform sampling.
    """
    assert "units" in timeseries.time.attrs
    time_unit = units.Unit(timeseries.time.attrs["units"])

    mean_diff = np.diff(timeseries.time).mean() * time_unit
    return (1.0 / mean_diff).to("Hz")


@cdc.validate_schemas
def freq_filter(
    timeseries: cdt.NDTimeSeries,
    fmin: Quantity,
    fmax: Quantity,
    butter_order=4,
):
    """Apply a Butterworth frequency filter."""

    check_dimensionality("fmin", fmin, "[frequency]")
    check_dimensionality("fax", fmax, "[frequency]")

    fny = sampling_rate(timeseries) / 2
    fmin = float(fmin / fny)
    fmax = float(fmax / fny)

    b, a = scipy.signal.butter(butter_order, (fmin, fmax), "bandpass")

    if (units := timeseries.pint.units) is not None:
        array = timeseries.pint.dequantify()

    result = xr.apply_ufunc(scipy.signal.filtfilt, b, a, array)

    if units is not None:
        result = result.pint.quantify(units)

    return result
