"""Frequency-related signal processing methods."""

from __future__ import annotations
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

    Note:
        This functions assumes uniform sampling.

    Args:
        timeseries: the input time series, coords (time,*).

    Returns:
        The sampling rate estimated by averaging time differences between samples.
    """
    assert "units" in timeseries.time.attrs
    time_unit = units.Unit(timeseries.time.attrs["units"])

    mean_diff = np.diff(timeseries.time).mean() * time_unit
    return (1.0 / mean_diff).to("Hz") # report sampling rate in Hz


@cdc.validate_schemas
def freq_filter(
    timeseries: cdt.NDTimeSeries,
    fmin: cdt.QFrequency,
    fmax: cdt.QFrequency,
    butter_order: int = 4,
) -> cdt.NDTimeSeries:
    """Apply a Butterworth bandpass frequency filter.

    Args:
        timeseries: the input time series, coords (time,*)
        fmin: lower threshold of the pass band
        fmax: higher threshold of the pass band
        butter_order: order of the filter

    Returns:
        The frequency-filtered time series
    """

    check_dimensionality("fmin", fmin, "[frequency]")
    check_dimensionality("fax", fmax, "[frequency]")

    fny = sampling_rate(timeseries) / 2
    fmin = float(fmin / fny)
    fmax = float(fmax / fny)

    if fmin == 0:
        sos = scipy.signal.butter(butter_order, fmax, "low", output="sos")
    elif fmax == 0:
        sos = scipy.signal.butter(butter_order, fmin, "high", output="sos")
    else:
        sos = scipy.signal.butter(butter_order, [fmin, fmax], "bandpass", output="sos")

    if (units := timeseries.pint.units) is not None:
        timeseries = timeseries.pint.dequantify()


    dims = timeseries.dims
    timeseries = timeseries.transpose(..., "time")
    result = xr.apply_ufunc(scipy.signal.sosfiltfilt, sos, timeseries)
    result = result.transpose(*dims)

    if units is not None:
        result = result.pint.quantify(units)

    return result
