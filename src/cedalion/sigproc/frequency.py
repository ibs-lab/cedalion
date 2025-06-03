"""Frequency-related signal processing methods."""

import numpy as np
import scipy.signal
import xarray as xr
import cedalion.typing as cdt
from cedalion import Quantity, units
from cedalion.validators import check_dimensionality
import cedalion.dataclasses as cdc
import statsmodels.api as sm


@cdc.validate_schemas
def sampling_rate(timeseries: cdt.NDTimeSeries) -> Quantity:
    """Estimate the sampling rate of the timeseries.

    Note:
        This functions assumes uniform sampling.

    Args:
        timeseries (:class:`NDTimeSeries`, (time,*)): the input time series

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
        timeseries (:class:`NDTimeSeries`, (time,*)): the input time series
        fmin (:class:`Quantity`, [frequency]): lower threshold of the pass band
        fmax (:class:`Quantity`, [frequency]): higher threshold of the pass band
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


def ar_filter(conc_ts: xr.DataArray, ar_order: int) -> xr.Dataset:
    """Apply autoregressive (AR) filtering to an fNIRS timeseries.

    Args:
    conc_ts : xr.DataArray
        The input concentration time series with dimensions (time, chromo, channel).
    ar_order : int
        The order of the autoregressive model.

    Returns:
    A dataset with three DataArrays:
        - 'original': the truncated original signal (without first ar_order points)
        - 'ar_prediction': the AR model prediction for each timeseries
        - 'residual': the residuals (filtered signal) after AR fitting
    """
    time = conc_ts.time.values[ar_order:]
    coords = {'time': time, 'chromo': conc_ts.chromo, 'channel': conc_ts.channel}
    shape = (len(time), len(conc_ts.chromo), len(conc_ts.channel))
    unit = conc_ts.pint.units if conc_ts.pint.units else 1

    original = np.full(shape, np.nan)
    prediction = np.full(shape, np.nan)
    residual = np.full(shape, np.nan)

    for idx_chromo, chromo in enumerate(conc_ts.chromo):
        for idx_ch, ch in enumerate(conc_ts.channel):
            y = conc_ts.sel(chromo=chromo, channel=ch).pint.dequantify().values
            if np.all(np.isfinite(y)):
                model = sm.tsa.AutoReg(y, lags=ar_order, old_names=False)
                result = model.fit()
                pred = result.predict(start=ar_order, end=len(y)-1)
                resid = result.resid
                original[:, idx_chromo, idx_ch] = y[ar_order:]
                prediction[:, idx_chromo, idx_ch] = pred
                residual[:, idx_chromo, idx_ch] = resid

    original_da = xr.DataArray(original, coords=coords,
                               dims=('time', 'chromo', 'channel')) * unit
    prediction_da = xr.DataArray(prediction, coords=coords,
                                dims=('time', 'chromo', 'channel')) * unit
    residual_da = xr.DataArray(residual, coords=coords,
                               dims=('time', 'chromo', 'channel')) * unit
    ds = xr.Dataset({
    'original': original_da,
    'ar_prediction': prediction_da,
    'residual': residual_da
    })

    return ds
