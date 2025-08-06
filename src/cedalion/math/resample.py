import cedalion.sigproc.frequency as freq
import xarray as xr
import cedalion
import numpy as np

units = cedalion.units


def resample(data: xr.DataArray, Fs: float = 4):
    """Function for temporally resampling (up or down) a time series.

    Args:
        data: xr.DataArray
        Fs: new sample rate (as float)

    Returns:
        resampled data
    """

    data2 = data.copy()
    Fs_curr = 1 / (data2.time[1] - data2.time[0])

    # STrip off the units to avoid the warning as part of the interp function
    data2 = data2.pint.dequantify()
    if Fs_curr > Fs:
        # If downsampling, filter at the Nyquist
        data2 = freq.freq_filter(
            data2, fmin=0 * units.Hz, fmax=Fs / 2 * units.Hz, butter_order=8
        )

    dt = 1 / Fs

    new_time = np.arange(data2.time[0].to_numpy(), data2.time[-1].to_numpy(), dt)
    data2 = data2.interp(time=new_time)

    # Fix the unit stripping issue
    data2 = data2.pint.quantify(data.pint.units)

    return data2
