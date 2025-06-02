import numpy as np
import cedalion.dataclasses as cdc
from cedalion.typing import NDTimeSeries
import xarray as xr


@cdc.validate_schemas
def ampd(amplitudes: NDTimeSeries, chunk_size=500, step_size=200):
    """Automatic Multiscale Peak Detection (AMPD) algorithm.

    This implementation is based on the AMPD algorithm described by Scholkmann et al. in cite:t:`Scholkmann2012`
    which detects peaks in a signal using a multiscale approach and local scalogram matrix.

    Args:
        amplitudes : xarray.DataArray
            Input data array containing signal data for each channel and wavelength.
        chunk_size : int, optional
            The size of each chunk to be processed (default is 600).
        step_size : int, optional
            Step size for overlapping chunks (default is 200).

    Returns:
        peaks : xarray.DataArray
            Output DataArray with the same shape as amplitudes where detected peaks
            are marked with `1`, and non-peaks are marked with `0`.

    Initial Contributors:
        Isa Musisi | w.musisi@campus.tu-berlin.de | 2024

    """
    # Prepare output array with the same structure as amplitudes, filled with zeros
    peaks = np.zeros_like(amplitudes, dtype=int)
    # Iterate over channels and wavelengths
    for ch in range(amplitudes.shape[0]):
        for wl in range(amplitudes.shape[1]):
            # Extract the time series for the current channel and wavelength
            signal = amplitudes[ch, wl, :].values
            length = len(signal)
            final_peaks = []

            time = np.arange(1, len(signal) + 1)

            # linear detrending (equivalent to MATLAB's polyfit and polyval)
            # eliminate baseline shift
            fit_polynomial = np.polyfit(time, signal, 1)
            fit_signal = np.polyval(fit_polynomial, time)

            # Detrended signal
            dtr_signal = signal - fit_signal

            # Process the signal in overlapping chunks
            for start in range(0, length, step_size):
                end = min(start + chunk_size, length)
                dtr_chunk = dtr_signal[start:end]

                # Initialize variables
                N = len(dtr_chunk)
                L = int(np.ceil(N / 2.0)) - 1  # Half the length of the signal minus 1
                # np.random.seed(0)
                # Local Scalogram Matrix (LSM), initialized with ones
                LSM = np.ones((L, N)) + np.random.rand(L, N)

                # Generate LSM by checking for local maxima (vectorized approach)
                for k in range(1, L + 1):
                    maxima_mask = (dtr_chunk[k:N - k] > dtr_chunk[:N - 2 * k]) & (
                            dtr_chunk[k:N - k] > dtr_chunk[2 * k:])
                    LSM[k - 1, k:N - k] = (~maxima_mask) * (np.ones(len(maxima_mask)) + np.random.rand(len(maxima_mask)))  # Set LSM to 0 where condition is true

                # LSM *= np.random.rand(L, N)

                # Sum across rows of LSM to compute G
                G = np.sum(LSM, axis=1)
                l = np.argmin(G) + 1  # Find the minimum G and the corresponding scale

                # Reduce the LSM to the first 'l' rows
                LSM = LSM[:l, :]

                # Compute the standard deviation across columns of LSM
                S = np.std(LSM, axis=0)

                # Find indices where the standard deviation is zero (this indicates a peak)
                initial_peaks = np.where(S == 0)[0] + start
                final_peaks.extend(initial_peaks)

            # Mark detected peaks in the output array
            peaks[ch, wl, final_peaks] = 1

    # Convert the peaks array back into an xarray.DataArray with the same coordinates and attributes
    peaks = xr.DataArray(peaks, coords=amplitudes.coords, dims=amplitudes.dims, attrs=amplitudes.attrs)

    return peaks