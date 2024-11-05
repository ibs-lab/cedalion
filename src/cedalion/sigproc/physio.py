import numpy as np
import cedalion.dataclasses as cdc
from cedalion.typing import NDTimeSeries
import xarray as xr


@cdc.validate_schemas
def ampd(amplitudes: NDTimeSeries, chunk_size=500, step_size=200):
    """
    Automatic Multiscale Peak Detection (AMPD) algorithm for xarray.DataArray
    with overlapping chunks and customized output format.

    This implementation is based on the AMPD algorithm described by Scholkmann et al.,
    which detects peaks in a signal using a multiscale approach and local scalogram matrix.

    Args:
        amplitudes : xarray.DataArray
            Input data array containing signal data for each channel and wavelength.
        chunk_size : int, optional
            The size of each chunk to be processed (default is 600).
        step_size : int, optional
            Step size for overlapping chunks (default is 200).

    Returns:
        peaks_xr : xarray.DataArray
            Output DataArray with the same shape as amplitudes where detected peaks
            are marked with `1`, and non-peaks are marked with `0`.

    References:
        Scholkmann, F., Boss, J., & Wolf, M. (2012). An Efficient Algorithm for Automatic
        Peak Detection in Noisy Periodic and Quasi-Periodic Signals. *Algorithms*, 5(4), 588-603.
        https://doi.org/10.3390/a5040588

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
    peaks_xr = xr.DataArray(peaks, coords=amplitudes.coords, dims=amplitudes.dims, attrs=amplitudes.attrs)

    return peaks_xr


@cdc.validate_schemas
def ampd_single(Signal, chunk_size=600, step_size=200):
    """
    Automatic Multiscale Peak Detection (AMPD) algorithm with vectorized operations,
    stricter peak filtering, and overlapping chunks.

    This implementation is based on the AMPD algorithm described by Scholkmann et al.,
    which detects peaks in a signal using a multiscale approach and local scalogram matrix.


    Args:
        Signal : array_like
            Input signal in which local maxima (peaks) will be detected.
        chunk_size : int, optional
            The size of each chunk to be processed (default is 600).
        step_size : int, optional
            The step size to use when iterating over the signal, allowing for overlapping chunks (default is 200).

    Returns:
        peaks : array
            Indices of the identified peaks in the input signal.

    References:
        Scholkmann, F., Boss, J., & Wolf, M. (2012). An Efficient Algorithm for Automatic
        Peak Detection in Noisy Periodic and Quasi-Periodic Signals. *Algorithms*, 5(4), 588-603.
        https://doi.org/10.3390/a5040588
    """
    # Ensure the input is a numpy array and validate it
    Signal = np.asarray(Signal, dtype=float)

    if Signal.ndim != 1:
        raise ValueError("Input argument 'Signal' must be a one-dimensional array")
    if len(Signal) <= 1:
        raise ValueError("Input argument 'Signal' must contain more than one element")

    time = np.arange(1, len(Signal) + 1)

    # Polynomial detrending (equivalent to MATLAB's polyfit and polyval)
    fit_polynomial = np.polyfit(time, Signal, 1)
    fit_signal = np.polyval(fit_polynomial, time)

    # Detrended signal
    dtr_signal = Signal - fit_signal

    final_peaks = []
    length = len(Signal)

    # Iterate over the signal in chunks with overlaps
    for start in range(0, length, step_size):
        end = min(start + chunk_size, length)
        dtr_chunk = dtr_signal[start:end]

        # Initialize variables
        N = len(dtr_chunk)
        L = int(np.ceil(N / 2.0)) - 1  # Half the length of the signal minus 1

        # Local Scalogram Matrix (LSM), initialized with ones
        LSM = np.ones((L, N)) + np.random.rand(L, N)

        # Generate LSM by checking for local maxima (vectorized approach)
        for k in range(1, L + 1):
            maxima_mask = (dtr_chunk[k:N - k] > dtr_chunk[:N - 2 * k]) & (
                    dtr_chunk[k:N - k] > dtr_chunk[2 * k:])
            LSM[k - 1, k:N - k] = (~maxima_mask) * (np.ones(len(maxima_mask)) + np.random.rand(len(maxima_mask)))

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

    final_peaks = np.unique(final_peaks)

    return np.array(final_peaks)


@cdc.validate_schemas
def ampd_orig(Signal):
    """
    Automatic Multiscale-based Peak Detection (AMPD) algorithm.

    Args:
        Signal : array_like
            A real 1-D array of data from which local maxima will be found.

    Returns:
        index : ndarray
            Indices of the identified peaks in `Signal`.

    References:
        Scholkmann, F., Boss, J., & Wolf, M. (2012). An Efficient Algorithm for Automatic
        Peak Detection in Noisy Periodic and Quasi-Periodic Signals. *Algorithms*, 5(4), 588-603.
        https://doi.org/10.3390/a5040588
    """
    # Input validation
    if not isinstance(Signal, np.ndarray) or Signal.ndim != 1:
        raise ValueError("Input argument 'Signal' must be a 1-dimensional array")
    if Signal.dtype != float:
        raise ValueError("Data type of input argument 'Signal' must be 'float'")

    # linear detrending
    time = np.arange(len(Signal))
    fitPolynomial = np.polyfit(time, Signal, 1)
    fitSignal = np.polyval(fitPolynomial, time)
    dtrSignal = Signal - fitSignal

    # Initialize variables
    N = len(dtrSignal)
    L = int(np.ceil(N / 2.0)) - 1
    # np.random.seed(0)  # For reproducibility of random numbers
    LSM = np.random.rand(L, N) + np.ones((L, N))  # r + alpha (alpha = 1)

    # Filling in the Local Scalogram Matrix (LSM)
    for k in range(1, L + 1):
        for i in range(k + 1, N - k):
            if dtrSignal[i] > dtrSignal[i - k] and dtrSignal[i] > dtrSignal[i + k]:
                LSM[k - 1, i] = 0

    # Summing over the scalogram matrix to find the scale with the minimum values
    G = np.sum(LSM, axis=1)
    l = np.argmin(G) + 1

    # Truncate the LSM matrix and compute the standard deviation across columns
    LSM_truncated = LSM[:l, :]
    S = np.std(LSM_truncated, axis=0)

    # Indices where the standard deviation is zero
    peak_indices = np.where(S == 0)[0]

    return peak_indices
