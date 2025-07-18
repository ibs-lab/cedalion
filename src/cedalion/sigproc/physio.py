"""Algorithms for handling physiogical components in fNIRS data."""

import numpy as np
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from sklearn.decomposition import PCA
import xarray as xr

@cdc.validate_schemas
def global_component_subtract(
    ts: cdt.NDTimeSeries,
    ts_weights: xr.DataArray = None,
    k: float = 0,
    spatial_dim: str = "channel",
    spectral_dim: str = None
) -> tuple:
    """Remove global (physiological) components from a time series by either weighted‐mean subtraction (if k=0) or PCA (if k>0).

    Returns both the corrected time series and the global component that was removed:
    the weighted‐mean regressor if k=0, or the average of backprojected  principal component time series if k>0.

    Parameters:
        ts : amplitudes (:class:`NDTimeSeries`):
            Input DataArray. Must have a "time" dimension, one dimension for space ("spatial_dim")
            (default is "channel", can be "vertex" or "parcel") and one for spectral info ("wavelength" or "chromophore").
        ts_weights : xr.DataArray, optional
            A DataArray of per‐(spatial_dim × spectral_dim) weigths. This is typically 1/(channel variance). 
            If None, all weights = 1 (no weighting). Must have same non-time dims as ts.
        k : float, default=0
            • k = 0: perform weighted‐mean subtraction (per spectral dim, e.g. HbX or wavelength).
            • k ≥ 1: remove the first int(k) principal components per spectral dimension.
            • 0 < k < 1: remove the minimum number of PCs whose cumulative explained variance ≥ k.
        spatial_dim : str, default "channel"
            Name of the spatial dimension, like channel, vertex or parcel, across PCA or averaging is performed. If absent, no subtraction is done.
        spectral_dim : str, optional
            Name of the spectral dimension (e.g. "wavelength" or "chromophore"). If None, inferred
            as the dimension in ts.dims that is neither "time" nor spatial_dim. #FIXME for more dimensions

    Returns:
        corrected : (:class:`NDTimeSeries`):
            The time series with global (physiological) components removed.
        global_component : (:class:`NDTimeSeries`):
            If k=0: the weighted‐mean regressor (dims: "time", spectral_dim).
            If k>0: the reconstructed PCA component(s) averaged across all channels (dims: "time", spectral_dim).

    Initial Contributors:
        Alexander von Lühmann | vonluehmann@tu-berlin.de | 2025
    """

    # Ensure “time” or "reltime" exists
    if "time" not in ts.dims:
        raise ValueError("Input ts must have a 'time' dimension.")

    # Infer spectral_dim if not provided
    if spectral_dim is None:
        other_dims = [d for d in ts.dims if d not in ("time", spatial_dim)]
        if len(other_dims) != 1:
            raise ValueError(
                f"Could not infer spectral_dim from ts.dims {ts.dims}. "
                "Please supply spectral_dim explicitly."
            )
        spectral_dim = other_dims[0]

    # If channel_dim is absent, simply return ts and a zeroed global_component
    if spatial_dim not in ts.dims:
        corrected = ts.copy()
        zero_global = xr.zeros_like(ts.isel({spatial_dim: 0})).drop_vars(spatial_dim)
        return corrected, zero_global

    # Validate that spectral_dim is indeed in ts.dims
    if spectral_dim not in ts.dims:
        raise ValueError(f"Spectral dimension '{spectral_dim}' not in ts.dims {ts.dims}.")

    # Preserve pint‐units if present, then strip them for numeric ops
    if hasattr(ts, "pint"):
        orig_units = ts.pint.units
        ts_vals = ts.pint.dequantify().copy()
    else:
        orig_units = 1
        ts_vals = ts.copy()

    # Build or validate ts_weights (variances)
    if ts_weights is None:
        weights = xr.ones_like(ts_vals.isel(time=0).drop_vars("time"))
    else:
        if not isinstance(ts_weights, xr.DataArray):
            raise ValueError("ts_weights must be an xarray.DataArray with dims (channel_dim, spectral_dim).")
        if set(ts_weights.dims) != {spatial_dim, spectral_dim}:
            raise ValueError(
                f"ts_weights must have dims ({spatial_dim},{spectral_dim}), but got {ts_weights.dims}."
            )
        if hasattr(ts_weights, "pint"):
            weights = ts_weights.pint.dequantify().copy()
        else:
            weights = ts_weights.copy()

    # Pull out coords and sizes
    time_coord = ts_vals.coords["time"]
    chan_coord = ts_vals.coords[spatial_dim]
    spec_coord = ts_vals.coords[spectral_dim]

    n_time = ts_vals.sizes["time"]
    n_chan = ts_vals.sizes[spatial_dim]

    # Prepare an empty DataArray for the corrected output (plain‐float)
    corrected = xr.zeros_like(ts_vals)

    # Prepare a DataArray for the global component
    global_comp = xr.DataArray(
        np.zeros((n_time, ts_vals.sizes[spectral_dim])),
        dims=("time", spectral_dim),
        coords={"time": time_coord, spectral_dim: spec_coord},
    )

    # ──────────────────────────────────────────────────────────────────────────────
    # CASE A: k == 0 → Weighted mean subtraction across channel_dim
    # ──────────────────────────────────────────────────────────────────────────────
    if k == 0:
        # Compute per‐(channel,spectral) weight = 1/var
        w = weights  # dims: (channel_dim, spectral_dim)

        # For each spectral slice s, build the global regressor:
        numerator = (ts_vals * w).sum(dim=spatial_dim)   # dims: (time, spectral)
        denominator = w.sum(dim=spatial_dim)             # dims: (spectral)
        gms = numerator / denominator                     # dims: (time, spectral)

        # Save gms as the global component
        global_comp[:] = gms

        # Next, find the best-fit scalar for each (channel, spectral)
        num = (ts_vals * gms).sum(dim="time")             # dims: (channel, spectral)
        denom = (gms * gms).sum(dim="time")               # dims: (spectral)
        scl = num / denom                                 # dims: (channel, spectral)

        # Build and subtract full component
        full_component = scl * gms  # dims: (time, channel, spectral)
        corrected = ts_vals - full_component

    # ──────────────────────────────────────────────────────────────────────────────
    # CASE B: k > 0 → PCA‐based removal (PCA always on channel_dim)
    # ──────────────────────────────────────────────────────────────────────────────
    else:
        # Determine if k is a fraction (0<k<1) or integer (k>=1)
        if 0 < k < 1:
            remove_frac = True
        else:
            remove_frac = False
            n_remove_int = int(np.floor(k))
            if n_remove_int < 1:
                raise ValueError(f"Invalid k={k}. For PCA, int(k) must be ≥ 1.")

        # Loop over each spectral slice independently
        for s in spec_coord.values:
            # Explicitly transpose so that data_matrix is (time, channel)
            ts_slice = ts_vals.sel({spectral_dim: s}).transpose("time", spatial_dim)
            data_matrix = ts_slice.values  # shape = (n_time, n_chan)

            # Pull out per‐channel variance for this slice. Since weights are assumed 1/var this is straightforward.
            # However, note that if weithts other than variance are provided, they will be treated as squared input for PCA.
            channel_vars = 1 / weights.sel({spectral_dim: s}).values  # shape = (n_chan,)

            # Pre‐whiten if weights provided
            if ts_weights is not None:
                sqrt_vars = np.sqrt(channel_vars)
                sqrt_vars = np.where(sqrt_vars == 0, 1.0, sqrt_vars)
                data_w = data_matrix / sqrt_vars[np.newaxis, :]  # shape = (n_time, n_chan)
            else:
                data_w = data_matrix.copy()

            # Decide how many PCs to remove
            if remove_frac:
                pca_full = PCA()
                pca_full.fit(data_w)
                cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                n_remove = np.searchsorted(cumvar, k) + 1
                n_remove = min(n_remove, n_chan - 1)
            else:
                n_remove = n_remove_int

            # No PCs to remove → global_comp = 0
            if n_remove < 1:
                corrected_slice = data_matrix
                global_comp.loc[:, s] = 0.0
            else:
                # Fit PCA(n_remove) on data_w (rows=time, cols=channel)
                pca = PCA(n_components=n_remove)
                scores = pca.fit_transform(data_w)             # (n_time, n_remove)
                reconstructed_w = pca.inverse_transform(scores)  # (n_time, n_chan)

                # Reconstruct in original units (also denormalize) and subtract
                if ts_weights is not None:
                    reconstructed = reconstructed_w * sqrt_vars[np.newaxis, :]
                else:
                    reconstructed = reconstructed_w

                corrected_slice = data_matrix - reconstructed

                # Assign the removed PCs averaged across channels to global_comp
                global_comp.loc[:, s] = np.mean(reconstructed, axis=1)

            # Write corrected_slice back into `corrected`
            corrected.loc[{spectral_dim: s}] = xr.DataArray(
                corrected_slice,
                dims=("time", spatial_dim),
                coords={"time": time_coord, spatial_dim: chan_coord},
            )

    # 1Reattach pint‐units
    if orig_units is not None and orig_units != 1:
        corrected = corrected * orig_units
        global_comp = global_comp * orig_units

    # Preserve any original attributes
    corrected.attrs = ts.attrs.copy()
    global_comp.attrs = {"description": "Global physiological component removed"}

    return corrected, global_comp


@cdc.validate_schemas
def ampd(amplitudes: cdt.NDTimeSeries, chunk_size: int = 500, step_size: int = 200):
    """Automatic Multiscale Peak Detection (AMPD) algorithm.

    This implementation is based on the AMPD algorithm described in
    :cite:t:`Scholkmann2012` which detects peaks in a signal using a multiscale approach
    and local scalogram matrix.

    Args:
        amplitudes : (:class:`NDTimeSeries`)
            Input data array containing signal data for each channel and wavelength.
        chunk_size : int, optional
            The size of each chunk to be processed (default is 600).
        step_size : int, optional
            Step size for overlapping chunks (default is 200).

    Returns:
        peaks : xarray.DataArray
            Output DataArray with the same shape as amplitudes where detected peaks
            are marked with `1`, and non-peaks are marked with `0`.

    References:
        Original paper: :cite:`Scholkmann2012`

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
                    maxima_mask = (dtr_chunk[k : N - k] > dtr_chunk[: N - 2 * k]) & (
                        dtr_chunk[k : N - k] > dtr_chunk[2 * k :]
                    )
                    LSM[k - 1, k : N - k] = (~maxima_mask) * (
                        np.ones(len(maxima_mask)) + np.random.rand(len(maxima_mask))
                    )  # Set LSM to 0 where condition is true

                # LSM *= np.random.rand(L, N)

                # Sum across rows of LSM to compute G
                G = np.sum(LSM, axis=1)
                ll = np.argmin(G) + 1  # Find the minimum G and the corresponding scale

                # Reduce the LSM to the first 'l' rows
                LSM = LSM[:ll, :]

                # Compute the standard deviation across columns of LSM
                S = np.std(LSM, axis=0)

                # Find indices where the standard deviation is zero
                # (this indicates a peak)
                initial_peaks = np.where(S == 0)[0] + start
                final_peaks.extend(initial_peaks)

            # Mark detected peaks in the output array
            peaks[ch, wl, final_peaks] = 1

    # Convert the peaks array back into an xarray.DataArray with the same coordinates
    # and attributes
    peaks = xr.DataArray(
        peaks, coords=amplitudes.coords, dims=amplitudes.dims, attrs=amplitudes.attrs
    )

    return peaks
