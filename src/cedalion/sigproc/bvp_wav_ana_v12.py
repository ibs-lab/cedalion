# Version 11 --> 12
#  - Umstellung: Input = Time-Series.
#  - extract_waveforms: Normalization 3 aus innerster for-Schleife herausgezogen
#       und rescaling (mean von 0 bis 1) ergänzt (Rechendauer vorher 37s nachher 20s)
#  - Plots aktualisiert
#  - BVP spezifische peakseek Korrektur wieder aus peakseek herausgenommen und
#       in extract_waveforms integriert.
#  - classify_waveforms: Klassifizierung nach delta hinzugefügt.

# Nochmal überprüfen
#  - interpft auch für downsampling erweitert

import numpy as np
from numpy.typing import ArrayLike
from typing import Dict, Tuple, Any
from scipy.interpolate import PchipInterpolator
from scipy.signal import detrend
from scipy.stats import zscore
from skmisc.loess import loess
import tkinter as tk
import xarray as xr
from pycwt.wavelet import _check_parameter_wavelet, cwt, wct_significance
from pycwt.helpers import ar1

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

import cedalion.typing as cdt
from cedalion import physunits
import cedalion.dataclasses as cdc
from cedalion.sigproc.frequency import sampling_rate, freq_filter
from cedalion.dataclasses.bvp_container import BVP_Container


# --- Helper functions used only in BVP Analysis -------------------

def dialog_artefact_removal() -> None:
    dialog = tk.Tk()
    dialog.title("Request - Artefact Removal done?")
    dialog.resizable(False, False)

    dialog.attributes("-topmost", True)
    dialog.lift()
    dialog.focus_force()
    dialog.grab_set()

    label = tk.Label(dialog,
                     text="ATTENTION!\nArtefact removal is necessary before calculating BVP parameters!",  # noqa: E501
                     font=('Calibir', 14),
                     justify="center")
    label.pack(padx=20, pady=20)

    result = {"choice": None}

    def do_continue():
        dialog.grab_release()
        result["choice"] = "continue"
        dialog.destroy()

    def do_abort():
        result["choice"] = "abort"
        dialog.destroy()

    btn_frame = tk.Frame(dialog)
    btn_frame.pack(pady=20)

    btn_continue = tk.Button(btn_frame, text="Continue",
                             command=do_continue, font=('Calibri', 12),
                             width=10)
    btn_continue.pack(side="left", padx=20)

    btn_abort = tk.Button(btn_frame, text="Abort",
                          command=do_abort,
                          font=('Calibri', 12), width=10)
    btn_abort.pack(side="right", padx=20)

    dialog.mainloop()
    return result["choice"]

import numpy as np


def interpft(x, ny):
    """Periodically resample a one-dimensional signal using Fourier interpolation.

    Parameters
    ----------
    x : array_like
        One period of an equally spaced signal.
    ny : int
        Desired number of output samples.

    Returns
    -------
    y : ndarray
        Periodically resampled signal with length `ny`.

    Notes
    -----
    - ny > len(x): Fourier upsampling through spectral zero-padding.
    - ny < len(x): Fourier downsampling through spectral truncation.
    - The signal is assumed to be periodic.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array.")

    if not isinstance(ny, (int, np.integer)):
        raise TypeError("ny must be an integer.")

    if ny < 1:
        raise ValueError("ny must be at least 1.")

    m = x.size

    if m == 0:
        raise ValueError("x must contain at least one sample.")

    # No resampling is necessary.
    if ny == m:
        return x.copy()

    X = np.fft.fft(x)
    Y = np.zeros(ny, dtype=complex)

    # Upsampling
    if ny > m:

        # Odd input length:
        # X = [DC, positive frequencies, negative frequencies]
        # There is no unpaired Nyquist coefficient.
        if m % 2 == 1:
            k = (m - 1) // 2

            # DC and positive frequencies
            Y[:k + 1] = X[:k + 1]

            # Negative frequencies
            if k > 0:
                Y[-k:] = X[-k:]

        # Even input length:
        # X[m // 2] is an unpaired Nyquist coefficient.
        # It must be split between positive and negative frequencies.
        else:
            k = m // 2

            # DC and positive frequencies below Nyquist
            Y[:k] = X[:k]

            # Negative frequencies above negative Nyquist
            if k > 1:
                Y[-(k - 1):] = X[-(k - 1):]

            # Split the original Nyquist coefficient
            Y[k] = X[k] / 2
            Y[-k] = X[k] / 2

    # Downsampling
    else:

        # Odd output length:
        # Retain DC and matching positive/negative frequency pairs.
        # There is no output Nyquist coefficient.
        if ny % 2 == 1:
            k = (ny - 1) // 2

            # DC and positive frequencies
            Y[:k + 1] = X[:k + 1]
            # Negative frequencies
            if k > 0:
                Y[-k:] = X[-k:]

        # Even output length:
        # The frequencies +k and -k become one output Nyquist bin,
        # so their coefficients must be combined.
        else:
            k = ny // 2

            # DC and positive frequencies below target Nyquist
            Y[:k] = X[:k]
            # Negative frequencies above target negative Nyquist
            if k > 1:
                Y[-(k - 1):] = X[-(k - 1):]
            # Unite the positive and negative target-Nyquist pair
            Y[k] = X[k] + X[-k]

    # Convert the resized spectrum back to the sample domain.
    y = np.fft.ifft(Y)

    # Correct for NumPy's inverse-FFT normalization.
    y *= ny / m

    # Remove insignificant imaginary round-off for real input.
    if np.isrealobj(x):
        y = y.real

    return y

def peakseek(
    x: ArrayLike,
    minpeakdist: int = 1,
    minpeakh: float | None = None
    ) -> Tuple[np.ndarray, np.ndarray]:

    x = np.asarray(x).flatten()

    # 1. Local maxima (including flat peaks)
    locs = np.where((x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1

    # 2. Apply minimum peak height
    if minpeakh is not None:
        locs = locs[x[locs] > minpeakh]

    # 3. Enforce minimum distance between peaks
    if minpeakdist > 1:
        while True:
            d = np.diff(locs) < minpeakdist
            if not np.any(d):
                break

            pks = x[locs]

            # Indices of violating pairs
            idx = np.where(d)[0]

            # Compare left and right peaks
            left_vals = pks[idx]
            right_vals = pks[idx + 1]

            # If right peak is higher → remove left
            remove_left = left_vals < right_vals

            # If left peak is higher or equal → remove right
            remove_right = ~remove_left

            # Convert pair positions to indices in locs
            del_idx = np.concatenate([
                idx[remove_left],         # left elements
                (idx + 1)[remove_right]   # right elements
            ])

            locs = np.delete(locs, del_idx)

    return locs, x[locs]

def bvp_single_ch(conc_ts: np.ndarray,
                  fs: float,
                  fs_new: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # --- Resample (upsample) the conc time series (new fs = 50 Hz) ---
    # Normalize data so that it starts from zero
    conc_ts = conc_ts - conc_ts[0]

    Duration_fs = len(conc_ts) / fs
    fs_ratio = fs_new / fs

    # --- Create time vectors (seconds) ---
    t_s_new = np.linspace(0, Duration_fs, int(fs_ratio * len(conc_ts)))
    t_s = np.linspace(0, Duration_fs, len(conc_ts))

    # --- PCHIP interpolation to preserve waveform shape ---
    pchip = PchipInterpolator(t_s, conc_ts)
    concTS_fs_new = pchip(t_s_new)

    # --- High-pass filtering via LOWESS smoothing ---
    windowsize_s = 7
    model = loess(t_s_new, concTS_fs_new,
                  span = windowsize_s * fs_new / len(concTS_fs_new),
                  degree=2, family='gaussian')
    model.fit()
    concTS_fs_new_trend = model.outputs.fitted_values

    # --- Remove trend → BVP time series ---
    bvp_ts = concTS_fs_new - concTS_fs_new_trend

    return bvp_ts, concTS_fs_new, concTS_fs_new_trend

def cmap_parula():

    parula_data = np.array([
        [0.2081, 0.1663, 0.5292],
        [0.2116, 0.1898, 0.5777],
        [0.2123, 0.2138, 0.6270],
        [0.2081, 0.2386, 0.6771],
        [0.1959, 0.2645, 0.7279],
        [0.1707, 0.2919, 0.7792],
        [0.1253, 0.3242, 0.8303],
        [0.0591, 0.3598, 0.8683],
        [0.0117, 0.3875, 0.8820],
        [0.0057, 0.4086, 0.8828],
        [0.0165, 0.4266, 0.8786],
        [0.0329, 0.4430, 0.8720],
        [0.0498, 0.4586, 0.8641],
        [0.0629, 0.4737, 0.8554],
        [0.0723, 0.4887, 0.8467],
        [0.0779, 0.5040, 0.8384],
        [0.0793, 0.5200, 0.8312],
        [0.0749, 0.5375, 0.8263],
        [0.0641, 0.5570, 0.8240],
        [0.0488, 0.5772, 0.8228],
        [0.0343, 0.5966, 0.8199],
        [0.0265, 0.6137, 0.8135],
        [0.0239, 0.6287, 0.8038],
        [0.0231, 0.6418, 0.7913],
        [0.0228, 0.6535, 0.7768],
        [0.0267, 0.6642, 0.7607],
        [0.0384, 0.6743, 0.7436],
        [0.0590, 0.6838, 0.7254],
        [0.0843, 0.6928, 0.7062],
        [0.1133, 0.7015, 0.6859],
        [0.1453, 0.7098, 0.6646],
        [0.1801, 0.7177, 0.6424],
        [0.2178, 0.7250, 0.6193],
        [0.2586, 0.7317, 0.5954],
        [0.3022, 0.7376, 0.5712],
        [0.3482, 0.7424, 0.5473],
        [0.3953, 0.7459, 0.5244],
        [0.4420, 0.7481, 0.5033],
        [0.4871, 0.7491, 0.4840],
        [0.5300, 0.7491, 0.4661],
        [0.5709, 0.7485, 0.4494],
        [0.6099, 0.7473, 0.4337],
        [0.6473, 0.7456, 0.4188],
        [0.6834, 0.7435, 0.4044],
        [0.7184, 0.7411, 0.3905],
        [0.7525, 0.7384, 0.3768],
        [0.7858, 0.7356, 0.3633],
        [0.8185, 0.7327, 0.3498],
        [0.8507, 0.7299, 0.3360],
        [0.8824, 0.7274, 0.3217],
        [0.9139, 0.7258, 0.3063],
        [0.9450, 0.7261, 0.2886],
        [0.9739, 0.7314, 0.2666],
        [0.9938, 0.7455, 0.2403],
        [0.9990, 0.7653, 0.2164],
        [0.9955, 0.7861, 0.1967],
        [0.9880, 0.8066, 0.1794],
        [0.9789, 0.8271, 0.1633],
        [0.9697, 0.8481, 0.1475],
        [0.9626, 0.8705, 0.1309],
        [0.9589, 0.8949, 0.1132],
        [0.9598, 0.9218, 0.0948],
        [0.9661, 0.9514, 0.0755],
        [0.9763, 0.9831, 0.0538],
    ])

    parula = LinearSegmentedColormap.from_list("parula", parula_data)

    return parula

def wct(
    y1,
    y2,
    dt,
    dj=1 / 12,
    s0=-1,
    J=-1,
    sig=True,
    significance_level=0.95,
    wavelet="morlet",
    normalize=True,
    **kwargs):
    """Copy of Wavelet coherence transform (WCT) from pycwt with two modifications.

    1)
    Original: aWCT = numpy.angle(W12)
    Modification: numpy.angle(S12)
    2)
    Original: return WCT, aWCT, coi, freq, sig
    Modification: return S12, S1, S2, WCT, aWCT, coi, freq, sig

    The WCT finds regions in time frequency space where the two time
    series co-vary, but do not necessarily have high power.

    Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.
    dj : float, optional
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    s0 : float, optional
        Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
        Number of scales less one. Scales range from s0 up to
        s0 * 2**(J * dj), which gives a total of (J + 1) scales.
        Default is J = (log2(N*dt/so))/dj.
    sig : bool
        set to compute signficance, default is True
    significance_level (float, optional) :
        Significance level to use. Default is 0.95.
    normalize (boolean, optional) :
        If set to true, normalizes CWT by the standard deviation of
        the signals.

    Returns:
    -------
    WCT : magnitude of coherence
    aWCT : phase angle of coherence
    coi (array like):
        Cone of influence, which is a vector of N points containing
        the maximum Fourier period of useful information at that
        particular time. Periods greater than those are subject to
        edge effects.
    freq (array like):
        Vector of Fourier equivalent frequencies (in 1 / time units)    coi :
    sig :  Significance levels as a function of scale
       if sig=True when called, otherwise zero.
    S12 : smoothed, scale-corrected Cross-Wavelet-Transform
    S1: smoothed continous wavelet transform of signal 1
    S2: smoothed continous wavelet transform of signal 2

    See also:
    --------
    cwt, xwt

    """  # noqa: D405

    wavelet = _check_parameter_wavelet(wavelet)

    # Checking some input parameters
    if s0 == -1:
        # smallest scale
        s0 = 2 * dt / wavelet.flambda()
    if J == -1:
        # Number of scales
        J = int(np.round(np.log2(y1.size * dt / s0) / dj))

    # Makes sure input signals are np arrays.
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    # Calculates the standard deviation of both input signals.
    std1 = y1.std()
    std2 = y2.std()
    # Normalizes both signals, if appropriate.
    if normalize:
        y1_normal = (y1 - y1.mean()) / std1
        y2_normal = (y2 - y2.mean()) / std2
    else:
        y1_normal = y1
        y2_normal = y2

    # Calculates the CWT of the time-series making sure the same parameters
    # are used in both calculations.
    _kwargs = dict(dj=dj, s0=s0, J=J, wavelet=wavelet)
    W1, sj, freq, coi, _, _ = cwt(y1_normal, dt, **_kwargs)
    W2, sj, freq, coi, _, _ = cwt(y2_normal, dt, **_kwargs)

    scales1 = np.ones([1, y1.size]) * sj[:, None]
    scales2 = np.ones([1, y2.size]) * sj[:, None]

    # Smooth the wavelet spectra before truncating.
    S1 = wavelet.smooth(np.abs(W1) ** 2 / scales1, dt, dj, sj)
    S2 = wavelet.smooth(np.abs(W2) ** 2 / scales2, dt, dj, sj)

    # Now the wavelet transform coherence
    W12 = W1 * W2.conj()
    scales = np.ones([1, y1.size]) * sj[:, None]
    S12 = wavelet.smooth(W12 / scales, dt, dj, sj)
    WCT = np.abs(S12) ** 2 / (S1 * S2)
    aWCT = np.angle(S12)

    # Calculates the significance using Monte Carlo simulations with 95%
    # confidence as a function of scale.
    if sig:
        a1, b1, c1 = ar1(y1)
        a2, b2, c2 = ar1(y2)

        sig = wct_significance(
            a1,
            a2,
            dt=dt,
            dj=dj,
            s0=s0,
            J=J,
            significance_level=significance_level,
            wavelet=wavelet,
            **kwargs,
        )
    else:
        sig = np.asarray([0])

    return S12, S1, S2, WCT, aWCT, coi, freq, sig

# --- BVP Analysis -------------------

def extract_bvp(hbo_conc_ts: cdt.NDTimeSeries, fs_new=50, request=True) -> cdt.NDTimeSeries:
    """Extracts the blood volume pulsation (BVP) time series from an
    HbO concentration time series.

    IMPORTANT: Artefact removal must be performed before calling this function.

    Args:
        hbo_conc_ts: HbO concentration time series from which the BVP
            signal should be extracted.
        fs_new: new sampling rate
        request: If True user will be asked, if artefact removal has been done.

    Returns:
        NDTimeSeries containing:
            - bvp_ts: extracted blood volume pulsation
            - hbo_conc_ts_50hz: resampled HbO signal at 50 Hz
            - low_freq_trend: estimated low-frequency trend

    Example:
        bvp_ts = extract_bvp(rec["conc"].sel(chromo="HbO"))
    """  # noqa: D205

    # --- Ask user to confirm artefact removal to avoid invalid analysis
    if request:
        choice = dialog_artefact_removal()
        if choice == "abort":
            print("\n\n ----- BVP analysis aborted by User -----\n\n")
            return

    # --- Determine original sampling rate in Hz
    fs_qty = sampling_rate(hbo_conc_ts)
    fs = float(fs_qty.to('Hz').magnitude)

    # --- Target sampling rate for BVP analysis
    fs_new = fs_new

    # --- Extract channel list and generate new time vector after resampling
    ch_list = hbo_conc_ts.channel.values
    time_s_new = np.linspace(
        0,
        hbo_conc_ts.time.values[-1],
        int((fs_new / fs) * hbo_conc_ts.sizes["time"]))

    # --- Preallocate data array:
    #     dimensions: channel × compound × time
    #     compounds: [bvp_ts, hbo_conc_ts_50hz, low_freq_trend]
    place_holder = np.zeros((len(ch_list), 3, len(time_s_new)))

    # --- Create output NDTimeSeries with proper metadata and coordinates
    bvp_ts = cdc.schemas.build_timeseries(
        place_holder,
        ["channel", "compound", "time"],
        time_s_new,
        ch_list,
        'uM',
        's',
        {
            "compound": ("compound", ["bvp_ts", "hbo_conc_ts_50hz", "low_freq_trend"]),
            "samples": ("time", np.arange(0, len(time_s_new))),
            "channel": ch_list,
            "source": ("channel", hbo_conc_ts.source.values),
            "detector": ("channel", hbo_conc_ts.detector.values),
        }
    )

    # --- Process each channel independently
    for ch in ch_list:
        # Select single-channel HbO signal
        actual_ts = hbo_conc_ts.sel(channel=ch)
        actual_ts_np = actual_ts.to_numpy()

        # Extract BVP, resampled signal, and low-frequency trend
        bvp_single_ts, hbo_conc_ts_50hz, low_freq_trend = bvp_single_ch(
            actual_ts_np, fs, fs_new
        )

        # Store results with correct physical units
        bvp_ts.loc[{"channel": ch, "compound": "bvp_ts"}] = (
            bvp_single_ts * physunits.units.uM
        )
        bvp_ts.loc[{"channel": ch, "compound": "low_freq_trend"}] = (
            low_freq_trend * physunits.units.uM
        )
        bvp_ts.loc[{"channel": ch, "compound": "hbo_conc_ts_50hz"}] = (
            hbo_conc_ts_50hz * physunits.units.uM
        )

    return bvp_ts

def extract_waveforms(
    bvp_ts: cdt.NDTimeSeries
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]]
    ]:
    """Extracts and normalizes single BVP waveforms from a blood volume
    pulsation (BVP) time series.

    For each channel, diastolic minima are detected and used to segment
    the BVP signal into individual pulse waveforms. Non-physiological
    segments are rejected based on local duration statistics.
    Each valid waveform is then normalized in amplitude and time.

    Processing steps per channel:
        1. Detection of diastolic minima (waveform boundaries)
        2. Extraction of waveform segments between consecutive minima
        3. Rejection of non-physiological segments (duration-based)
        4. Detection of systolic maxima within each waveform
        5. Normalization:
           - linear detrending (y-normalization)
           - resampling to fixed length of 100 samples (x-normalization)
           - z-scoring

    Args:
        bvp_ts: BVP time series (compound="bvp_ts") created by extract_bvp.

    Returns:
        Tuple of two dictionaries, both keyed by channel name:

        1. output_user:
            Compact waveform features intended for downstream or
            user-facing analysis:
                - bvp_max_value: systolic peak amplitudes
                - bvp_max_idx: indices of systolic peaks
                - bvp_min_value: diastolic minima amplitudes
                - bvp_min_idx: indices of diastolic minima

        2. output_details:
            Detailed waveform data intended for inspection, debugging,
            or advanced analyses:
                - list_wav_raw_and_y_normal: raw and detrended waveforms
                  with corresponding time vectors
                - nparray_wav_xy_normal_all: matrix of resampled waveforms
                - nparray_wav_xy_normal_zscore_all: z-scored waveform matrix

    Example:
        wav_storage_user, wav_storage_details = extract_waveforms(bvp_ts.sel(compound="bvp_ts"))
    """  # noqa: D205, E501

    # --- Determine sampling rate in Hz ---
    fs_qty = sampling_rate(bvp_ts)
    fs = float(fs_qty.to('Hz').magnitude)

    # --- Parameters for diastolic minima detection ---
    min_peak_dist = int(fs / 2)   # minimum distance between minima
    min_peak_height = 0           # minimum height of minima

    wav_storage_user = {}
    wav_storage_details = {}

    # --- Process each channel independently ---
    for ch in bvp_ts.channel.values:
        actual_ts = bvp_ts.sel(channel=ch)
        actual_ts_np = actual_ts.to_numpy().squeeze()

        # --- Detect diastolic minima (invert signal to find minima as peaks) ---
        minima_idx, minima_value = peakseek(
            -1 * actual_ts_np,
            min_peak_dist,
            min_peak_height)
        minima_value = -1 * minima_value

        # --- BVP analysis specific correction of peakseek ---
        # Compare the duration of the pulse with its neighbors
        # and filter out if it is too short
        i = 0
        while i < len(minima_idx) - 1:
            start = max(0, i-3)
            end = min(len(minima_idx), i+5)

            diff = np.diff(minima_idx[start:end])
            if len(diff) == 0:
                i += 1
                continue

            diff_median = np.median(diff)

            if i == 0 and minima_idx[i+1] - minima_idx[i] < diff_median * 0.6:
                help_var = minima_value[i:i+2]
                del_idx = np.argmax(help_var)
                del_pos = i + del_idx

                minima_idx = np.delete(minima_idx, del_pos)
                minima_value = np.delete(minima_value, del_pos)

                continue

            if minima_idx[i+1] - minima_idx[i] < diff_median * 0.6:
                help_var = minima_value[i-1:i+2]
                del_idx = np.argmax(help_var)
                del_pos = i - 1 + del_idx

                minima_idx = np.delete(minima_idx, del_pos)
                minima_value = np.delete(minima_value, del_pos)

                if del_idx == 1:
                    i = i-1

                continue

            i = i + 1

        # Compare the distance between the median minimum value and zero
        # with the distance between the actual minimum value and zero, and
        # filter the value out if the acutal distance is too small.
        minimma_value_median = np.median(minima_value)
        dist_median_zero = np.abs(0 - minimma_value_median)

        del_idx_list = []
        n = len(minima_idx)

        for i in range(n):
            actual_dist_zero = abs(0 - minima_value[i])
            if actual_dist_zero < dist_median_zero * 0.1:
                del_idx_list.append(i)

        minima_idx = np.delete(minima_idx, del_idx_list)
        minima_value = np.delete(minima_value, del_idx_list)

        # --- Initialize lists ---
        bvp_waveforms = []
        bvp_max_value = []
        bvp_max_idx = []
        bvp_wav_xy_normal_all = []

        # --- Iterate over consecutive minima to extract waveforms ---
        for i in range(len(minima_idx) - 1):

            # --- Duration check using local median ---
            start = max(0, i - 3)
            end = min(len(minima_idx), i + 5)
            diff = np.diff(minima_idx[start:end])

            if len(diff) == 0:
                continue

            diff_median = np.median(diff)

            # Reject abnormally long segments (non-physiological)
            if minima_idx[i + 1] - minima_idx[i] > diff_median * 1.5:
                continue

            # --- Extract waveform segment ---
            wav = actual_ts_np[minima_idx[i]: minima_idx[i + 1]]

            # --- Detect systolic maximum within waveform ---
            local_max_idx = np.argmax(wav)
            bvp_max_value.append(wav[local_max_idx])
            bvp_max_idx.append(local_max_idx + minima_idx[i])

            # --- Time vector for raw waveform ---
            wav_len = minima_idx[i + 1] - minima_idx[i]
            time_wav_s = np.linspace(0, wav_len / fs, wav_len)

            # --- Normalization #1: remove linear trend ---
            trend_x = np.array([0, len(wav) - 1])
            trend_y = np.array([wav[0], wav[-1]])
            pchip_trend = PchipInterpolator(trend_x, trend_y)
            trend = pchip_trend(np.arange(len(wav)))
            wav_y_normal = wav - trend

            # --- Normalization #2: resample waveform to fixed length (100) ---
            wav_xy_normal = interpft(wav_y_normal, 100)

            # --- Collect waveform-level data ---
            bvp_waveforms.append({
                "wav_raw": wav,
                "wav_y_normal": wav_y_normal,
                "wav_time_s": time_wav_s
            })

            bvp_wav_xy_normal_all.append(wav_xy_normal)

        # --- Convert lists to NumPy arrays ---
        bvp_max_value = np.array(bvp_max_value)
        bvp_max_idx = np.array(bvp_max_idx)
        bvp_wav_xy_normal_all = np.array(bvp_wav_xy_normal_all).T

        # --- Normalization #3: z-score waveform ---
        bvp_wav_xy_normal_zscore_all = zscore(bvp_wav_xy_normal_all)

        for i in range(bvp_wav_xy_normal_zscore_all.shape[1]):
            help_wav = bvp_wav_xy_normal_zscore_all[:,i]

            trend_x = np.array([0, len(help_wav) - 1])
            trend_y = np.array([help_wav[0], help_wav[-1]])
            pchip_trend = PchipInterpolator(trend_x, trend_y)
            trend = pchip_trend(np.arange(len(help_wav)))
            help_wav_detrended = help_wav - trend

            bvp_wav_xy_normal_zscore_all[:,i] = help_wav_detrended

        help_mean = np.mean(bvp_wav_xy_normal_zscore_all, axis=1)
        help_max = help_mean.max()

        bvp_wav_xy_normal_zscore_all = (bvp_wav_xy_normal_zscore_all / help_max)

        # --- Store channel-wise results ---
        wav_storage_user[ch] = {
            "bvp_max_value": bvp_max_value,
            "bvp_max_idx": bvp_max_idx,
            "bvp_min_value": minima_value,
            "bvp_min_idx": minima_idx}
        wav_storage_details[ch] ={
            "list_wav_raw_and_y_normal": bvp_waveforms,
            "nparray_wav_xy_normal_all": bvp_wav_xy_normal_all,
            "nparray_wav_xy_normal_zscore_all": bvp_wav_xy_normal_zscore_all}

    return wav_storage_user, wav_storage_details

def remove_artifact_waveforms(
    bvp_ts: cdt.NDTimeSeries,
    wav_storage_user: dict,
    wav_storage_details: dict
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]]
    ]:
    """Removes artifactual BVP waveforms based on deviation from the
    mean normalized waveform.

    For each channel, a mean waveform is computed from the z-scored,
    xy-normalized waveforms. For every individual waveform, a deviation
    metric is calculated as the summed absolute distance to this mean
    waveform. This deviation serves as an artifact score.
    Waveforms with deviation values above the 97.5th percentile are
    classified as artifacts and removed.
    The cleaned waveform matrices are stored alongside the original data
    without overwriting them.

    Args:
        bvp_ts: BVP time series used only to iterate over available channels.
        wav_storage_user: Channel-wise dictionary created by
            extract_waveforms, containing user-facing waveform data.
        wav_storage_details: Channel-wise dictionary created by
            extract_waveforms, containing detailed waveform matrices.

    Returns:
        Tuple of two dictionaries (updated in-place and returned for
        convenience):

        1. wav_storage_user:
            Extended with artifact-cleaned matrices of x/y-normalized and
            z-scored waveforms:
                - nparray_wav_xy_normal_zscore_all_woa

        2. wav_storage_details:
            Extended with artifact-cleaned matrices and diagnostic information:
                - nparray_wav_xy_normal_all_woa
                - P_975: upper deviation percentile threshold
                - bvp_wav_dev: deviation score per waveform

    Example:
        wav_storage_user, wav_storage_details = remove_artifact_waveforms(
            bvp_ts,
            wav_storage_user,
            wav_storage_details
        )
    """  # noqa: D205

    for ch in bvp_ts.channel.values:
        actual_wavs_final = wav_storage_details[ch]["nparray_wav_xy_normal_zscore_all"]
        actual_wavs_xynorm = wav_storage_details[ch]["nparray_wav_xy_normal_all"]

        # Number of waveforms
        n_bvp_wav = actual_wavs_final.shape[1] - 1

        # Mean waveform (over all waveforms)
        bvp_wav_final_mean = np.mean(actual_wavs_final, axis=1)

        # Compute deviation metric per waveform (absolute deviation)
        bvp_wav_dev = np.zeros(n_bvp_wav)
        for i in range(n_bvp_wav):
            bvp_wav_dev[i] = np.sum(np.abs(actual_wavs_final[:, i]
                                           - bvp_wav_final_mean))

        # Percentile threshold
        p_975 = np.percentile(bvp_wav_dev, 97.5)

        # Artifact indices
        idx_bvp_wav_p975 = np.where(bvp_wav_dev > p_975)[0]

        # Remove artifacts and store output
        wav_storage_user[ch]["nparray_wav_xy_normal_zscore_all_woa"] = np.delete(
            actual_wavs_final, idx_bvp_wav_p975, axis=1)
        wav_storage_user[ch]["nparray_wav_xy_normal_all_woa"] = np.delete(
            actual_wavs_xynorm, idx_bvp_wav_p975, axis=1)
        wav_storage_details[ch]["P_975"] = p_975
        wav_storage_details[ch]["bvp_wav_dev"] = bvp_wav_dev

    return wav_storage_user, wav_storage_details

def classify_waveforms(
        bvp_cont: BVP_Container,
        classification_metric: str) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]]
    ]:
    """Classifies artifact-cleaned BVP waveforms based on their
    systolic peak amplitudes.

    For each channel, the classification metric is computed (see section Args).
    Waveforms are then classified into three amplitude-based groups using
    percentile thresholds:
        - type 1:  < 25th percentile
        - type 2: 25th to 75th percentile
        - type 3:  > 75th percentile

    Args:
        bvp_cont: the BVP Container which includes the two storages built by
        the function "extract_waveforms", and edited by the function
        "remove_artifact_waveforms" and "classify_waveforms". To create
        this storage a blood volume pulse time series created by the function
        "extract_bvp" is necessary.
        classification_metric: string that defines which metric is used
        for classification. Use 'max' for using the highest maximum of each
        xy-normalized and z-scored waveform, and 'delta' for using the vertical
        distance between the highest maximum and the lowest local minimum
        of each xy-normalized waveform.

    Returns:
        Tuple of two dictionaries (updated in-place and returned):
        Updates for classification with delta are written in brackets.
        1. wav_storage_user:
            Extended with waveform classes:
                - nparray_wav_max_type1 (nparray_wav_delta_type1)
                - nparray_wav_max_type2 (nparray_wav_delta_type2)
                - nparray_wav_max_type3 (nparray_wav_delta_type3)
        2. wav_storage_details:
            Extended with classification diagnostics:
                - max_bvp_wav (delta_bvp_wav): per-waveform peak amplitudes
                - max_P_25 (delta_P_25): 25th percentile threshold
                - max_P_75 (delta_P75): 75th percentile threshold
                - (text_num_del_wavs: Text for printing. Tells the user how many
                    waveforms are discarded and the total amount of waveforms
                    for each channel.)

    Example:
        bvp_cont.wav_storage_user, bvp_cont.wav_storage_details = classify_waveforms(
        bvp_cont, 'delta')
    """  # noqa: D205, E501

    bvp_ts= bvp_cont['bvp_ts']
    wav_storage_user = bvp_cont.wav_storage_user
    wav_storage_details= bvp_cont.wav_storage_details

    if classification_metric == 'max':
    # --- Iterate over all channels ---
        for ch in bvp_ts.channel.values:

            # --- Artifact-cleaned, xy_normalized and z-scored waveforms ---
            actual_wavs = (
                wav_storage_user[ch]["nparray_wav_xy_normal_zscore_all_woa"]
            )

            # --- Compute highest maximum for each waveform ---
            max_bvp_wav = np.max(actual_wavs, axis=0)

            # --- Percentile-based classification thresholds ---
            p25 = np.percentile(max_bvp_wav, 25)
            p75 = np.percentile(max_bvp_wav, 75)

            # --- Boolean masks for waveform classes ---
            idx_type1 = max_bvp_wav < p25
            idx_type2 = max_bvp_wav > p75
            idx_type3 = (max_bvp_wav > p25) & (max_bvp_wav < p75)

            # --- Store classified waveform groups ---
            wav_storage_user[ch]["nparray_wav_max_type1"] = (
                actual_wavs[:, idx_type1])
            wav_storage_user[ch]["nparray_wav_max_type2"] = (
                actual_wavs[:, idx_type2])
            wav_storage_user[ch]["nparray_wav_max_type3"] = (
                actual_wavs[:, idx_type3])

            # --- Store classification metrics ---
            wav_storage_details[ch]["max_bvp_wav"] = max_bvp_wav
            wav_storage_details[ch]["max_P_25"] = p25
            wav_storage_details[ch]["max_P_75"] = p75

    if classification_metric == 'delta':
    # --- Iterate over all channels ---
        for ch in bvp_ts.channel.values:

            # --- Artifact-cleaned, xy_normalized waveforms ---
            actual_wavs = (
                wav_storage_user[ch]["nparray_wav_xy_normal_all_woa"]
            )

            # --- Compute vertical distance between the highest maximum
            # and the lowest local minimum for each waveform ---
            max_bvp_wav = np.max(actual_wavs, axis=0)
            min_bvp_wav = max_bvp_wav.copy()

            for i in range(actual_wavs.shape[1]):
                all_min_locs, all_min_values = peakseek(-1 * actual_wavs[:,i])
                all_min_values = -1 * all_min_values

                if len(all_min_locs) > 1:
                    help_min_locs = all_min_locs[(all_min_locs > 20) &
                                                 (all_min_locs < 60)]
                    help_min_values = actual_wavs[:,i][help_min_locs]

                    if len(help_min_values) == 0:
                        continue

                    min_bvp_wav[i] = min(help_min_values)

                if len(all_min_locs) == 0:
                    continue

                if len(all_min_locs) == 1:
                    if float(all_min_locs) > 20 and float(all_min_locs) < 60:
                        min_bvp_wav[i] = float(all_min_values)
                    if float(all_min_locs) <= 20 or float(all_min_locs) >= 60:
                        continue

            delta_bvp_wav = max_bvp_wav - min_bvp_wav
            del_idx = np.where(delta_bvp_wav == 0)

            text_num_del_wavs = f'{ch}:  {len(del_idx[0])}  of  {actual_wavs.shape[1]}'

            delta_bvp_wav = np.delete(delta_bvp_wav, del_idx)
            actual_wavs = np.delete(actual_wavs, del_idx, 1)

            # --- Percentile-based classification thresholds ---
            p25 = np.percentile(delta_bvp_wav, 25)
            p75 = np.percentile(delta_bvp_wav, 75)

            # --- Boolean masks for waveform classes ---
            idx_typ1 = delta_bvp_wav < p25
            idx_typ2 = delta_bvp_wav > p75
            idx_typ3 = (delta_bvp_wav > p25) & (delta_bvp_wav < p75)

            # --- Store classified waveform groups ---
            wav_storage_user[ch]["nparray_wav_delta_type1"] = (
                actual_wavs[:, idx_typ1])
            wav_storage_user[ch]["nparray_wav_delta_type2"] = (
                actual_wavs[:, idx_typ2])
            wav_storage_user[ch]["nparray_wav_delta_type3"] = (
                actual_wavs[:, idx_typ3])

            # --- Store classification metrics ---
            wav_storage_details[ch]["delta_bvp_wav"] = delta_bvp_wav
            wav_storage_details[ch]["delta_P_25"] = p25
            wav_storage_details[ch]["delta_P_75"] = p75
            wav_storage_details[ch]["text_num_del_wavs"] = text_num_del_wavs

    return wav_storage_user, wav_storage_details

def extract_bvpa(
    bvp_ts: cdt.NDTimeSeries,
    wav_storage_user: dict
) -> cdt.NDTimeSeries:
    """Extracts the blood volume pulse amplitude (BVPA) time series from
    a blood volume pulse (BVP) signal.

    For each channel, upper and lower envelopes of the BVP signal are
    estimated using systolic maxima and diastolic minima obtained from
    waveform analysis. The raw BVPA is computed as the difference between
    these envelopes. Additionally, a smoothed BVPA signal is generated
    using LOESS smoothing.

    The resulting BVPA-related time series are stored as separate
    compounds in a single NDTimeSeries.

    Args:
        bvp_ts: BVP time series created by `extract_bvp` (compound="bvp_ts").
        wav_storage_user: Channel-wise dictionary created by
            `extract_waveforms` (subsequent processing by
            `remove_artifact_waveforms` or `classify_waveforms` is
            possible).

    Returns:
        NDTimeSeries with dimensions (channel x compound x time) containing:
            - env_up: upper BVP envelope (interpolated systolic maxima)
            - env_down: lower BVP envelope (interpolated diastolic minima)
            - bvpa_raw: raw blood volume pulse amplitude
            - bvpa_smooth: LOESS-smoothed BVPA signal

    Example:
        bvpa_ts = extract_bvpa(
            bvp_ts.sel(compound="bvp_ts"),
            wav_storage_user
        )
    """  # noqa: D205

    fs_qty = sampling_rate(bvp_ts)
    fs = float(fs_qty.to('Hz').magnitude)

    ch_list = bvp_ts.channel.values
    time = bvp_ts.time.values
    place_holder = np.zeros((len(ch_list), 4, len(time)))

    bvpa_ts = cdc.schemas.build_timeseries(
                place_holder,
                ["channel", "compound", "time"],
                time,
                ch_list,
                'uM',
                's',
                {"compound": ("compound", ["env_up", "env_down", "bvpa_raw", "bvpa_smooth"]),  # noqa: E501
                "samples": ("time", np.arange(0, len(time))),
                "source": ("channel", bvp_ts.source.values),
                "detector": ("channel", bvp_ts.detector.values)})

    for ch in ch_list:
        actual_bvp_ts = bvp_ts.sel(channel=ch)
        actual_bvp_ts_np = actual_bvp_ts.to_numpy().squeeze()
        actual_wav_storage = wav_storage_user[ch]

        # --- upper Envelope ---
        x_up = np.concatenate(([0], actual_wav_storage["bvp_max_idx"]))
        y_up = np.concatenate(([actual_wav_storage["bvp_max_value"][0]],
                               actual_wav_storage["bvp_max_value"]))
        pchip_up = PchipInterpolator(x_up, y_up)
        env_up = pchip_up(np.arange(len(actual_bvp_ts_np)))

        # --- lower Envelope ---
        x_down = np.concatenate(([0], actual_wav_storage["bvp_min_idx"]))
        y_down = np.concatenate(([actual_wav_storage["bvp_min_value"][0]],
                                 actual_wav_storage["bvp_min_value"]))
        pchip_down = PchipInterpolator(x_down, y_down)
        env_down = pchip_down(np.arange(len(actual_bvp_ts_np)))

        # --- BVPA = upper Envelope - lower Envelope ---
        bvpa_raw = env_up - env_down

        # --- LOESS smoothing (1 sec window = 50 samples) ---
        model = loess(np.arange(len(bvpa_raw)), bvpa_raw,
                    span=fs / len(bvpa_raw), degree=2, family='gaussian')
        model.fit()
        bvpa_smooth = model.outputs.fitted_values

        bvpa_ts.loc[{"channel": ch, "compound": "env_up"}
                    ] = env_up * physunits.units.uM
        bvpa_ts.loc[{"channel": ch, "compound": "env_down"}
                    ] = env_down * physunits.units.uM
        bvpa_ts.loc[{"channel": ch, "compound": "bvpa_raw"}
                    ] = bvpa_raw * physunits.units.uM
        bvpa_ts.loc[{"channel": ch, "compound": "bvpa_smooth"}
                    ] = bvpa_smooth * physunits.units.uM

    return bvpa_ts

def extract_pulse_rate(
    bvp_ts: cdt.NDTimeSeries,
    wav_storage_user: dict
) -> cdt.NDTimeSeries:
    """Extracts the pulse rate (PR) time series from a blood volume pulse (BVP)
    signal using diastolic minima.

    For each channel, the temporal distances between consecutive diastolic
    minima are computed and converted to pulse rate values (beats per minute).
    Non-physiological intervals are corrected using a local median-based
    heuristic. The pulse rate is then interpolated to a continuous time series
    and additionally smoothed using LOESS.
    The resulting pulse rate signals are stored as separate compounds in a
    single NDTimeSeries.

    Args:
        bvp_ts: BVP time series created by `extract_bvp` (compound="bvp_ts").
        wav_storage_user: Channel-wise dictionary created by
            `extract_waveforms` (subsequent processing by
            `remove_artifact_waveforms` or `classify_waveforms` is
            possible).

    Returns:
        NDTimeSeries with dimensions (channel x compound x time) containing:
            - pulse_rate: interpolated pulse rate time series (beats per minute)
            - pulse_rate_smooth: LOESS-smoothed pulse rate time series

    Example:
        pulse_rate_ts = extract_pulse_rate(
            bvp_ts.sel(compound="bvp_ts"),
            wav_storage_user
        )
    """  # noqa: D205

    fs_qty = sampling_rate(bvp_ts)
    fs = float(fs_qty.to('Hz').magnitude)

    ch_list = bvp_ts.channel.values
    time = bvp_ts.time.values
    place_holder = np.zeros((len(ch_list), 2, len(time)))

    pulse_rate_ts = cdc.schemas.build_timeseries(
                place_holder,
                ["channel", "compound", "time"],
                time,
                ch_list,
                'min**-1',
                's',
                {"compound": ("compound", ["pulse_rate", "pulse_rate_smooth"]),
                "samples": ("time", np.arange(0, len(time))),
                "source": ("channel", bvp_ts.source.values),
                "detector": ("channel", bvp_ts.detector.values)})

    for ch in ch_list:
        actual_wav_storage = wav_storage_user[ch]
        minima_idx = actual_wav_storage["bvp_min_idx"]

        # --- PR_raw: Differences between maxima ---
        pulse_rate_dist = np.diff(minima_idx)

        # --- filters out non physiological pulses (if diff between minima is too
        #     large --> value replaced by mean of neighbors) ---
        n = len(pulse_rate_dist)

        for i in range(n - 1):
            start = max(0, i-3)
            end = min(n, i+5)

            window = pulse_rate_dist[start:end]
            if len(window) == 0:
                continue

            window_median = np.median(window)

            if pulse_rate_dist[i] > (window_median * 1.6) and i == 0:
                pulse_rate_dist[i] = pulse_rate_dist[i+1]
                continue

            if pulse_rate_dist[i] > (window_median * 1.6) and i == n:
                pulse_rate_dist[i] = pulse_rate_dist[i-1]
                continue

            if pulse_rate_dist[i] > (window_median * 1.6):
                pulse_rate_dist[i] = np.mean([pulse_rate_dist[i-1], pulse_rate_dist[i+1]])  # noqa: E501

        # --- Interpolated PR time series ---
        pchip_x = minima_idx[1:]
        pchip_x_new = np.arange(minima_idx[1], minima_idx[-1] + 1)

        pchip = PchipInterpolator(pchip_x, pulse_rate_dist)
        pulse_rate_intp_help = pchip(pchip_x_new)
        pulse_rate_intp = 1.0 / (pulse_rate_intp_help / fs / 60)

        # --- Append initial and final sequences ---
        pulse_rate_start_seq = np.ones(minima_idx[1] - 1) * pulse_rate_intp[0]
        pulse_rate_end_seq = np.ones(len(time) - minima_idx[-1]) * pulse_rate_intp[-1]

        pulse_rate_intp_full = np.concatenate([pulse_rate_start_seq,
                                               pulse_rate_intp,
                                               pulse_rate_end_seq])

        # --- Loess smoothing (5 second window) ---
        N = 5 * fs
        span = N / len(pulse_rate_intp_full)

        model = loess(np.arange(len(pulse_rate_intp_full)), pulse_rate_intp_full,
                    span=span, degree=2, family='gaussian')
        model.fit()
        pulse_rate_smooth = model.outputs.fitted_values

        pulse_rate_ts.loc[{"channel": ch, "compound": "pulse_rate"}
                          ] = pulse_rate_intp_full * physunits.units.min**-1
        pulse_rate_ts.loc[{"channel": ch, "compound": "pulse_rate_smooth"}
                          ] = pulse_rate_smooth * physunits.units.min**-1

    return pulse_rate_ts

def filter_pulse_rate(pulse_rate_ts: cdt.NDTimeSeries,
                      fmin: float,
                      fmax: float,
                      butter_order: int = 4) -> cdt.NDTimeSeries:
    """Filters the pulse rate time series created by the function "extract_pulse_rate"
    by using a Butterworth bandpass filter (Cedalion function: freq_filter).
    The time series "pulse_rate_filt" is added to the input NDTimeSeries.

    Args:
        pulse_rate_ts: pulse rate time series created by the function
        'extract_pulse_rate'.
        fmin: lower frequency border
        fmax: upper freqency border
        butter_order: order of Butterworth filter

    Syntax:
        pulse_rate_ts = filter_pulse_rate(
            pulse_rate_ts,
            fmin=0.1, fmax=0.45,
            butter_order=2)
    """  # noqa: D205

    fmin = fmin * physunits.units.Hz
    fmax = fmax * physunits.units.Hz

    pulse_rate_smooth_ts = pulse_rate_ts.sel(compound="pulse_rate_smooth")

    pulse_rate_filt_ts_help = freq_filter(pulse_rate_smooth_ts,
                                     fmin,
                                     fmax,
                                     butter_order)
    pulse_rate_filt_ts = pulse_rate_filt_ts_help.expand_dims(compound=['pulse_rate_filt'])  # noqa: E501
    pulse_rate_ts_exp = xr.concat([pulse_rate_ts, pulse_rate_filt_ts], dim='compound')

    return pulse_rate_ts_exp

def calc_wav_coh_bvpa_pr(ts_1: cdt.NDTimeSeries,
                           ts_2: cdt.NDTimeSeries,
                           wav_storage_details: dict,
                           dj=1/12, s0=None, J=None,
                           do_zscore=True, do_detrend=True, do_sig=False,
                           significance_level=0.95
                           ):
    """Calculates the wavelet coherence between BVPA and PR time series.

       Mother-function = Morlet.

    Args:
        ts_1: first time series.
        ts_2: second time series (must be of same length as ts_1).
        wav_storage_details: Channel-wise dictionary.
        dj: Spacing between discrete scales. Default value is 1/12.
            Smaller values will result in better scale resolution, but
            slower calculation.
        s0: Smallest scale of the wavelet. Default value is 2*dt.
        J: Number of scales less one. Scales range from s0 up to
           s0 * 2**(J * dj), which gives a total of (J + 1) scales.
           Default is J = (log2(N*dt/so))/dj.
        do_zscore: set True for zscoring time series.
        do_detrend: set True for detrending time series.
        do_sig: set True for calculationg significance of coherence.
        significance_level: Significance level to use. Default is 0.95.

    Returns:
        Extensions of output_details:
            - wavelet_coherence
            - phase
            - cone_of_interest
            - frequency
            - significance

    Example:
        bvp_cont.wav_storage_details = calc_wavelet_coherence(
            bvp_cont['bvpa_ts'].sel(compound='bvpa_smooth'),
            bvp_cont['pulse_rate_ts'].sel(compound='pulse_rate_smooth'),
            bvp_cont.wav_storage_details,
            s0=0.5, J=100
)
    """

    fs_qty = sampling_rate(ts_1)
    fs = float(fs_qty.to('Hz').magnitude)
    dt = 1.0 / fs

    # ch_list = ts_1.channel.values
    ch_list = ['S4D10']

    for ch in ch_list:
        y_ts_1 = ts_1.sel(channel=ch).to_numpy()
        y_ts_2 = ts_2.sel(channel=ch).to_numpy()

        y_ts_1 = np.asarray(y_ts_1, dtype=float)
        y_ts_2 = np.asarray(y_ts_2, dtype=float)
        if y_ts_1.shape != y_ts_2.shape or y_ts_1.ndim != 1:
            raise ValueError("y_ts_1 and y_ts_2 must be 1D-arrays of same length.")

        n = y_ts_1.size
        time = np.arange(n) * dt

        # ----- remove trend -----
        if do_detrend:
            y_ts_1 = detrend(y_ts_1, type='linear')
            y_ts_2 = detrend(y_ts_2, type='linear')

        if s0 is None:
            s0 = 2 * dt
        if J is None:
            J = int(np.log2(n * dt / s0) / dj)

        # ----- calculate wavelet coherence -----
        S12, S1, S2, WCT, aWCT, coi, freq, significance = wct(
            y_ts_1, y_ts_2, dt, dj=dj, s0=s0, J=J,
            normalize=do_zscore,
            sig=do_sig,
            significance_level=significance_level
        )

        wav_storage_details[ch]["wavelet_coherence"] = WCT
        wav_storage_details[ch]["phase"] = aWCT
        wav_storage_details[ch]["cone_of_interest"] = coi
        wav_storage_details[ch]["frequency"] = freq
        wav_storage_details[ch]["significance"] = significance
        wav_storage_details[ch]["wc_time"] = time
        wav_storage_details[ch]["cross_wavelet_transform"] = S12
        wav_storage_details[ch]["cwt_signal1"] = S1
        wav_storage_details[ch]["cwt_signal2"] = S2

    return wav_storage_details


# --- BVP Plots -------------------

def plot_concts_bvpts(bvp_cont: BVP_Container, ch: str, container_name: str = "") -> None:
    """Creates a 2 x 1 subplot:
        - Upper: upsampled HbO conc time series and its low frequency trend
        - Lower: blood volume pulse time series and the respective systolic
          maxima and diastoloc minima.

    Args:
        bvp_cont: the BVP Container which includes the blood volume pulse time series
        created by the function "extract_bvp" and the two storages built by
        the function "extract_waveforms".
        ch: string that specifies the channel which sould be plotted.
        container_name: name of BVP Container.

    Example:
        plot_concts_bvpts(bvp_cont, "S1D15")
    """  # noqa: D205

    time_min = bvp_cont['bvp_ts'].time
    source = bvp_cont['bvp_ts'].coords["source"].sel(channel=ch).item()
    detector = bvp_cont['bvp_ts'].coords["detector"].sel(channel=ch).item()

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    plt.rcParams.update({'font.size': 10})
    fig.patch.set_facecolor('white')

    ax = axes[0]
    ax.plot(time_min, bvp_cont['bvp_ts'].sel(compound='hbo_conc_ts_50hz', channel=ch),
            color=[0.259, 0.478, 0.729], linewidth=0.5, label='[O₂Hb] raw data')
    ax.plot(time_min, bvp_cont['bvp_ts'].sel(compound='low_freq_trend', channel=ch),
            linewidth=1, color=[1, 0, 0], label='[O₂Hb] low—frequency trend')
    ax.set_title('[O₂Hb] ('+source+' | '+detector+', $f_s$ = 50 Hz)',
                  fontweight='bold', fontsize=14)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('[O₂Hb] [µM]')
    ax.autoscale(enable=True, tight=True)
    ax.legend(facecolor="white", framealpha=1)

    ax = axes[1]
    ax.plot(time_min, bvp_cont['bvp_ts'].sel(channel=ch, compound='bvp_ts'),
            color=[0.259, 0.478, 0.729], linewidth=0.5)
    ax.plot(time_min[bvp_cont.wav_storage_user[ch]['bvp_min_idx']],
            bvp_cont.wav_storage_user[ch]['bvp_min_value'],
            '+', color='r', markersize=8, linewidth=2, label='Minima')
    ax.plot(time_min[bvp_cont.wav_storage_user[ch]['bvp_max_idx']],
            bvp_cont.wav_storage_user[ch]['bvp_max_value'],
            '+', color='k', markersize=8, linewidth=2, label='Maxima')
    ax.set_title('Blood volume pulse (BVP) ('+source+' | '+detector+')',
                  fontweight='bold', fontsize=14)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('BVP [µM]')
    ax.autoscale(enable=True, tight=True)
    ax.legend(facecolor="white", framealpha=1)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)

    if not container_name == '':
        fig.tight_layout(rect=[0, 0, 1, 0.965])
        fig.suptitle(container_name, fontsize=14)
    else:
        plt.tight_layout()

    plt.show()

def plot_wavs_4x(bvp_cont: BVP_Container, ch: str) -> None:
    """Creates a 2 x 2 subplot:
        - Upper left: BVP waveforms (not normalized).
        - Upper right: BVP waveforms (detrended).
        - Lower left: BVP waveforms (detrended + equal length).
        - Lower right: BVP waveforms (detrended + equal length + z-score).

    Args:
        bvp_cont: the BVP Container which includes the two storages built by
        the function "extract_waveforms". To create this storage a blood volume pulse
        time series created by the function "extract_bvp" is necessary.
        ch: string that specifies the channel which sould be plotted.

    Example:
        plot_wavs_4x(bvp_cont, "S1D15")
    """  # noqa: D205

    subplot_0_0 = bvp_cont.wav_storage_details[ch]['list_wav_raw_and_y_normal']
    subplot_0_1 = subplot_0_0
    subplot_1_0 = bvp_cont.wav_storage_details[ch]['nparray_wav_xy_normal_all']
    subplot_1_1 = bvp_cont.wav_storage_details[ch]['nparray_wav_xy_normal_zscore_all']


    fig, axes = plt.subplots(2, 2, figsize=(11,8))
    plt.rcParams.update({'font.size': 10})
    fig.patch.set_facecolor('white')

    color_bvp = [0.259, 0.478, 0.729]

    # --- Subplot (0,0) : BVP waveforms (not normalized) ---
    ax = axes[0, 0]
    for i in range(len(subplot_0_0)):
        ax.plot(subplot_0_0[i]['wav_time_s'], subplot_0_0[i]['wav_raw'],
                color=color_bvp, linewidth=0.5, alpha=0.2)
    ax.set_title('BVP waveforms (raw data)',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('[O₂Hb] [µM]')
    ax.autoscale(enable=True, tight=True)

    # --- Subplot (0,1) : BVP waveforms (detrended) ---
    ax = axes[0, 1]
    for i in range(len(subplot_0_1)):
        ax.plot(subplot_0_1[i]['wav_time_s'], subplot_0_1[i]['wav_y_normal'],
                color=color_bvp, linewidth=0.5, alpha=0.2)
    ax.set_title('BVP waveforms (detrended)',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('[O₂Hb] [AU]')
    ax.autoscale(enable=True, tight=True)

    # --- Subplot (1,0) : BVP waveforms (detrended + equal length) ---
    ax = axes[1, 0]
    ax.plot(subplot_1_0, color=color_bvp, linewidth=0.5, alpha=0.2)
    ax.plot(np.mean(subplot_1_0, axis=1), color='r', linewidth=2)
    ax.set_title('BVP waveforms (detrended, normalized: time)',
                fontweight='bold', fontsize=10)
    ax.set_xlabel('Time [samples]')
    ax.set_ylabel('[O₂Hb] [AU]')
    ax.autoscale(enable=True, tight=True)

    # --- Subplot (1,1) : BVP waveforms (detrended + equal length + z-score) ---
    ax = axes[1, 1]
    ax.plot(subplot_1_1, color=color_bvp, linewidth=0.5, alpha=0.2)
    ax.plot(np.mean(subplot_1_1, axis=1), color='r', linewidth=2)
    ax.set_title('BVP waveforms (detrended, normalized: time, amplitude)',
                fontweight='bold', fontsize=10)
    ax.set_xlabel('Time [samples]')
    ax.set_ylabel('[O₂Hb] [AU]')
    ax.autoscale(enable=True, tight=True)
    ax.axhline(1, color='k', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_wavs_woa(bvp_cont: BVP_Container, ch: str, container_name: str = "") -> None:
    """Creates a 2 x 2 subplot:
        - Upper left: Deviation metric curve.
        - Upper right: Deviation metric histogram.
        - Lower left: BVP waveforms (detrended + equal length).
        - Lower right: BVP waveforms (detrended + equal length + z-score).

    Args:
        bvp_cont: the BVP Container which includes the two storages built by
        the function "extract_waveforms" and edited by the function
        "remove_artifact_waveforms". To create this storage a blood volume pulse
        time series created by the function "extract_bvp" is necessary.
        ch: string that specifies the channel which sould be plotted.
        container_name: name of BVP Container.

    Example:
        plot_wavs_woa(bvp_cont, "S1D15")
    """  # noqa: D205

    subplot_0_0 = bvp_cont.wav_storage_details[ch]['bvp_wav_dev']
    subplot_0_1 = subplot_0_0
    subplot_1_0 = bvp_cont.wav_storage_user[ch]['nparray_wav_xy_normal_all_woa']
    subplot_1_1 = bvp_cont.wav_storage_user[ch]['nparray_wav_xy_normal_zscore_all_woa']  # noqa: E501

    fig, axes = plt.subplots(2, 2, figsize=(11,8))
    plt.rcParams.update({'font.size': 10})
    fig.patch.set_facecolor('white')

    color_bvp = [0.259, 0.478, 0.729]

    # --- Subplot (0,0) : Waveform deviation index (WDI) curve ---
    ax = axes[0, 0]
    ax.plot(subplot_0_0, color=color_bvp, linewidth=0.8)
    ax.set_title('Waveform deviation index (WDI)', fontweight='bold', fontsize=12)
    ax.set_xlabel('BVP waveform [n]')
    ax.set_ylabel('WDI')
    ax.autoscale(enable=True, tight=True)
    # percentile line
    ax.axhline(bvp_cont.wav_storage_details[ch]['P_975'],
               color='r', linestyle='--', linewidth=1)

    # --- Subplot (1,1) : Waveform deviation index (WDI) histogram ---
    ax = axes[0, 1]
    ax.hist(subplot_0_1, bins=100, color=color_bvp)
    ax.set_title('WDI distribution', fontweight='bold', fontsize=12)
    ax.set_xlabel('WDI')
    ax.set_ylabel('Count')
    ax.autoscale(enable=True, tight=True)
    # percentile line
    ax.axvline(bvp_cont.wav_storage_details[ch]['P_975'],
               color='r', linestyle='--', linewidth=1)

    # --- Subplot (1,0) : BVP waveforms woa (detrended + equal length) ---
    ax = axes[1, 0]
    ax.plot(subplot_1_0, color=color_bvp, linewidth=0.5, alpha=0.1)
    ax.plot(np.mean(subplot_1_0, axis=1), color='r', linewidth=2)
    ax.set_title('BVP waveforms (detrended, normalized: time)',
                fontweight='bold', fontsize=10)
    ax.set_xlabel('Time [samples]')
    ax.set_ylabel('[O₂Hb] [µM]')
    ax.autoscale(enable=True, tight=True)

    # --- Subplot (1,1) : BVP waveforms woa (detrended + equal length + z-score) ---
    ax = axes[1, 1]
    ax.plot(subplot_1_1, color=color_bvp, linewidth=0.5, alpha=0.1)
    ax.plot(np.mean(subplot_1_1, axis=1), color='r', linewidth=2)
    ax.set_title('BVP waveforms (detrended, normalized: time, amplitude)',
                fontweight='bold', fontsize=10)
    ax.set_xlabel('Time [samples]')
    ax.set_ylabel('[O₂Hb] [AU]')
    ax.autoscale(enable=True, tight=True)
    ax.axhline(1, color='k', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)

    if not container_name == '':
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.suptitle(container_name, fontsize=12, fontweight='bold')
    else:
        plt.tight_layout()
    plt.show()

def plot_wavs_classes(bvp_cont: BVP_Container,
                      ch: str,
                      classification_index: str) -> None:
    """Creates a 2 x 2 subplot:
        - Upper left: Distribution of Max.
        - Upper right: Waveforms Class S.
        - Lower left: Waveforms Class M.
        - Lower right: Waveforms Class L.
    The Waveforms are detrended, equal lenth, zscored and without artefacts.

    Args:
        bvp_cont: the BVP Container which includes the two storages built by
        the function "extract_waveforms", and edited by the function
        "remove_artifact_waveforms" and "classify_waveforms". To create
        this storage a blood volume pulse time series created by the function
        "extract_bvp" is necessary.
        ch: string that specifies the channel which sould be plotted.
        classification_index: string that defines which classes should be plotted.
        Use 'max' for the highest maximum of each xy-normalized and z-scored waveform,
        and 'delta' for the vertical distance between the highest maximum and the
        lowest local minimum of each xy-normalized waveform.

    Example:
        plot_wavs_classes(bvp_cont, "S1D15", 'delta')
    """  # noqa: D205

    if classification_index == 'max':
        subplot_0_0 = bvp_cont.wav_storage_details[ch]['max_bvp_wav']
        subplot_0_1 = bvp_cont.wav_storage_user[ch]['nparray_wav_max_type1']
        subplot_1_0 = bvp_cont.wav_storage_user[ch]['nparray_wav_max_type2']
        subplot_1_1 = bvp_cont.wav_storage_user[ch]['nparray_wav_max_type3']
        title = 'Distribution of Max'
        p25 = 'max_P_25'
        p75 = 'max_P_75'

    if classification_index == 'delta':
        subplot_0_0 = bvp_cont.wav_storage_details[ch]['delta_bvp_wav']
        subplot_0_1 = bvp_cont.wav_storage_user[ch]['nparray_wav_delta_type1']
        subplot_1_0 = bvp_cont.wav_storage_user[ch]['nparray_wav_delta_type2']
        subplot_1_1 = bvp_cont.wav_storage_user[ch]['nparray_wav_delta_type3']
        title = 'Distribution of Delta'
        p25 = 'delta_P_25'
        p75 = 'delta_P_75'

    fig, axes = plt.subplots(2, 2, figsize=(11,8))
    plt.rcParams.update({'font.size': 10})
    fig.patch.set_facecolor('white')

    color_bvp = [0.259, 0.478, 0.729]

    # --- (1) Histogram Distribution of Max ---
    ax = axes[0, 0]
    ax.hist(subplot_0_0, bins=100)
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('Max')
    ax.set_ylabel('Number')
    ax.autoscale(enable=True, tight=True)
    ax.axvline(bvp_cont.wav_storage_details[ch][p25], color='k', linestyle='--')
    ax.axvline(bvp_cont.wav_storage_details[ch][p75], color='k', linestyle='--')

    # --- (2) Class S ---
    ax = axes[0, 1]
    ax.plot(subplot_0_1, color=color_bvp, linewidth=0.5, alpha=0.1)
    ax.plot(np.mean(subplot_0_1, axis=1), 'r', linewidth=2)
    ax.set_title('BVP waveform: shape type 1', fontweight='bold', fontsize=12)
    ax.set_xlabel('Time [samples]')
    ax.set_ylabel('[O₂Hb] [AU]')
    ax.autoscale(enable=True, tight=True)
    if classification_index == 'max':
        ax.axhline(1, color='k', linestyle='--', linewidth=0.5)

    # --- (3) Class M ---
    ax = axes[1, 0]
    ax.plot(subplot_1_0, color=color_bvp, linewidth=0.5, alpha=0.1)
    ax.plot(np.mean(subplot_1_0, axis=1), 'r', linewidth=2)
    ax.set_title('BVP waveform: shape type 2', fontweight='bold', fontsize=12)
    ax.set_xlabel('Time [samples]')
    ax.set_ylabel('[O₂Hb] [AU]')
    ax.autoscale(enable=True, tight=True)
    if classification_index == 'max':
        ax.axhline(1, color='k', linestyle='--', linewidth=0.5)

    # --- (4) Class L ---
    ax = axes[1, 1]
    ax.plot(subplot_1_1, color=color_bvp, linewidth=0.5, alpha=0.1)
    ax.plot(np.mean(subplot_1_1, axis=1), 'r', linewidth=2)
    ax.set_title('BVP waveform: shape type 3', fontweight='bold', fontsize=12)
    ax.set_xlabel('Time [samples]')
    ax.set_ylabel('[O₂Hb] [AU]')
    ax.autoscale(enable=True, tight=True)
    if classification_index == 'max':
        ax.axhline(1, color='k', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def plot_bvpts_bvpats(bvp_cont: BVP_Container, ch: str) -> None:
    """Creates a 2 x 1 subplot:
        - Upper: blood volume pulse time series and the upper and lower envelopes
        - Lower: blood volume pulse amplitude time series (raw and smoothed).

    Args:
        bvp_cont: the BVP Container which includes the blood volume pulse time series
        created by the function "extract_bvp", the two storages built by
        the function "extract_waveforms" and the blood volume pulse amplitude time
        series creatd by the function "extract_bvpa".
        ch: string that specifies the channel which sould be plotted.

    Example:
        plot_bvpts_bvpats(bvp_cont, "S1D15")
    """  # noqa: D205

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7), constrained_layout=True)
    plt.rcParams.update({'font.size': 10})
    fig.patch.set_facecolor('white')

    source = bvp_cont['bvp_ts'].coords["source"].sel(channel=ch).item()
    detector = bvp_cont['bvp_ts'].coords["detector"].sel(channel=ch).item()

    # Subplot: bvp_ts + env_up + env_down
    ax = axes[0]
    ax.plot(bvp_cont['bvp_ts'].time/60,
            bvp_cont['bvp_ts'].sel(channel=ch, compound="bvp_ts"),
            color=[0.259, 0.478, 0.729], linewidth=0.5)
    ax.plot(bvp_cont['bvpa_ts'].time/60,
            bvp_cont['bvpa_ts'].sel(channel=ch, compound="env_up"),
            color='r', linewidth=0.5, label='Upper envelope')
    ax.plot(bvp_cont['bvpa_ts'].time/60,
            bvp_cont['bvpa_ts'].sel(channel=ch, compound="env_down"),
            color='tab:orange', linewidth=0.5, label='Lower envelope')
    ax.autoscale(enable=True, tight=True)
    ax.set_title('Blood volume pulse (BVP) ('+source+' | '+detector+')')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('BVP [µM]')
    ax.legend(facecolor="white", framealpha=1)

    # Subplot: bvpa_raw + bvpa_smooth
    ax = axes[1]
    ax.plot(bvp_cont['bvpa_ts'].time/60,
            bvp_cont['bvpa_ts'].sel(channel=ch, compound="bvpa_raw"),
            color=[0.259, 0.478, 0.729], linewidth=0.5, label='Raw')
    ax.plot(bvp_cont['bvpa_ts'].time/60,
            bvp_cont['bvpa_ts'].sel(channel=ch, compound="bvpa_smooth"),
            color='r', linewidth=0.5, label='Smooth')
    ax.autoscale(enable=True, tight=True)
    ax.set_title('Blood volume pulse amplitude (BVPA) ('+source+' | '+detector+')')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('BVPA [µM]')
    ax.legend(facecolor="white", framealpha=1)

    plt.show()

def plot_pulse_rate(bvp_cont: BVP_Container, ch: str) -> None:
    """Creates a plot including the pulse rate and its shmoothed version.

    Args:
        bvp_cont: the BVP Container which includes the blood volume pulse time series
        created by the function "extract_bvp", the two storages built by
        the function "extract_waveforms" and the pulse rate time
        series creatd by the function "extract_pulse_rate".
        ch: string that specifies the channel which sould be plotted.

    Example:
        plot_pulse_rate(bvp_cont, "S1D15")
    """  # noqa: D205

    fig, ax = plt.subplots(figsize=(13.5, 7), constrained_layout=True)
    plt.rcParams.update({'font.size': 10})
    fig.patch.set_facecolor('white')

    source = bvp_cont['bvp_ts'].coords["source"].sel(channel=ch).item()
    detector = bvp_cont['bvp_ts'].coords["detector"].sel(channel=ch).item()

    ax.plot(bvp_cont['pulse_rate_ts'].time/60,
            bvp_cont['pulse_rate_ts'].sel(channel=ch, compound="pulse_rate"),
            color=[0.259, 0.478, 0.729], linewidth=0.5, label="Raw")
    ax.plot(bvp_cont['pulse_rate_ts'].time/60,
            bvp_cont['pulse_rate_ts'].sel(channel=ch, compound="pulse_rate_smooth"),  # noqa: E501
            color='r', linewidth=0.5, label="Smooth")
    ax.autoscale(enable=True, tight=True)
    ax.set_title('Pulse rate (PR) ('+source+' | '+detector+')')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('PR [1/min]')
    ax.legend(facecolor="white", framealpha=1)

    plt.show()

def plot_bvpa_pr_comparison(bvp_cont: BVP_Container, ch: str) -> None:
    """Creates a 2 x 1 subplot:
        - Upper: BVPA + Pulse Rate (seperate y axes).
        - Lower: Pulse Rate filtered.

    Args:
        bvp_cont: the BVP Container which includes the blood volume pulse time series
        created by the function "extract_bvp", the two storages built by
        the function "extract_waveforms", the blood volume pulse amplitude time
        series creatd by the function "extract_bvpa", and the pulse rate time
        series creatd by the function "extract_pulse_rate" and edited by the function
        "filter_pulse_rate".
        ch: string that specifies the channel which sould be plotted.

    Example:
        plot_bvpa_pr_comparison(bvp_cont, "S1D15")
    """  # noqa: D205

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7), constrained_layout=True)
    plt.rcParams.update({'font.size': 10})
    fig.patch.set_facecolor('white')

    color_bvpa = [0.259, 0.478, 0.729]
    color_pr  = [0.959, 0.278, 0.329]

    pulse_rate_filtered = (
        bvp_cont['pulse_rate_ts'].sel(channel=ch, compound="pulse_rate_smooth") -
        bvp_cont['pulse_rate_ts'].sel(channel=ch, compound="pulse_rate_filt"))

    ax = axes[0]
    ax.plot(
        bvp_cont['bvpa_ts'].time / 60,
        bvp_cont['bvpa_ts'].sel(channel=ch, compound="bvpa_smooth"),
        color=color_bvpa,
        linewidth=0.5,
        label='BVPA'
    )
    ax.set_ylabel('BVPA [AU]')
    ax.set_xlabel('Time [min]')
    ax.set_zorder(1)

    ax_pr = ax.twinx()
    ax_pr.plot(
        bvp_cont['pulse_rate_ts'].time / 60,
        bvp_cont['pulse_rate_ts'].sel(channel=ch, compound="pulse_rate_smooth"),
        color=color_pr,
        linewidth=0.5,
        label='Pulse Rate'
    )
    ax_pr.set_ylabel('Pulse Rate [1/min]')
    ax_pr.set_zorder(0)

    ax.set_title('BVPA and PR')
    lines_bvpa, labels_bvpa = ax.get_legend_handles_labels()
    lines_pr, labels_pr = ax_pr.get_legend_handles_labels()
    legend = ax.legend(lines_bvpa + lines_pr, labels_bvpa + labels_pr,
                       facecolor="white", framealpha=1)
    legend.set_zorder(10)
    ax.autoscale(enable=True, tight=True)
    ax.patch.set_visible(False)

    ax = axes[1]
    ax.plot(bvp_cont['pulse_rate_ts'].time/60,
            pulse_rate_filtered,
            color=color_pr, linewidth=0.5)
    ax.autoscale(enable=True, tight=True)
    ax.set_title('PR filtered')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Pulse Rate [1/min]')

    plt.show()

def plot_concts_bvpats_pr(bvp_cont: BVP_Container, ch: str) -> None:
    """Creates a 3 x 1 subplot:
        - Upper: upsampled HbO conc time series and its low frequency trend
        - Middle: blood volume pulse amplitude time series smoothed.
        - Lower: pulse rate and its shmoothed version.

    Args:
        bvp_cont: the BVP Container which includes the blood volume pulse time series
            created by the function "extract_bvp", the two storages built by
            the function "extract_waveforms", the blood volume pulse amplitude time
            series creatd by the function "extract_bvpa" and the pulse rate time
            series creatd by the function "extract_pulse_rate".
        ch: string that specifies the channel which sould be plotted.

    Example:
        plot_concts_bvpats_pr(bvp_cont, "S1D15")
    """  # noqa: D205

    time_min = bvp_cont['bvp_ts'].time/60

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), constrained_layout=True)
    plt.rcParams.update({'font.size': 10})
    fig.patch.set_facecolor('white')

    source = bvp_cont['bvp_ts'].coords["source"].sel(channel=ch).item()
    detector = bvp_cont['bvp_ts'].coords["detector"].sel(channel=ch).item()

    ax = axes[0]
    ax.plot(time_min, bvp_cont['bvp_ts'].sel(compound='hbo_conc_ts_50hz', channel=ch),
            color=[0.259, 0.478, 0.729], linewidth=0.5, label='HbO concentration')
    ax.plot(time_min, bvp_cont['bvp_ts'].sel(compound='low_freq_trend', channel=ch),
            linewidth=1, color=[1, 0, 0], label='low frequency trend')
    ax.set_title('[O₂Hb] ('+source+' | '+detector+')',
                  fontweight='bold', fontsize=14)
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('[O₂Hb] [µM]')
    ax.autoscale(enable=True, tight=True)
    ax.legend(facecolor="white", framealpha=1)

    ax = axes[1]
    ax.plot(time_min, bvp_cont['bvpa_ts'].sel(channel=ch, compound="bvpa_smooth"), color='r', linewidth=0.5)  # noqa: E501
    ax.autoscale(enable=True, tight=True)
    ax.set_title('BVPA ('+source+' | '+detector+')')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('BVPA [µM]')

    ax = axes[2]
    ax.plot(time_min, bvp_cont['pulse_rate_ts'].sel(channel=ch, compound="pulse_rate"),  # noqa: E501
            color=[0.259, 0.478, 0.729], linewidth=0.5, label="pulse rate")
    ax.plot(time_min, bvp_cont['pulse_rate_ts'].sel(channel=ch, compound="pulse_rate_smooth"),  # noqa: E501
            color='r', linewidth=0.5, label="pulse rate smooth")
    ax.autoscale(enable=True, tight=True)
    ax.set_title('PR ('+source+' | '+detector+')')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Pulse Rate [1/min]')
    ax.legend(facecolor="white", framealpha=1)

def plot_coherence_bvpa_pr(bvp_cont: BVP_Container, ch: str,
                           coherence_thresh=0.9,
                           arrow_step_time=30,
                           arrow_step_period: int=4) -> None:
    """Plots time series and wavelet coherence between BVPA and PR.

    Args:
        bvp_cont: BVP Container which includes the blood volume pulse time series
            created by the function "extract_bvp" and the two storages.
        ch: string that specifies the channel which sould be plotted.
        coherence_thresh: threshold above which the phase-arrows should be plotted.
        arrow_step_time: steps between lines of arrow-grid in the direction of time
            in seconds.
        arrow_step_period: steps between lines of arrow-grid in the direction of
            frequency. Higher values lead to lower arrow density.

    Example:
        plot_coherence_bvpa_pr(bvp_cont, "S1D15")
    """

    WCT = bvp_cont.wav_storage_details[ch]["wavelet_coherence"]
    aWCT = bvp_cont.wav_storage_details[ch]["phase"]
    coi = bvp_cont.wav_storage_details[ch]["cone_of_interest"]
    freq = bvp_cont.wav_storage_details[ch]["frequency"]
    significance = bvp_cont.wav_storage_details[ch]["significance"]
    time = bvp_cont.wav_storage_details[ch]["wc_time"] / 60
    n = time.size

    fs_qty = sampling_rate(bvp_cont['bvpa_ts'])
    fs = float(fs_qty.to('Hz').magnitude)
    arrow_step_time = int(arrow_step_time * fs)

    source = bvp_cont['bvp_ts'].coords["source"].sel(channel=ch).item()
    detector = bvp_cont['bvp_ts'].coords["detector"].sel(channel=ch).item()

    # ----- PLOT -----
    fig = plt.figure(figsize=(13.5, 7))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[80, 1],
                  height_ratios=[1, 1], wspace=0.01, hspace=0.27)
    plt.rcParams.update({'font.size': 10})
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.08)

    color_bvpa = [0.259, 0.478, 0.729]
    color_pr  = [0.959, 0.278, 0.329]

    # ----- Comparison BVPA and PR
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.plot(bvp_cont['bvpa_ts'].time / 60,
                bvp_cont['bvpa_ts'].sel(channel=ch, compound="bvpa_smooth"),
                color=color_bvpa, linewidth=0.5, label='BVPA')
    ax_top.set_ylabel('BVPA [AU]')
    ax_top.set_xlabel('Time [min]')
    ax_top.set_zorder(1)

    ax_top_pr = ax_top.twinx()
    ax_top_pr.plot(bvp_cont['pulse_rate_ts'].time / 60,
                   bvp_cont['pulse_rate_ts'].sel(channel=ch, compound="pulse_rate_smooth"),  # noqa: E501
                   color=color_pr, linewidth=0.5, label='PR')
    ax_top_pr.set_ylabel('PR [1/min]')
    ax_top_pr.set_zorder(0)
    ax_top_pr.autoscale(enable=True, tight=True)

    ax_top.set_title("Blood volume pulse amplitude and pulse rate ("
                     +source+" | "+detector+")", fontweight='bold')
    lines_bvpa, labels_bvpa = ax_top.get_legend_handles_labels()
    lines_pr, labels_pr = ax_top_pr.get_legend_handles_labels()
    legend = ax_top.legend(lines_bvpa + lines_pr, labels_bvpa + labels_pr,
                       facecolor="white", framealpha=1,#
                       loc="upper right")
    legend.set_zorder(10)

    ax_top.xaxis.set_major_locator(MaxNLocator(nbins=14, prune=None))
    ax_top.patch.set_visible(False)

    # ----- Coherence-Scalogram
    ax_bottom = fig.add_subplot(gs[1, 0])

    levels = np.linspace(0, 1, 50)
    cmap = cmap_parula()
    cf = ax_bottom.contourf(time, np.log2(freq), WCT, levels=levels,
                                 cmap=cmap, vmin=0, vmax=1)

    ax_bottom.set_title("Wavelet coherence (BVPA–PR coupling)", fontweight='bold')
    ax_bottom.set_xlabel("Time [min]")
    ax_bottom.xaxis.set_major_locator(MaxNLocator(nbins=14, prune=None))
    yticks = np.array([0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 2])
    ax_bottom.set_yticks(np.log2(yticks))
    ax_bottom.set_yticklabels([f"{v:g}" for v in yticks])
    ax_bottom.set_ylim(np.log2(freq.min()), np.log2(2))
    ax_bottom.set_ylabel("Frequency [Hz]")

    # Cone of Influence (COI)
    coi = np.asarray(coi)
    coi_freq = 1.0 / coi
    ax_bottom.fill_between(
        time,
        np.log2(freq.min()) * np.ones_like(time),
        np.log2(coi_freq),
        facecolor="grey",
        alpha=0.6,
        edgecolor="k",
        linewidth=0.2,)

    # Significance
    if not significance == [0]:
        sig_arr = np.asarray(significance)

        if sig_arr.ndim == 1 and sig_arr.size == WCT.shape[0]:
            sig2d = sig_arr[:, None] * np.ones_like(WCT)
            ax_bottom.contour(time, np.log2(freq), WCT - sig2d,
                                   levels=[0], linewidths=1.0)

        elif sig_arr.ndim == 2 and sig_arr.shape == WCT.shape:
            ax_bottom.contour(time, np.log2(freq), WCT - sig_arr,
                                   levels=[0], linewidths=1.0)

    # Phase-Arrows
    TT, FF = np.meshgrid(time, freq)

    inside_coi = FF >= coi_freq[np.newaxis, :]
    strong_coh = WCT >= coherence_thresh
    mask = inside_coi & strong_coh

    ti = np.arange(0, n, arrow_step_time)
    si = np.arange(0, freq.size, arrow_step_period)
    T_sub = TT[np.ix_(si, ti)]
    P_sub = FF[np.ix_(si, ti)]
    phi_sub = aWCT[np.ix_(si, ti)]
    mask_sub = mask[np.ix_(si, ti)]

    U = np.cos(phi_sub)
    V = np.sin(phi_sub)

    ax_bottom.quiver(T_sub[mask_sub], np.log2(P_sub[mask_sub]),
                        U[mask_sub], V[mask_sub],
                        angles="uv", pivot="mid",
                        scale_units="width", scale=100,
                        width=0.001, headwidth=6, headlength=7)

    # ----- Color bar
    ax_cbar = fig.add_subplot(gs[1, 1])


    cbar = fig.colorbar(cf, cax=ax_cbar)
    cbar.set_ticks(np.arange(0, 1.01, 0.2))
    cbar.set_label('Coherence', rotation=90, labelpad=7)

    plt.show()
