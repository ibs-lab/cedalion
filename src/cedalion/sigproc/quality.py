"""Signal quality metrics and channel pruning functionality."""

from __future__ import annotations
import logging
from functools import reduce

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from scipy import signal
from scipy.stats import gaussian_kde
from scipy.stats import median_abs_deviation

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
import cedalion.sigproc.frequency as freq
from cedalion import Quantity, units
from cedalion.typing import NDTimeSeries
import cedalion.nirs as nirs
from .frequency import freq_filter, sampling_rate

logger = logging.getLogger("cedalion")

CLEAN = True
TAINTED = False

@cdc.validate_schemas
def prune_ch(
    amplitudes: cdt.NDTimeSeries,
    masks: list[cdt.NDTimeSeries],
    operator: str,
    flag_drop: bool = True,
):
    """Prune channels from the the input data array using quality masks.

    Args:
        amplitudes (:class:`NDTimeSeries`): input time series
        masks (:class:`list[NDTimeSeries]`) : list of boolean masks with coordinates
            comptabile to amplitudes

        operator: operators for combination of masks before pruning data_array

            - "all": logical AND, keeps channel if it is good across all masks
            - "any": logical OR, keeps channel if it is good in any mask/metric

        flag_drop: if True, channels are dropped from the data_array, otherwise they are
            set to NaN (default: True)

    Returns:
        A tuple (amplitudes_pruned, prune_list), where amplitudes_pruned is
        a the original time series channels pruned (dropped) according to quality masks.
        A list of removed channels is returned in prune_list.
    """

    # check if all dimensions in the all the masks are also existing in data_array
    for mask in masks:
        if not all(dim in amplitudes.dims for dim in mask.dims):
            raise ValueError(
                "mask dimensions must be a subset of data_array dimensions"
            )

    # combine quality masks according to operator instruction
    if operator.lower() == "all":
        # sets True where all masks (metrics) are true
        # Combine the DataArrays using a boolean "AND" operation across elements
        mask = reduce(lambda x, y: x & y, masks)

    elif operator.lower() == "any":
        # sets True where any mask (metric) is true
        # Combine the DataArrays using a boolean "OR" operation across elements
        mask = reduce(lambda x, y: x | y, masks)
    else:
        raise ValueError(f"unsupported operator '{operator}'")

    # apply mask to drop channels
    if flag_drop:
        amplitudes, prune_list = xrutils.apply_mask(
            amplitudes, mask, "drop", dim_collapse="channel"
        )
    else:
        amplitudes, prune_list = xrutils.apply_mask(
            amplitudes, mask, "nan", dim_collapse="channel"
        )

    return amplitudes, prune_list


@cdc.validate_schemas
def psp(
    amplitudes: NDTimeSeries,
    window_length: cdt.QTime,
    psp_thresh: float,
    cardiac_fmin: cdt.QFrequency = 0.5 * units.Hz,
    cardiac_fmax: cdt.QFrequency = 2.5 * units.Hz,
):
    """Calculate the peak spectral power.

    The peak spectral power metric is based on :cite:t:`Pollonini2014` /
    :cite:t:`Pollonini2016`.

    Args:
        amplitudes (:class:`NDTimeSeries`, (channel, wavelength, time)): input time
            series
        window_length (:class:`Quantity`, [time]): size of the computation window
        psp_thresh: if the calculated PSP metric falls below this threshold then the
            corresponding time window should be excluded.
        cardiac_fmin : minimm frequency to extract cardiac component
        cardiac_fmax : maximum frequency to extract cardiac component

    Returns:
        A tuple (psp, psp_mask), where psp is a DataArray with coords from the input
        NDTimeseries containing the peak spectral power. psp_mask is a boolean mask
        DataArray with coords from psp, true where psp_thresh is met.
    """

    amp = _extract_cardiac(amplitudes, cardiac_fmin, cardiac_fmax)

    amp = amp.pint.dequantify()

    amp = (amp - amp.mean("time")) / amp.std("time")

    # convert window_length to samples
    nsamples = (window_length * sampling_rate(amp)).to_base_units()
    nsamples = int(np.ceil(nsamples))

    # This creates a new DataArray with a new dimension "window", that is
    # window_len_samples large. The time dimension will contain the time coordinate of
    # the first sample in the window. Setting the stride size to the same value as the
    # window length will result in non-overlapping windows.
    windows = amp.rolling(time=nsamples).construct("window", stride=nsamples)
    windows = windows.fillna(1e-6)

    # Vectorized signal extraction and correlation
    sig = windows.transpose("channel", "time", "wavelength", "window").values
    nchannel = windows.sizes["channel"]
    ntime = windows.sizes["time"]  # after rolling this is the number of windows

    lags = np.arange(-nsamples + 1, nsamples)
    nlags = len(lags)
    norm_unbiased = nsamples - np.abs(lags)  # shape (nlags,)

    hamming_window = np.hamming(nlags)
    hamming_window_norm = np.sum(hamming_window) ** 2

    # nsample / (sigma(wl1)*sigma(wl2)) , shape(nchannel, ntime)
    corr_coeff_denom = nsamples / np.sqrt(np.sum(np.power(sig, 2), axis=-1)).prod(-1)

    corr = np.zeros((ntime, nchannel, nlags))
    for w in range(ntime):  # loop over windows
        for ch in range(nchannel):
            corr[w, ch, :] = signal.correlate(
                sig[ch, w, 0, :], sig[ch, w, 1, :], "full"
            )

    corr *= corr_coeff_denom.T[:, :, None]
    norm_unbiased = nsamples - np.abs(lags)  # shape (nlags,)
    corr /= norm_unbiased[None, None, :]
    corr *= hamming_window[None, None, :]

    fft_out = np.fft.rfft(corr, axis=-1)  # shape (ntime, nchannel, nfreqs)
    power = (np.abs(fft_out) ** 2) / hamming_window_norm

    psp = np.max(power, axis=2).T  # shape(nchannel, ntime)

    # keep dims channel and time
    psp_xr = windows.isel(wavelength=0, window=0).drop_vars("wavelength").copy(data=psp)

    # Apply threshold mask
    psp_mask = xrutils.mask(psp_xr, CLEAN)
    psp_mask = psp_mask.where(psp_xr > psp_thresh, other=TAINTED)

    return psp_xr, psp_mask


@cdc.validate_schemas
def gvtd(amplitudes: NDTimeSeries, stat_type: str = "default", n_std: int = 10):
    """Calculate GVTD metric based on :cite:t:`Sherafati2020`.

    Args:
        amplitudes (:class:`NDTimeSeries`, (channel, wavelength, time)): input time
            series

        stat_type (string): statistic of GVTD time trace to use to set the threshold
            (see _get_gvtd_threshold). Default = 'default'

        n_std (int): number of standard deviations for consider above the statistic of
            interest.

    Returns:
        A DataArray with coords from the input NDTimeseries containing the GVTD metric.
    """

    fcut_min = 0.01
    fcut_max = 0.5

    od = nirs.int2od(amplitudes)
    od = xr.where(np.isinf(od), 0, od)
    od = xr.where(np.isnan(od), 0, od)
    od.time.attrs["units"] = units.s
    od_filtered = od.cd.freq_filter(fcut_min, fcut_max, 4)

    od_filtered = od_filtered.pint.dequantify() # OD is dimensionless

    # Step 1: Find the matrix of the temporal derivatives
    dataDiff = od_filtered - od_filtered.shift(time=-1)

    # Step 2: Find the RMS across the channels for each time-point of dataDiff
    GVTD = np.sqrt((dataDiff[:, :-1] ** 2).mean(dim="channel"))

    # Step 3: Add a zero in the beginning for GVTD to have the same number of
    # time-points as your original dataMatrix
    GVTD = GVTD.squeeze()
    GVTD.values = np.hstack([0, GVTD.values[:-1]])
    GVTD = GVTD.drop_vars("wavelength")

    # Step 4: Scale to have units of OD/s
    GVTD *= freq.sampling_rate(amplitudes)

    # Apply threshold mask
    thresh = _get_gvtd_threshold(GVTD, stat_type=stat_type, n_std=n_std)

    GVTD_mask = xrutils.mask(GVTD, CLEAN)
    GVTD_mask = GVTD_mask.where(GVTD < thresh, other=TAINTED)

    return GVTD, GVTD_mask


def _get_gvtd_threshold(
    GVTD: NDTimeSeries,
    stat_type: str = "default",
    n_std: int = 10,
):
    """Calculate GVTD threshold based on :cite:t:`Sherafati2020`.

    Args:
        GVTD (:class:`NDTimeSeries`, (time,)): GVTD timetrace

        stat_type (string): statistic of GVTD time trace to use to set the threshold

            - *default*: threshold is the mode plus the distance between the smallest
              GVTD value and the mode.
            - *histogram_mode*: threshold is the mode plus the standard deviation of the
              points below the mode * n_std.
            - *kdensity_mode*: use kdensity estimation to find the mode the gvtd
              distribution. threshold is this mode pluts the standard deviation of
              points below the mode * n_std.
            - *parabolic_mode*: use parabolic interpolation to estimate the mode.
              threshold is this mode pluts the standard deviation of points below the
              mode*n_std.
            - *mean*: same as histogram_mode but using the mean instead of the mode.
            - *median*: same as histogram_mode but using the median instead of the mode.
            - *MAD*: same as histogram_mode but using the MAD instead of the mode.

        n_std (int): number of standard deviations for consider above the statistic of
            interest.

    Returns:
        thresh (float): the threshold above which GVTD is considered motion.
    """

    units = GVTD.pint.units
    GVTD = GVTD.pint.dequantify()

    if stat_type == "default":
        min_counts_per_bin = 5

        n_bins = int(np.round(GVTD.shape[0] / min_counts_per_bin))

        bin_size = GVTD.max() / n_bins

        N, edges = np.histogram(
            GVTD.values, bins=n_bins, range=(0, np.max(GVTD.values))
        )

        argmax = np.argmax(N)

        run_mode = edges[argmax] + bin_size / 2

        # To find the motion threshold, the distance between the smallest GVTD value and
        # the mode is calculated left tail is used, as it is dominated by the
        # physiological signal and not motion artifacts.
        min_val_to_mode_dist = run_mode - np.min(GVTD)

        # Motion threshold is defined as mode plus a multiplier of the standard
        # deviation of the data-points below the mode
        thresh = run_mode + min_val_to_mode_dist

    elif stat_type == "histogram_mode":
        min_counts_per_bin = 5

        n_bins = int(np.round(GVTD.shape[0] / min_counts_per_bin))

        bin_size = GVTD.max() / n_bins

        N, edges = np.histogram(
            GVTD.values, bins=n_bins, range=(0, np.max(GVTD.values))
        )

        argmax = np.argmax(N)

        run_mode = edges[argmax] + bin_size / 2

        # Find time points below the mode
        points_below_mode = GVTD[GVTD < run_mode]

        # Number of points below the mode
        num_points_below_mode = points_below_mode.size

        # RMS of points below the mode
        rms_points_below_mode = np.sum((points_below_mode - run_mode) ** 2)

        # Standard deviation of points below the mode
        left_std_run = np.sqrt(rms_points_below_mode / num_points_below_mode)

        # Motion threshold is defined as mode plus a multiplier of the standard
        # deviation of the data-points below the mode
        thresh = run_mode + n_std * left_std_run

    elif stat_type == "kdensity_mode":
        # consider only gvtd values that are finite and positive
        mask = np.isfinite(GVTD) & (GVTD > 0)
        gvtd_log = np.log(GVTD[mask])

        # Kernel density estimate for the log of gvtdTimeTrace
        kde = gaussian_kde(gvtd_log)
        xi = np.linspace(np.min(gvtd_log), np.max(gvtd_log), 1000)
        f = kde(xi)

        # Find the mode (the point with maximum density)
        idx = np.argmax(f)
        run_mode = np.exp(xi[idx])

        # Find time points below the mode
        points_below_mode = GVTD[GVTD < run_mode]

        # Number of points below the mode
        num_points_below_mode = points_below_mode.size

        # RMS of points below the mode
        rms_points_below_mode = np.sum((points_below_mode - run_mode) ** 2)

        # Standard deviation of points below the mode
        left_std_run = np.sqrt(rms_points_below_mode / num_points_below_mode)

        # Motion threshold is defined as mode plus a multiplier of the standard
        # deviation of the data-points below the mode
        thresh = run_mode + n_std * left_std_run

    elif stat_type == "parabolic_mode":
        min_counts_per_bin = 5

        n_bins = int(np.round(GVTD.shape[0] / min_counts_per_bin))

        bin_size = GVTD.max() / n_bins

        N, edges = np.histogram(
            GVTD.values, bins=n_bins, range=(0, np.max(GVTD.values))
        )

        argmax = np.argmax(N)

        centers = edges + bin_size.values / 2

        centers = centers[:-1]

        # Identify the adjacent points around the mode
        x1, y1 = centers[argmax - 1], N[argmax - 1]
        x2, y2 = centers[argmax], N[argmax]
        x3, y3 = centers[argmax + 1], N[argmax + 1]

        # Calculate the quadratic approximation for the mode
        num = (x2**2 - x1**2) * (y2 - y3) - (x2**2 - x3**2) * (y2 - y1)
        denom = (x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1)
        run_mode = 0.5 * (num / denom)

        # Find time points below the mode
        points_below_mode = GVTD[GVTD < run_mode]

        # Number of points below the mode
        num_points_below_mode = points_below_mode.size

        # RMS of points below the mode
        rms_points_below_mode = np.sum((points_below_mode - run_mode) ** 2)

        # Standard deviation of points below the mode
        left_std_run = np.sqrt(rms_points_below_mode / num_points_below_mode)

        # Motion threshold is defined as mode plus a multiplier of the standard
        # deviation of the data-points below the mode
        thresh = run_mode + n_std * left_std_run

    elif stat_type == "median":
        # Assuming gvtdTimeTrace is a numpy array
        run_median = np.median(GVTD)

        # Find time points below the median
        points_below_median = GVTD[GVTD < run_median]

        # Number of points below the median
        num_points_below_median = points_below_median.size

        # RMS of points below the median
        rms_points_below_median = np.sum((points_below_median - run_median) ** 2)

        # Standard deviation of points below the median
        left_std_run = np.sqrt(rms_points_below_median / num_points_below_median)

        # Motion threshold is defined as the median plus a multiplier of the standard
        # deviation of the data-points below the median
        thresh = run_median + n_std * left_std_run

    elif stat_type == "mean":
        # Assuming gvtdTimeTrace is a numpy array
        run_mean = np.mean(GVTD)

        # Find time points below the mean
        points_below_mean = GVTD[GVTD < run_mean]

        # Number of points below the mean
        num_points_below_mean = points_below_mean.size

        # RMS of points below the mean
        rms_points_below_mean = np.sum((points_below_mean - run_mean) ** 2)

        # Standard deviation of points below the mean
        left_std_run = np.sqrt(rms_points_below_mean / num_points_below_mean)

        # Motion threshold is defined as the mean plus a multiplier of the standard
        # deviation of the data-points below the mean
        thresh = run_mean + n_std * left_std_run

    elif stat_type == "MAD":
        # Calculate the MAD (median absolute deviation) with a scaling factor of 1
        run_mad = median_abs_deviation(GVTD, scale=1)

        # Motion threshold is defined as a multiplier of the MAD
        thresh = n_std * run_mad

    else:
        raise ValueError(f"Unknown stat '{stat_type}'")

    return thresh * units


@cdc.validate_schemas
def sci(
    amplitudes: NDTimeSeries,
    window_length: Quantity,
    sci_thresh: float,
    cardiac_fmin: cdt.QFrequency = 0.5 * units.Hz,
    cardiac_fmax: cdt.QFrequency = 2.5 * units.Hz,
):
    """Calculate the scalp-coupling index.

    The scalp-coupling index metric is based on :cite:t:`Pollonini2014` /
    :cite:t:`Pollonini2016`.

    Args:
        amplitudes (:class:`NDTimeSeries`, (channel, wavelength, time)): input time
            series
        window_length (:class:`Quantity`, [time]): size of the computation window
        sci_thresh: if the calculated SCI metric falls below this threshold then the
            corresponding time window should be excluded.
        cardiac_fmin : minimm frequency to extract cardiac component
        cardiac_fmax : maximum frequency to extract cardiac component

    Returns:
        A tuple (sci, sci_mask), where sci is a DataArray with coords from the input
        NDTimeseries containing the scalp-coupling index. Sci_mask is a boolean mask
        DataArray with coords from sci, true where sci_thresh is met.
    """

    assert "wavelength" in amplitudes.dims  # FIXME move to validate schema

    amp = _extract_cardiac(amplitudes, cardiac_fmin, cardiac_fmax)

    amp = (amp - amp.mean("time")) / amp.std("time")

    # convert window_length to samples
    nsamples = (window_length * sampling_rate(amp)).to_base_units()
    nsamples = int(np.ceil(nsamples))

    # This creates a new DataArray with a new dimension "window", that is
    # window_len_samples large. The time dimension will contain the time coordinate of
    # the first sample in the window. Setting the stride size to the same value as the
    # window length will result in non-overlapping windows.
    windows = amp.rolling(time=nsamples).construct("window", stride=nsamples)

    sci = (windows - windows.mean("window")).prod("wavelength").sum("window") / nsamples
    sci /= windows.std("window").prod("wavelength") # dims: channel, time

    # create sci mask and update accoording to sci_thresh
    sci_mask = xrutils.mask(sci, CLEAN)
    sci_mask = sci_mask.where(sci > sci_thresh, TAINTED)

    return sci, sci_mask

def _extract_cardiac(
    amplitudes: NDTimeSeries,
    cardiac_fmin: cdt.QFrequency,
    cardiac_fmax: cdt.QFrequency,
):
    """Apply a bandpass or highpass filter to extract the cardiac component."""

    fs = sampling_rate(amplitudes)
    fny =  fs / 2
    if fny < cardiac_fmin:
        raise ValueError("sampling rate is not sufficient to extract cardiac component")
    elif fny > cardiac_fmax:
        return freq_filter(amplitudes, cardiac_fmin, cardiac_fmax, butter_order=4)
    else:  # fny in [cardiac_fmin, cardiac_fmax] -> highpass
        return freq_filter(
            amplitudes, fmin=cardiac_fmin, fmax=0 * units.Hz, butter_order=4
        )


@cdc.validate_schemas
def snr(amplitudes: cdt.NDTimeSeries, snr_thresh: float = 2.0):
    """Calculates signal-to-noise ratio for each channel and other dimension.

    SNR is the ratio of the average signal over time divided by its standard deviation.

    Args:
        amplitudes (:class:`NDTimeSeries`, (time, *)): the input time series
        snr_thresh:  threshold (unitless) below which a channel should be excluded.

    Returns:
        A tuple (snr, snr_mask) , where snr is a DataArray with coords from input
        NDTimeseries containing the ratio between mean and std of the amplitude signal
        for all channels. snr_mask is a boolean mask DataArray with the same coords
        as snr, true where snr_threshold is met.

    References:
        Based on Homer3 v1.80.2 "hmR_PruneChannels.m" (:cite:t:`Huppert2009`)
    """

    # calculate SNR
    snr = amplitudes.mean("time") / amplitudes.std("time")
    # create snr mask and update accoording to snr thresholds
    snr_mask = xrutils.mask(snr, CLEAN)
    snr_mask = snr_mask.where(snr > snr_thresh, TAINTED)

    return snr, snr_mask


@cdc.validate_schemas
def mean_amp(amplitudes: cdt.NDTimeSeries, amp_range: tuple[Quantity, Quantity]):
    """Calculate mean amplitudes and mask channels outside amplitude range.

    Args:
        amplitudes (:class:`NDTimeSeries`, (time, *)):  input time series
        amp_range: if amplitudes.mean("time") < amp_threshs[0] or > amp_threshs[1]
            then it is excluded as an active channel in amp_mask
    Returns:
        A tuple (mean_amp, amp_mask), where mean_amp is DataArray with coords from
        input NDTimeseries containing the amplitudes averaged over time. The boolean
        DataArray amp_mask  is true where amp_threshs are met

    References:
        Based on Homer3 v1.80.2 "hmR_PruneChannels.m" (:cite:t:`Huppert2009`)
    """
    # FIXME: default parameters in Homer3 were (1e4, 1e7). Adopt?

    # calculate mean amplitude
    mean_amp = amplitudes.mean("time")
    # create amplitude mask and update according to amp_range thresholds
    amp_mask = xrutils.mask(mean_amp, CLEAN)
    amp_mask = amp_mask.where(
        (mean_amp > amp_range[0]) & (mean_amp < amp_range[1]), TAINTED
    )

    return mean_amp, amp_mask


@cdc.validate_schemas
def sd_dist(
    amplitudes: cdt.NDTimeSeries,
    geo3D: cdt.LabeledPointCloud,
    sd_range: tuple[Quantity, Quantity] = (0 * units.cm, 4.5 * units.cm),
):
    """Calculate source-detector separations and mask channels outside a distance range.

    Args:
        amplitudes (:class:`NDTimeSeries`, (channel, *)): input time series
        geo3D (:class:`LabeledPointCloud`): 3D optode coordinates
        sd_range: if source-detector separation < sd_range[0] or > sd_range[1]
             then it is excluded as an active channelin sd_mask

    Returns:
        A tuple (sd_dist, sd_mask), where sd_dist contains the channel distances
        and sd_mask is a boolean `NDTimeSeries`, indicating where distances fall into
        sd_range.

    References:
        Based on Homer3 v1.80.2 "hmR_PruneChannels.m" (:cite:t:`Huppert2009`)
    """

    # calculate channel distances
    sd_dist = xrutils.norm(
        geo3D.loc[amplitudes.source] - geo3D.loc[amplitudes.detector],
        dim=geo3D.points.crs,
    ).round(3)
    # create sd_mask and update according to sd_thresh thresholds
    sd_mask = xrutils.mask(sd_dist, CLEAN)
    sd_mask = sd_mask.where((sd_dist > sd_range[0]) & (sd_dist < sd_range[1]), TAINTED)

    return sd_dist, sd_mask





@cdc.validate_schemas
def id_motion(
    fNIRSdata: cdt.NDTimeSeries,
    t_motion: Quantity = 0.5 * units.s,
    t_mask: Quantity = 1.0 * units.s,
    stdev_thresh: float = 50.0,
    amp_thresh: float = 5.0,
) -> cdt.NDTimeSeries:
    """Identify motion artifacts in an fNIRS input data array.

    If any active data channel exhibits a signal change greater than std_thresh or
    amp_thresh, then a segment of data around that time point is marked
    as a motion artifact.

    Args:
        fNIRSdata (:class:`NDTimeSeries`, (time, channel, *)): input time series

        t_motion (:class:`Quantity`, [time]): time interval for motion artifact
            detection. Checks for signal change indicative of a motion artifact over
            time range t_motion.


        t_mask (:class:`Quantity`, [time]): time range to mask around motion artifacts.
            Mark data over +/- t_mask around the identified motion artifact as a
            motion artifact.

        stdev_thresh: threshold for std deviation of signal change. If the signal d for
            any given active channel changes by more than stdev_thresh * stdev(d) over
            the time interval tMotion, then this time point is marked as a motion
            artifact. The standard deviation is determined for each channel
            independently. Typical value ranges from 5 to 20. Use a value of 100 or
            greater if you wish for this condition to not find motion artifacts.

        amp_thresh: threshold for amplitude of signal change. If the signal d for any
            given active channel changes by more that amp_thresh over the time interval
            t_motion, then this time point is marked as a motion artifact. Typical value
            ranges from 0.01 to 0.3 for optical density units. Use a value greater than
            100 if you wish for this condition to not find motion artifacts.

    Returns:
        a DataArray that has at least the dimensions channel and time, dtype is boolean.
        At each time point, CLEAN indicates data included and TAINTED indicates motion
        artifacts.

    References:
        Based on Homer3 v1.80.2 "hmR_MotionArtifact.m" and "hmR_MotionArtifactByChannel"
        (:cite:t:`Huppert2009`).
    """

    # TODO assert OD units, otherwise issue a warning

    # t_motion in samples rounded to the nearest sample
    t_motion_samples = t_motion / fNIRSdata.time.diff(dim="time").mean()
    t_motion_samples = t_motion_samples.pint.dequantify()
    t_motion_samples = int(t_motion_samples.round())
    # t_mask in samples rounded to the nearest sample
    t_mask_samples = t_mask / fNIRSdata.time.diff(dim="time").mean()
    t_mask_samples = t_mask_samples.pint.dequantify()
    t_mask_samples = int(t_mask_samples.round())

    # calculate the "std_diff", the standard deviation of the approx 1st derivative of
    # each channel over time
    std_diff = fNIRSdata.diff(dim="time").std(dim="time")
    # calc motion correction threshold
    mc_thresh = std_diff * stdev_thresh

    # calculate the differences across different time shifts from 1 to t_motion_samples
    diff = []
    for ii in range(1, t_motion_samples + 1):
        # Shift the data by X samples to the left in the 'time' dimension
        shifted_data = fNIRSdata.shift(time=-ii, fill_value=0)
        # zero padding of original data where shifted data is shorter
        fNIRSdata0 = fNIRSdata.copy()
        strt_zeroidx = fNIRSdata0.time[-ii]
        fNIRSdata0.loc[dict(time=slice(strt_zeroidx, None))] = 0
        # calc absolute differences
        diff.append(abs(shifted_data - fNIRSdata0))

    # calculate max_diff across all available time delays
    max_diff = xr.concat(diff, dim="diff").max(dim="diff")

    # create mask for artifact indication. True indicates artifact.
    art_ind = xrutils.mask(fNIRSdata, True)
    # updates mask according to motion correction thresholds mc_thresh and amp_thresh:
    # sets elements to true if either is exceeded.
    art_ind = art_ind.where((max_diff > mc_thresh) | (max_diff > amp_thresh), False)

    # apply mask to data to mask points surrounding motion artifacts
    # convolution kernel for masking
    ckernel = np.ones(2 * t_mask_samples + 1)

    # create mask: convolve the artifact indicators with the kernel to mask surrounding
    # samples. > 0 makes the result of the convolution again a boolean array
    ma_mask = xrutils.convolve(art_ind.astype(int), ckernel, "time") > 0

    # set time points marked as artifacts (True) to TAINTED.
    ma_mask = xr.where(ma_mask, TAINTED, CLEAN)

    return ma_mask


@cdc.validate_schemas
def id_motion_refine(ma_mask: cdt.NDTimeSeries, operator: str):
    """Refines motion artifact mask to simplify and quantify motion artifacts.

    Args:
        ma_mask (:class:`NDTimeSeries`, (time, channel, *)): motion artifact mask as
            generated by id_motion().

        operator: operation to apply to the mask. Available operators:

            - *by_channel*: collapses the mask along the amplitude/wavelength/chromo
              dimension to provide a single motion artifact mask per channel (default)
              over time
            - *all*: collapses the mask along all dimensions to provide a single motion
              artifact marker for all channels over time i.e. an artifact detected in
              any channel masks all channels.

    Returns:
        A tuple (ma_mask_new, ma_info), where `ma_mask_new` is the updated motion
        artifact mask, collapsed according to operator and `ma_info` is a
        `pandas.DataFrame` that contains 1) channels with motion artifacts, 2) # of
        artifacts detected per channel and 3) fraction of artifacts/total time.
    """
    # combine artifact masks (if multiple masks are provided).
    # Will result in a single mask containing all motion indicators
    mask = reduce(lambda x, y: x | y, ma_mask)

    # collapse motion artifact masks according to operator instruction
    if operator.lower() == "by_channel":
        # find whether "wavelength" or "concentration" exists as a dimension in ma_mask
        # and collapse, otherwise assert an error
        if "wavelength" in ma_mask.dims:
            ma_mask_new = ma_mask.all(dim="wavelength")
        elif "concentration" in ma_mask.dims:
            ma_mask_new = ma_mask.all(dim="concentration")
        else:
            raise ValueError(
                "ma_mask must have either 'wavelength' "
                "or 'concentration' as a dimension"
            )

        ## --- extract motion artifact info --- ##
        # extract channels that had motion artifacts
        ch_wma = ma_mask_new.all(dim="time")
        ch_labels = ch_wma.where(ch_wma, drop=False).channel.values
        # for every channel in ch_label calculate the fraction of time points that are
        # true over the total number of time points
        ch_frac = (
            1 - (ma_mask_new.sel(channel=ch_labels).sum(dim="time")
            / ma_mask_new.sizes["time"])
        ).to_series()
        # Count number of motion artifacts (transitions in the mask) for each channel
        transitions = ma_mask_new.astype(int).diff(dim="time") == -1
        transitions_ct = transitions.sum(dim="time").to_series()
        # Store motion artifact info in a pandas dataframe
        ma_info = pd.DataFrame(
            {"ma_fraction": ch_frac, "ma_count": transitions_ct}
        ).reset_index()

    # collapse mask along all dimensions
    elif operator.lower() == "all":
        dims2collapse = [dim for dim in ma_mask.dims if dim != "time"]
        ma_mask_new = ma_mask.all(dim=dims2collapse)

        ## --- extract motion artifact info --- ##
        global_frac = (mask.sum(dim="time") / mask.sizes["time"]).values
        # Count number of motion artifacts (transitions in the mask) for each channel
        transitions = mask.astype(int).diff(dim="time") == 1
        transitions_ct = transitions.sum(dim="time").values
        # Store motion artifact info in a pandas dataframe
        ma_info = pd.DataFrame(
            {
                "channel": ["all channels combined"],
                "ma_fraction": [global_frac],
                "ma_count": [transitions_ct],
            }
        )

    else:
        raise ValueError(f"unsupported operator '{operator}'")

    # set time points marked as artifacts (True) again to TAINTED.
    #ma_mask_new = xr.where(ma_mask_new, TAINTED, CLEAN)

    return ma_mask_new, ma_info


@cdc.validate_schemas
def detect_outliers_std(
    ts: cdt.NDTimeSeries, t_window: cdt.QTime, iqr_threshold=2
):
    """Detect outliers in fNIRSdata based on standard deviation of signal.

    Args:
        ts :class:`NDTimeSeries`, (time, channel, *): fNIRS timeseries data
        t_window :class:`Quantity`: time window over which to calculate std. deviations
        iqr_threshold: interquartile range threshold (detect outlier as any std.
            deviation outside iqr_threshold * [25th percentile, 75th percentile])

    Returns:
        mask that is a DataArray containing TRUE anywhere the data is clean and FALSE
        anytime an outlier is detected based on the standard deviation

    References:
        Based on Homer3 v1.80.2 "hmrR_tInc_baselineshift_Ch_Nirs.m"
        (:cite:t:`Jahani2018`)
    """

    ts = ts.pint.dequantify()
    fs = freq.sampling_rate(ts)

    window_size = int(np.round(fs * t_window))

    ts_lowpass = ts.cd.freq_filter(0, 0.5, butter_order=4)

    # FIXME shift due to window center. accounted for in original method but correct?
    # stride==1, i.e. there are as many windows as time samples
    windowed_std = ts_lowpass.rolling(
        time=window_size, min_periods=1, center=True
    ).std()

    qmin, qmax = 0.25, 0.75
    quantiles = windowed_std.quantile([qmin, 0.5, qmax], dim="time")

    IQR = quantiles.sel(quantile=qmax) - quantiles.sel(quantile=qmin)
    threshold_high = quantiles.sel(quantile=qmax) + iqr_threshold * IQR
    threshold_low = quantiles.sel(quantile=qmin) - iqr_threshold * IQR

    mask = xr.where(
        (windowed_std < threshold_low) | (threshold_high < windowed_std),
        TAINTED,
        CLEAN,
    )

    return mask


@cdc.validate_schemas
def detect_outliers_grad(ts: cdt.NDTimeSeries, iqr_threshold: float = 1.5):
    """Detect outliers in fNIRSdata based on gradient of signal.

    Args:
        ts (:class:`NDTimeSeries`, (time, channel, *)): fNIRS timeseries data
        iqr_threshold: interquartile range threshold (detect outlier as any gradient
            outside iqr_threshold * [25th percentile, 75th percentile])

    Returns:
        mask that is a DataArray containing TRUE anywhere the data is clean and FALSE
        anytime an outlier is detected

    References:
        Based on Homer3 v1.80.2 "hmrR_tInc_baselineshift_Ch_Nirs.m"
        (:cite:t:`Jahani2018`)
    """

    ts = ts.pint.dequantify()

    #ts_lowpass = ts.cd.freq_filter(0, 2, butter_order=4) # FIXME
    ts_lowpass = freq.freq_filter(ts, 0 * units.Hz, 2*units.Hz, butter_order=4)

    axis_time = list(ts_lowpass.dims).index("time")
    kernel_shape = np.ones(ts.ndim, dtype=int)
    kernel_shape[axis_time] = 3
    grad_kernel  = np.asarray([-1,0,1]).reshape(kernel_shape)

    assert ts_lowpass.dims[-1]
    gradient = xr.apply_ufunc(
        signal.oaconvolve,
        ts_lowpass,
        grad_kernel,
        kwargs={"mode": "same", "axes": axis_time },
    )

    qmin, qmax = 0.25, 0.75
    quantiles = gradient.quantile([qmin, 0.5, qmax], dim="time")

    IQR = quantiles.sel(quantile=qmax) - quantiles.sel(quantile=qmin)
    threshold_high = quantiles.sel(quantile=qmax) + iqr_threshold * IQR
    threshold_low = quantiles.sel(quantile=qmin) - iqr_threshold * IQR

    mask = xr.where(
        (gradient > threshold_high) | (gradient < threshold_low),
        TAINTED,
        CLEAN,
    )

    return mask


@cdc.validate_schemas
def detect_outliers(
    ts: cdt.NDTimeSeries,
    t_window_std: cdt.QTime,
    iqr_threshold_std: float = 2,
    iqr_threshold_grad: float = 1.5,
):
    """Detect outliers in fNIRSdata based on standard deviation and gradient of signal.

    Args:
        ts (:class:`NDTimeSeries`, (time, channel, *)): fNIRS timeseries data
        t_window_std (:class:`Quantity`): time window over which to calculate std. devs.
        iqr_threshold_grad: interquartile range threshold (detect outlier as any
            gradient outside iqr_threshold * [25th percentile, 75th percentile])
        iqr_threshold_std: interquartile range threshold (detect outlier as any standard
            deviation outside iqr_threshold * [25th percentile, 75th percentile])

    Returns:
        mask that is a DataArray containing TRUE anywhere the data is clean and FALSE
        anytime an outlier is detected

    References:
        Based on Homer3 v1.80.2 "hmrR_tInc_baselineshift_Ch_Nirs.m"
        (:cite:t:`Jahani2018`)
    """
    mask_std = detect_outliers_std(ts, t_window_std, iqr_threshold_std)
    mask_grad = detect_outliers_grad(ts, iqr_threshold_grad)

    return mask_std & mask_grad


def _mask1D_to_segments(mask: ArrayLike):
    """Find consecutive segments for a boolean mask.

    Args:
        mask (ArrayLike): boolean mask

    Returns:
        Given a boolean mask, this function returns an integer array `segments` of
        shape (nsegments,3) in which
        - segments[:,0] is the first index of the segment
        - segments[:,1]-1 is the last index of the segment and
        - segments[:,2] is the integer-converted mask value in that segment
    """

    # FIXME decide how to index:
    # [start,finish] as currently implemented or [start, finish]

    # pad mask on both ends with guaranteed state changes
    mask = np.r_[~mask[0], mask, ~mask[-1]]

    # find the indices where mask changes
    idx = np.flatnonzero(mask[1:] != mask[:-1])

    segments = np.fromiter(zip(idx[:-1], idx[1:], mask[idx[1:]]), dtype=(int, 3))

    return segments


def _calculate_snr(ts, fs, segments):
    """Calculate signal to noise ratio for a time series.

    Args:
        ts (ArrayLike): Time series
        fs (float): Sampling rate
        segments (ArrayLike): Segments of the time series

    Returns:
        float: Signal to noise ratio
    """
    # Calculate signal to noise ratio by considering only longer segments.
    # Only segments longer than 3s are used. Segments may be clean or tainted.
    long_seg_snr = [
        abs(seg.mean()) / (seg.std() + 1e-16)
        for i0, i1, _ in segments
        for seg in [ts[i0:i1]]
        if (i1 - i0) > (3 * fs)
    ]
    #print(f"long_seg_snr: {long_seg_snr}")
    if len(long_seg_snr) > 0:
        snr = np.mean(long_seg_snr) # FIXME was mean and suceptible to outliers
    else:
        # if there is no segment longer than 3s calculate snr ratio from all time points
        snr = abs(ts.mean()) / (ts.std() + 1e-16)

    return snr

def _calculate_delta_threshold(ts, segments, threshold_samples):
    """Calculate delta threshold for a time series.

    Args:
        ts (ArrayLike): Time series
        segments (ArrayLike): Segments of the time series
        threshold_samples (int): Threshold samples

    Returns:
        float: Delta threshold
    """
    # for long segments (>threshold_samples (0.5s)) that are not marked as artifacts
    # calculate the absolute differences of samples that are threshold_samples away
    # from each other
    seg_deltas = [
        np.abs(
            ts[np.arange(i0 + threshold_samples, i1)]
            - ts[np.arange(i0, i1 - threshold_samples)]
        )
        for i0, i1, seg_type in segments
        if (seg_type == CLEAN) and ((i1 - i0) > threshold_samples)
    ]

    seg_deltas = np.hstack(seg_deltas)
    # threshold defined by the 50% quantile of these differences, was ssttdd_thresh
    seg_deltas_thresh = np.quantile(seg_deltas, 0.5)

    return seg_deltas_thresh


def detect_baselineshift(ts: cdt.NDTimeSeries, outlier_mask: cdt.NDTimeSeries):
    """Detect baselineshifts in fNIRSdata.

    Args:
        ts (:class:`NDTimeSeries`, (time, channel, *)): fNIRS timeseries data
        outlier_mask (:class:`NDTimeSeries`): mask containing FALSE anytime an outlier
            is detected in signal

    Returns:
        mask that is a DataArray containing TRUE anywhere the data is clean and FALSE
        anytime a baselineshift or outlier is detected.

    References:
        Based on Homer3 v1.80.2 "hmrR_tInc_baselineshift_Ch_Nirs.m"
        (:cite:t:`Jahani2018`)
    """
    ts = ts.pint.dequantify()

    #ts = ts.stack(measurement=["channel", "wavelength"]).sortby("wavelength")
    #outlier_mask = outlier_mask.stack(measurement=["channel", "wavelength"]).sortby(
    #    "wavelength"
    #)

    # FIXME
    assert "channel" in ts.dims
    assert "wavelength" in ts.dims

    fs = ts.cd.sampling_rate # FIXME

    pad_samples = int(np.round(12 * fs))  # extension for padding. 12s
    threshold_samples = int(
        np.round(0.5 * fs)
    )  # threshold for baseline shift detection

    ts_lowpass = ts.cd.freq_filter(
        0, 2, butter_order=4
    )  # filter the data between 0-2Hz

    # pad timeseries for 12s before and after with edge values
    ts_padded = ts_lowpass.pad(time=pad_samples, mode="edge")

    # FIXME why does the original implementation filter twice?
    ts_lowpass_padded = ts_padded.cd.freq_filter(
        0, 2, butter_order=4
    )  # filter the data between 0-2Hz

    outlier_mask_padded = outlier_mask.pad(time=pad_samples, mode="edge")

    shift_mask = xrutils.mask(ts_padded, CLEAN)

    snr_thresh = 3

    for ch in ts_padded.channel.values:
        channel_snrs = []
        channel_masks = []
        for wl in ts.wavelength.values:
            channel = ts_padded.sel(channel=ch, wavelength=wl).values
            channel_lowpass = ts_lowpass_padded.sel(channel=ch, wavelength=wl).values
            channel_mask = outlier_mask_padded.sel(channel=ch, wavelength=wl).values

            segments = _mask1D_to_segments(channel_mask)

            channel_snrs.append(_calculate_snr(channel, fs, segments))

            # was motion_kind. delta between segment start and end
            segment_deltas = np.abs(
                channel[segments[:, 1] - 1] - channel[segments[:, 0]]
            )

            segment_delta_threshold = _calculate_delta_threshold(
                channel_lowpass, segments, threshold_samples
            )

            channel_mask = np.ones(len(channel), dtype=bool)
            seg_length_min = 0.1 * fs
            seg_length_max = 0.49999 * fs
            for i_seg, (i0, i1, seg_type) in enumerate(segments):
                if seg_type == CLEAN:
                    continue

                seg_length = i1 - i0

                # flag segments where the delta between start and end is too large
                if segment_deltas[i_seg] > segment_delta_threshold:
                    channel_mask[i0:i1] = TAINTED

                # flag segments that are too short
                if (seg_length_min < seg_length) and (seg_length < seg_length_max):
                    channel_mask[i0:i1] = TAINTED

                # flag segments that are too long?
                # if seg_length > fs:
                #    channel_mask[i0:i1] = True

            channel_masks.append(channel_mask)

        channel_snrs = np.asarray(channel_snrs)

        if (channel_snrs < snr_thresh).all():
            # if all wavelengths for this channel are below the snr threshold
            # mark all wavelengths as tainted.
            logger.debug(f"marking complete channel {ch} as TAINTED due to low SNR.")
            shift_mask.sel(channel=ch).values[:] = TAINTED
        else:
            # take the wavelength with the highest SNR and use its mask for all other
            # wavelengths
            max_snr_mask = channel_masks[np.argmax(channel_snrs)]
            for wl in ts.wavelength.values:
                shift_mask.sel(channel=ch, wavelength=wl).values[:] = max_snr_mask

    # remove padding
    shift_mask = shift_mask.isel(time=slice(pad_samples,-pad_samples))

    return shift_mask



def stimulus_mask(df_stim : pd.DataFrame, mask : xr.DataArray) -> xr.DataArray:
    """Create a mask which events overlap with periods flagged as tainted in mask.

    Args:
        df_stim: stimulus data frame
        mask: signal quality mask. Must contain dimensions 'channel' and 'time'

    Returns:
        A boolean mask with dimensions "stim", "channel".
        The stim dimension matches the stimulus dataframe. Stimuli are marked as
        TAINTED when there is any TAINTED flag in the mask between onset and onset+
        duration.
    """
    assert mask.ndim == 2
    assert "channel" in mask.dims
    assert "time" in mask.dims

    result = np.zeros((len(df_stim), mask.sizes["channel"]), dtype=bool)

    for i, r in df_stim.iterrows():
        tmp = mask.sel(
            time=(r["onset"] <= mask.time) & (mask.time < (r["onset"] + r["duration"]))
        )
        result[i,:] = (tmp == CLEAN).all("time")

    return xr.DataArray(
        result,
        dims=["stim", "channel"],
        coords=xrutils.coords_from_other(
            mask,
            dims=["channel"],
            stim=("stim", df_stim.index),
            trial_type=("stim", df_stim.trial_type),
        ),
    )

def repair_amp(amp: xr.DataArray, median_len=3, interp_nan=True, **kwargs):
    """Replace nonpositive amp values and optionally fill NaNs.

    TODO: Optimize handling of sequential nonpositive values.

    Args:
        amp: Amplitude data
        median_len: Window size for the median filter
        interp_nan: If True, interpolate NaNs in the data
        **kwargs: Additional arguments for xarray interpolate_na function, such
            as method = "linear" (default), method = "nearest", etc. See xarray
            documentation for more details.
    """
    pad_width = median_len // 2

    # Fill NaNs
    if interp_nan:
        amp = amp.pint.dequantify()
        amp = amp.interpolate_na(dim="time", **kwargs)
        amp = amp.pint.quantify()

    # Replace nonpositive values with a small value
    unit = amp.pint.units
    amp = amp.where(amp>0, 1e-18 * unit)

    if median_len > 1:
        # Pad the data before applying the median filter
        padded_amp = amp.pad(time=(pad_width, pad_width), mode="edge")

        # Apply median filter
        filtered_padded_amp = (
            padded_amp.rolling(time=median_len, center=True)
            .reduce(np.median)
        )
        # Trim the padding after applying the filter
        return filtered_padded_amp.isel(time=slice(pad_width, -pad_width))

    return amp


def measurement_variance(
    ts: xr.DataArray,
    list_bad_channels: list = None,
    bad_rel_var: float = 1e6,
    bad_abs_var: float = None,
    calc_covariance: bool = False,
) -> xr.DataArray:
    """Estimate measurement variance or covariance from an fNIRS time series.

    Args:
    ts: Input time series with dimensions (time, channel, chromo/wavelength).
    list_bad_channels: List of channel names (e.g. ["S2D4", "S2D10"]) to be treated as
        bad.
    bad_rel_var: Multiplier for worst-case variance for bad channels if `bad_abs_var` is
        not provided.
    bad_abs_var: Absolute variance to assign to bad channels. Overrides `bad_rel_var` if
        provided.
    calc_covariance: If True, returns a 3D covariance matrix: (other_dim, channel,
        channel). If False, returns a 2D variance array: (chromo, channel).

    Returns:
    xr.DataArray: Variance array (shape: channel, chromo/wavelength) or covariance
        array (shape: chromo/wavelength, channel1, channel2)
    """
    if list_bad_channels is None:
        list_bad_channels = []

    # Identify other dimension (chromo/wavelength)
    other_dim = xrutils.other_dim(ts, "time", "channel")
    other_dim_values = ts[other_dim].values

    # Get units
    unit = ts.pint.units if hasattr(ts, "pint") else 1
    ts_copy = ts.pint.dequantify() if hasattr(ts, "pint") else ts.copy()

    # Compute variance
    var = ts_copy.var(dim="time")

    # Create bad channel mask
    zero_var_mask = (var == 0)
    bad_channels_from_zero_var = set()
    if zero_var_mask.any():
        for channel in ts.channel.values:
            if zero_var_mask.sel(channel=channel).any():
                bad_channels_from_zero_var.add(channel)

    # Combine with explicitly passed bad channels
    all_bad_channels = bad_channels_from_zero_var.union(set(list_bad_channels))
    valid_bad_channels = [ch for ch in all_bad_channels if ch in ts.channel.values]

    # Create mask for entire bad channels (across all other_dim coordinates)
    bad_channel_mask = ts.channel.isin(valid_bad_channels)

    # Compute bad variance fill value
    good_var = var.where(~bad_channel_mask)
    max_good_var = good_var.max().item() if good_var.notnull().any() else 1.0
    var_fill_value = bad_abs_var if bad_abs_var is not None else bad_rel_var * \
        max_good_var

    # Replace variance of bad channels
    var = var.where(~bad_channel_mask, other=var_fill_value)

    if not calc_covariance:
        return var * unit**2

    # Initialize 3D covariance array, shape (other_dim, channel, channel)
    channels = ts_copy.channel.values
    n_other_dim, n_channels = len(other_dim_values), len(channels)
    cov_matrix_3d = np.zeros((n_other_dim, n_channels, n_channels))
    bad_ch_indices = np.array([ch in valid_bad_channels for ch in channels])

    # Compute covariance for each other_dim coordinate
    for i, other_val in enumerate(other_dim_values):
        data_slice = ts_copy.sel({other_dim: other_val})
        data_matrix = data_slice.values  # Shape: (n_channels, n_time)
        cov_matrix_2d = np.cov(data_matrix, ddof=1)  # Shape: (n_channels, n_channels)

        # Replace covariance of bad channels
        if valid_bad_channels:
            # Get max off-diagonal covariance from good channels
            good_cov_matrix = cov_matrix_2d.copy()
            good_cov_matrix[bad_ch_indices, :] = np.nan
            good_cov_matrix[:, bad_ch_indices] = np.nan
            np.fill_diagonal(good_cov_matrix, np.nan)
            max_good_cov = np.nanmax(np.abs(good_cov_matrix)) if not \
                np.isnan(good_cov_matrix).all() else 0

            # Replace covariance of bad channels
            cov_fill_value = var_fill_value if bad_abs_var is not None else \
                bad_rel_var * max_good_cov
            cov_matrix_2d[bad_ch_indices, :] = cov_fill_value
            cov_matrix_2d[:, bad_ch_indices] = cov_fill_value
            bad_diag_mask = np.diag(bad_ch_indices)
            cov_matrix_2d[bad_diag_mask] = var_fill_value

        cov_matrix_3d[i] = cov_matrix_2d

    # Create coordinate arrays
    coords = {
        other_dim: other_dim_values,
        "channel1": channels,
        "channel2": channels,
    }

    cov_da = xr.DataArray(
        cov_matrix_3d,
        dims=(other_dim, "channel1", "channel2"),
        coords=coords,
    )

    return cov_da * unit**2
