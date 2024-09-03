"""Signal quality metrics and channel pruning functionality."""

import logging
from functools import reduce
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from scipy import signal

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
import cedalion.sigproc.frequency as freq
from cedalion import Quantity, units
from cedalion.typing import NDTimeSeries

from .frequency import freq_filter, sampling_rate

logger = logging.getLogger("cedalion")

CLEAN = True
TAINTED = False

@cdc.validate_schemas
def prune_ch(
    amplitudes: cdt.NDTimeSeries, masks: list[cdt.NDTimeSeries], operator: str
):
    """Prune channels from the the input data array using quality masks.

    Args:
        amplitudes (:class:`NDTimeSeries`): input time series
        masks (:class:`list[NDTimeSeries]`) : list of boolean masks with coordinates
            comptabile to amplitudes

        operator: operators for combination of masks before pruning data_array

            - "all": logical AND, keeps channel if it is good across all masks
            - "any": logical OR, keeps channel if it is good in any mask/metric

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
    amplitudes, prune_list = xrutils.apply_mask(
        amplitudes, mask, "drop", dim_collapse="channel"
    )

    return amplitudes, prune_list


# PSP > threshold is CLEAN
@cdc.validate_schemas
def psp(
    amplitudes: NDTimeSeries,
    window_length: Annotated[Quantity, "[time]"],
    psp_thresh: float,
):
    # FIXME make these configurable
    cardiac_fmin = 0.5 * units.Hz
    cardiac_fmax = 2.5 * units.Hz

    amplitudes = amplitudes.pint.dequantify()
    amp = freq_filter(amplitudes, cardiac_fmin, cardiac_fmax, butter_order=4)
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
    fs = amp.cd.sampling_rate

    psp = np.zeros([len(windows["channel"]), len(windows["time"])])
    
    # Vectorized signal extraction and correlation
    sig = windows.transpose("channel", "time", "wavelength", "window").values
    psp = np.zeros((sig.shape[0], sig.shape[1]))
    lags = np.arange(-nsamples + 1, nsamples)

    for w in range(sig.shape[1]): # loop over windows
        sig_temp = sig[:,w,:,:]
        # FIXME assumes 2 wavelengths
        corr = np.array(
            [
                signal.correlate(sig_temp[ch, 0, :], sig_temp[ch, 1, :], "full")
                for ch in range(sig.shape[0])
            ]
        )

        # FIXME assumes 2 wavelengths
        corr = corr /(nsamples - np.abs(lags))

        nperseg = corr.shape[1]
        window = np.hamming(nperseg)
        window_seg = corr * window
        
        fft_out = np.fft.rfft(window_seg, axis=1)
        psd = (np.abs(fft_out) ** 2) / (fs * np.sum(window ** 2))
        freqs = np.fft.rfftfreq(nperseg, 1/fs)

        # for ch in range(sig.shape[0]):
        #     window = signal.windows.hamming(len(corr[ch,:]))
        #     f, pxx = signal.welch(
        #         corr[ch, :],
        #         window=window,
        #         nfft=len(corr[ch, :]),
        #         fs=fs,
        #         scaling="density",
        #     )

        psp[:, w] = np.max(psd, 1)

    # keep dims channel and time
    
    psp_xr = windows.isel(wavelength=0, window=0).drop_vars("wavelength").copy(data=psp)

    # Apply threshold mask
    psp_mask = xrutils.mask(psp_xr, CLEAN)
    psp_mask = psp_mask.where(psp_xr > psp_thresh, other=TAINTED)

    return psp_xr, psp_mask


@cdc.validate_schemas
def sci(amplitudes: NDTimeSeries, window_length: Quantity, sci_thresh: float):
    """Calculate the scalp-coupling index.

    The scalp-coupling index metric is based on :cite:t:`Pollonini2014` /
    :cite:t:`Pollonini2016`.

    Args:
        amplitudes (:class:`NDTimeSeries`, (channel, wavelength, time)): input time
            series
        window_length (:class:`Quantity`, [time]): size of the computation window
        sci_thresh: if the calculated SCI metric falls below this threshold then the
            corresponding time window should be excluded.

    Returns:
        A tuple (sci, sci_mask), where sci is a DataArray with coords from the input
        NDTimeseries containing the scalp-coupling index. Sci_mask is a boolean mask
        DataArray with coords from sci, true where sci_thresh is met.
    """

    assert "wavelength" in amplitudes.dims  # FIXME move to validate schema

    # FIXME make these configurable
    cardiac_fmin = 0.5 * units.Hz
    cardiac_fmax = 2.5 * units.Hz

    amp = freq_filter(amplitudes, cardiac_fmin, cardiac_fmax, butter_order=4)
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


@cdc.validate_schemas
def snr(amplitudes: cdt.NDTimeSeries, snr_thresh: float = 2.0):
    """Calculates signal-to-noise ratio for each channel and other dimension.

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
    t_motion_samples = int(t_motion_samples.round())
    # t_mask in samples rounded to the nearest sample
    t_mask_samples = t_mask / fNIRSdata.time.diff(dim="time").mean()
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
        ma_mask :class:`NDTimeSeries`, (time, channel, *): motion artifact mask as
        generated by id_motion().

        operator: operation to apply to the mask. Available operators:

            - "by_channel": collapses the mask along the amplitude/wavelength/chromo
                dimension to provide a single motion artifact mask per channel (default)
                over time
            - "all": collapses the mask along all dimensions to provide a single motion
                artifact marker for all channels over time i.e. an artifact detected in
                any channel masks all channels.

    Returns:
        A tuple (ma_mask_new, ma_info), where `ma_mask_new` is the updated motion
        artifact mask, collapsed according to operator and `ma_info` is a
        `pandas.DataFrame` that contains 1) channels with motion artifacts, 2) # of
        artifacts detected per channel and 3) fraction of artifacts/total time.
    """

    # implementation expects artifacts to be marked as True
    ma_mask = ma_mask == TAINTED

    # combine artifact masks (if multiple masks are provided).
    # Will result in a single mask containing all motion indicators
    mask = reduce(lambda x, y: x | y, ma_mask)

    # collapse motion artifact masks according to operator instruction
    if operator.lower() == "by_channel":
        # find whether "wavelength" or "concentration" exists as a dimension in ma_mask
        # and collapse, otherwise assert an error
        if "wavelength" in ma_mask.dims:
            ma_mask_new = ma_mask.any(dim="wavelength")
        elif "concentration" in ma_mask.dims:
            ma_mask_new = ma_mask.any(dim="concentration")
        else:
            raise ValueError(
                "ma_mask must have either 'wavelength' "
                "or 'concentration' as a dimension"
            )

        ## --- extract motion artifact info --- ##
        # extract channels that had motion artifacts
        ch_wma = ma_mask_new.any(dim="time")
        ch_labels = ch_wma.where(ch_wma, drop=True).channel.values
        # for every channel in ch_label calculate the fraction of time points that are
        # true over the total number of time points
        ch_frac = (
            ma_mask_new.sel(channel=ch_labels).sum(dim="time")
            / ma_mask_new.sizes["time"]
        ).to_series()
        # Count number of motion artifacts (transitions in the mask) for each channel
        transitions = ma_mask_new.astype(int).diff(dim="time") == 1
        transitions_ct = transitions.sum(dim="time").to_series()
        # Store motion artifact info in a pandas dataframe
        ma_info = pd.DataFrame(
            {"ma_fraction": ch_frac, "ma_count": transitions_ct}
        ).reset_index()

    # collapse mask along all dimensions
    elif operator.lower() == "all":
        dims2collapse = [dim for dim in ma_mask.dims if dim != "time"]
        ma_mask_new = ma_mask.any(dim=dims2collapse)

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
    ma_mask_new = xr.where(ma_mask_new, TAINTED, CLEAN)

    return ma_mask_new, ma_info


@cdc.validate_schemas
def detect_outliers_std(
    ts: cdt.NDTimeSeries, t_window: Annotated[Quantity, "[time]"], iqr_threshold=2
):
    """Detect outliers in fNIRSdata based on standard deviation of signal."""

    ts = ts.pint.dequantify()
    fs = freq.sampling_rate(ts)

    # window size in samples TODO t_window units
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
def detect_outliers_grad(ts: cdt.NDTimeSeries, iqr_threshold=1.5):
    """Detect outliers in fNIRSdata based on gradient of signal."""

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
    t_window_std: Annotated[Quantity, "[time]"],
    iqr_threshold_std : float =2,
    iqr_threshold_grad : float =1.5,
):
    mask_std = detect_outliers_std(ts, t_window_std, iqr_threshold_std)
    mask_grad = detect_outliers_grad(ts, iqr_threshold_grad)

    return mask_std & mask_grad


def _mask1D_to_segments(mask: ArrayLike):
    """Find consecutive segments for a boolean mask.

    Given a boolean mask, this function returns an integer array `segements` of
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

            channel_mask = np.zeros(len(channel), dtype=bool)
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
