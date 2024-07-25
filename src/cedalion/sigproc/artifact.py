import numpy as np
from functools import reduce
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity, units
import xarray as xr
from scipy import signal
import pint
import pandas as pd
import logging
import time
from numpy.typing import ArrayLike

logger = logging.getLogger("cedalion")

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
        At each time point, False indicates data included and True indicates motion
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

    # create mask for artifact indication
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

    return ma_mask


#%% DETECTING OUTLIERS - OLD

@cdc.validate_schemas
def detect_outliers(fNIRSdata: cdt.NDTimeSeries, t_motion: Quantity):
    """Detect outliers in fNIRSdata based on standard deviation and gradient of signal 
    
    Based on Homer3 [1] v1.80.2 "hmrR_tInc_baselineshift_Ch_Nirs.m"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3

    INPUTS: 
        fNIRSdata: NDTimeSeries, input fNIRS data array with at least time and channel dimensions. Data should be in optical density units
        t_motion: time in seconds for calculating windowed standard deviation 
        
    OUTPUTS:
        M: xarray that has at least dimensions channels and time, units are boolean. Each time point is marked either True to indicate data
        is an outlier and False to indicate otherwise
    """
    IQR_GRAD_THRESH = 1.5
    IQR_STD_THRESH = 2
    
    fNIRSdata = fNIRSdata.stack(measurement = ['channel', 'wavelength']).sortby('wavelength').pint.dequantify()

    fs =  fNIRSdata.cd.sampling_rate
    window_size = int(np.round(fs*t_motion)) # define window size for calculating std 

    # initialize mask 
    M = xrutils.mask(fNIRSdata, True)
    

    for m, measurement in enumerate(fNIRSdata['measurement']):
        
        channel = fNIRSdata.sel(measurement=measurement)
        
        # filter data between 0-2Hz
        channel_lpf2 = channel.cd.freq_filter(0, 2, butter_order=4)
                
        # filter data between 0-0.5Hz
        channel_lpf05 = channel.cd.freq_filter(0, 0.5, butter_order=4)
        
        # detect outliers in std variations of the signal 
        windows = channel_lpf05.rolling(time=window_size).construct("window", stride=1)
        windows = windows[window_size:]
        sigSTD = windows.std("window")

        # define thresholds for std based on the first and fourth quartiles of the signals std
        quants_std = sigSTD.quantile([0.25,0.5,0.75])
        quants_std = np.quantile(sigSTD, [0.25, 0.5, 0.75])
        IQR_std = quants_std[2] - quants_std[0]
        high_std = quants_std[2] + IQR_std*IQR_STD_THRESH
        low_std = quants_std[0] - IQR_std*IQR_STD_THRESH
        # create mask to identify where std is above or below the two thresholds
        high = np.where(sigSTD > high_std)[0]
        low = np.where(sigSTD < low_std)[0]
        std_mask = np.unique(np.hstack([high, low]))

        offset = np.round(window_size/2).astype(int)
        std_mask=offset+std_mask;

        # detect outliers in gradient of the signal
        grad = xr.apply_ufunc(signal.convolve, channel_lpf2, [-1, 0, 1],'same', input_core_dims=[["time"],[],[]], output_core_dims=[["time"]])    
        # define thresholds for grad based on the first and fourth quartiles of the signal gradient
        quants_grad = grad.quantile([0.25,0.5,0.75])
        IQR_grad = quants_grad[2] - quants_grad[0]
        high_grad = quants_grad[2] + IQR_grad*IQR_GRAD_THRESH
        low_grad = quants_grad[0] - IQR_grad*IQR_GRAD_THRESH
        
        # create mask to idenfity where grad is above or below the two thresholds
        high = np.where(grad > high_grad)[0]
        low = np.where(grad < low_grad)[0]
        grad_mask = np.unique(np.hstack([high, low]))
        
        # union of both masks
        masks = np.unique(np.hstack([std_mask, grad_mask]))
        masks = masks
        
        M.values[masks,m] = 0

    # apply masks to fNIRSdata
    M_array, M_elements = xrutils.apply_mask(fNIRSdata, M, 'nan', 'none')
    
    M = M.unstack('measurement').pint.quantify()
    M_array = M_array.unstack('measurement').pint.quantify()
    
    M = M.transpose("channel","wavelength","time")
    M_array = M_array.transpose("channel","wavelength","time")
    
    return M, M_array

#%% DETECT OUTLIERS - NEW 

@cdc.validate_schemas
def detect_outliers_std(ts: cdt.NDTimeSeries, t_window: Quantity, iqr_threshold=2):
    """Detect outliers in fNIRSdata based on standard deviation of signal."""

    ts = ts.pint.dequantify()
    fs = ts.cd.sampling_rate

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

    mask = (windowed_std < threshold_low) | (threshold_high < windowed_std)

    return mask


@cdc.validate_schemas
def detect_outliers_grad(ts: cdt.NDTimeSeries, iqr_threshold=1.5):
    """Detect outliers in fNIRSdata based on gradient of signal."""

    ts = ts.pint.dequantify()

    ts_lowpass = ts.cd.freq_filter(0, 2, butter_order=4)

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

    mask = (gradient > threshold_high) | (gradient < threshold_low)

    return mask


@cdc.validate_schemas
def detect_outliers_2(
    ts: cdt.NDTimeSeries,
    t_window_std: Quantity,
    iqr_threshold_std=2,
    iqr_threshold_grad=1.5,
):
    mask_std = detect_outliers_std(ts, t_window_std, iqr_threshold_std)
    mask_grad = detect_outliers_grad(ts, iqr_threshold_grad)

    return mask_std | mask_grad



#%% MOTION ID REFINE

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

    return ma_mask_new, ma_info

        
#%% DETECTING BASELINE SHIFTS - OLD

def detect_baselineshift(fNIRSdata: cdt.NDTimeSeries, M: cdt.NDTimeSeries):
    """
    Detects baseline shifts in fNIRS data using motion detection and correction techniques.

    Based on Homer3 [1] v1.80.2 "hmrR_tInc_baselineshift_Ch_Nirs.m"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3
    Inputs:
        fNIRSdata (cdt.NDTimeSeries): The fNIRS data as a NDTimeSeries object.
        M (xr.DataArray): The motion detection array.

    Returns:
        tIncCh (cdt.NDTimeSeries): xarray that has at least dimensions time, units are boolean. Each time point is marked either True to indicate data
        is an outlier and False to indicate otherwise - collapsed across channels
        tInc (cdt.NDTimeSeries): xarray containing information from tIncCh collapsed across channels
        
    """
    t1 = time.time()
    fNIRSdata = fNIRSdata.stack(measurement = ['channel', 'wavelength']).sortby('wavelength').pint.dequantify()
    M = M.stack(measurement = ['channel', 'wavelength']).sortby('wavelength').pint.dequantify()

    fs = fNIRSdata.cd.sampling_rate

    extend = int(np.round(12*fs)) # extension for padding
    Nthresh = int(np.round(0.5*fs)) # threshold for baseline shift detection (samples in 0.5 sec)

    fNIRSdata_lpf2 = fNIRSdata.cd.freq_filter(0, 2, butter_order=4) # fitler the data between 0-2Hz

    fNIRSdata_pad = fNIRSdata_lpf2.pad(time=extend, mode='edge') # pad data by 12 seconds 
    fNIRSdata_pad['samples'].values = np.arange(len(fNIRSdata_pad))
    nT_pad = len(fNIRSdata_pad.samples)

    tInc = xrutils.mask(fNIRSdata_pad, True) # initialize mask for data to indicate timepoints that are included
    mean_SNR_threshold = np.zeros(len(fNIRSdata)) # initialize array for signal to noise ratio threshold per channel

    M_pad = M.pad(time=extend, mode='edge') # pad M by 12 seconds also
    M_pad['samples'].values = np.arange(len(M_pad))

    mean_SNR_threshold = np.zeros(len(fNIRSdata.measurement))
        
    for m,measurement in enumerate(fNIRSdata.measurement):
     
        channel = fNIRSdata_pad.sel(measurement=measurement)
        channel_M = M_pad.sel(measurement=measurement)
        if sum(channel_M.values) == 0:
            tinc = np.ones(nT_pad)
            tInc[:,m] = tinc

            continue
        
        sig = channel_M.copy()
        sig.values = channel_M.values.astype(int)
        # sig = channel.copy()
        # sig.values = np.ones(channel.shape) # initialize array for baseline shift detection
        # sig.values[np.where(channel_M.values == True)[0]] = 0 # set indices where motion is detected to 0
        
        # find locations where signal goes from clean -> motion -> clean
        temp = sig.diff('time')
        
        # create lists that indicate the start and finish of each motion segment
        lstMs = np.where(temp==-1)[0]
        lstMf = np.where(temp==1)[0]
        
        if len(lstMs) == 0:
            lstMs = [0]
        if len(lstMf) == 0:
            lstMf = np.asarray([len(sig)-1])
        if lstMs[0] > lstMf[0]:
            lstMs = np.insert(lstMs, 0, 0)
        if lstMs[-1] > lstMf[-1]:
            lstMf = np.append(lstMf, len(sig)-1)
        
        # get OD amplitude at starts and finish of each motion segment
        meanpH = channel[lstMs]
        meanpL = channel[lstMf]

        motion_kind = abs(meanpH.values-meanpL.values) # take the difference between the start and finish of each segment

        
        noise = channel.copy() # define the noise of the signal
        channel = channel.cd.freq_filter(0, 2, butter_order=4) # refilter the data 
       
        tt=0
        a=[]
        b=[]
        sigtemp=[]
        signoise=[]
        
        # create list of segments on either side of a baseline shift  
        for kk in range(len(sig)-1):
        
            if sig.values[kk] == 1:
                a.append(channel.values[kk])
                b.append(noise.values[kk])
                if sig.values[kk+1] == 0 and sig.values[kk] == 1:
                    sigtemp.append(a)
                    signoise.append(b)
                    a=[]
                    b=[]
                    tt=tt+1
                    
        ssttdd = []
        tempo2=[]
        # for each segment, if it is longer than Nthresh, find difference between elements separated by Nthresh
        for ss in range(tt):
            tempo = sigtemp[ss]
            if len(tempo)>Nthresh:
                for jj in range(len(tempo)-Nthresh):
                    tempo2.append((abs(tempo[jj+Nthresh]-tempo[jj])))
                    
            ssttdd.extend(tempo2) 
            tempo2 = []
             
        # TODO fix this 
        ssttdd = np.array(ssttdd)
        ssttdd_thresh = np.quantile(ssttdd,0.5) # get threshold defined by these differences
  
            
        countnoise = 0 # keep track of the number of segments that are longer than Nthresh
        SNR_threshold = []
        for jj in range(len(signoise)):
            if len(signoise[jj]) > 3*fs:
                countnoise = countnoise+1
                dmean = np.mean(signoise[jj]);
                dstd = np.std(signoise[jj]);
                SNR_threshold.append(abs(dmean)/(dstd+1e-16)) # get the signal to noise ratio for each segment if segment is longer than 3 seconds
            
        if len(SNR_threshold[1:len(SNR_threshold)-1]) > 0:
            mean_SNR_threshold[m] = np.mean(SNR_threshold) # get the mean signal to noise ratio for all segments longer than 3 seconds
        
        tinc = np.ones(nT_pad)
        # loop over each motion segment 
        for kk in range(len(lstMs)):
            
            if motion_kind[kk] > ssttdd_thresh: # if the difference between the start and finish of a segment is greater than the threshold, set the segment to 0
                tinc[lstMs[kk]:lstMf[kk]] = 0
                
            if ((lstMf[kk] - lstMs[kk]) > 0.1*fs) and ((lstMf[kk] - lstMs[kk]) < 0.49999*fs): # if the segment is longer than 0.1 seconds and shorter than 0.5 seconds, set the segment to 0
                tinc[lstMs[kk]:lstMf[kk]] = 0

            # FIXME comparison time and frequency?? bug?
            if lstMf[kk] - lstMs[kk] > fs: # if the segment is longer than the sampling frequency, set the segment to 0
                tinc[lstMs[kk]:lstMf[kk]] = 0

        tInc[:,m] = tinc

    # get SNR for all channels if it has not already been calculated 
    for ww, measurement in enumerate(fNIRSdata['measurement']):
        if np.isnan(mean_SNR_threshold[ww]) or mean_SNR_threshold[ww] == 0:
            mean_SNR_threshold[ww] = abs(fNIRSdata.sel(measurement=measurement).mean('time')/fNIRSdata.sel(measurement=measurement).std('time'))

    nChan = len(fNIRSdata['channel'])//2  # get the number of channels 

    SNRvalue = 3

    # make sure the same mask is applied to both channels
    for ww in range(nChan):
        
        if mean_SNR_threshold[ww] < SNRvalue and mean_SNR_threshold[ww+nChan] < SNRvalue: # if the SNR is less than the threshold for both channels, set channels to ones
            tInc[:,ww+nChan] = np.ones(len(fNIRSdata_pad))
            tInc[:,ww] = np.ones(len(fNIRSdata_pad))
        
        elif mean_SNR_threshold[ww] > SNRvalue and mean_SNR_threshold[ww+nChan] < SNRvalue: # if the SNR is greater than the threshold for the first channel and less than the threshold for the second channel, set the second channel to the first channel
            tInc[:,ww+nChan] = tInc[:,ww]
        
        elif mean_SNR_threshold[ww] < SNRvalue and mean_SNR_threshold[ww+nChan] > SNRvalue: # if the SNR is less than the threshold for the first channel and greater than the threshold for the second channel, set the first channel to the second channel
            tInc[:,ww] = tInc[:,ww+nChan]
        
    # remove the padding
    tIncCh = tInc[extend:-extend,:].pint.quantify()

    tIncall = tIncCh[:,0]

    # get the union of the masks for all channels
    for kk in range(tInc.shape[1]):
        tIncall = tIncall * tIncCh[:,kk]

    tInc = tIncall
    tIncCh = tIncCh.unstack('measurement').pint.quantify()

    t2 = time.time()
    logger.debug(f"finished detect_baselineshift in t2-t1 {t2-t1:.3f}s")

    return tInc, tIncCh

#%% DETECT BASELINE SHIFTS - NEW 

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

CLEAN = False
TAINTED = True

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


def detect_baselineshift_2(ts: cdt.NDTimeSeries, outlier_mask: cdt.NDTimeSeries):
    ts = ts.pint.dequantify()
    
    #ts = ts.stack(measurement=["channel", "wavelength"]).sortby("wavelength")
    #outlier_mask = outlier_mask.stack(measurement=["channel", "wavelength"]).sortby(
    #    "wavelength"
    #)

    # FIXME 
    assert "channel" in ts.dims
    assert "wavelength" in ts.dims

    fs = ts.cd.sampling_rate

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


