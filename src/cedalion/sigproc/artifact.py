import numpy as np
from functools import reduce
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity, units
import xarray as xr
from scipy import signal
import pint


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



    # calulate max_diff across channels for different time delays
    



    return fNIRSdata

#%%

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
        sigSTD = windows.std("window")
        
        # define thresholds for std based on the first and fourth quartiles of the signals std
        quants_std = sigSTD.quantile([0.25,0.5,0.75])
        IQR_std = quants_std[2] - quants_std[0]
        high_std = quants_std[2] + IQR_std*IQR_STD_THRESH
        low_std = quants_std[0] - IQR_std*IQR_STD_THRESH
        # create mask to identify where std is above or below the two thresholds
        std_mask = xrutils.mask(sigSTD, True)
        std_mask = std_mask.where((sigSTD > high_std) | (sigSTD < low_std), False)
        
        # detect outliers in gradient of the signal
        grad = xr.apply_ufunc(signal.convolve, channel_lpf2, [-1, 0, 1],'same', input_core_dims=[["time"],[],[]], output_core_dims=[["time"]])    
        
        # define thresholds for grad based on the first and fourth quartiles of the signal gradient
        quants_grad = grad.quantile([0.25,0.5,0.75])
        IQR_grad = quants_grad[2] - quants_grad[0]
        high_grad = quants_grad[2] + IQR_grad*IQR_GRAD_THRESH
        low_grad = quants_grad[0] - IQR_grad*IQR_GRAD_THRESH
        # create mask to idenfity where grad is above or below the two thresholds
        grad_mask = xrutils.mask(grad, True)
        grad_mask = grad_mask.where((grad > high_grad) | (grad < low_grad), False) 
  
        # union of both masks
        masks = [std_mask, grad_mask]
        mask_channel = reduce(lambda x, y: x | y, masks)
        
        M.values[:,m] = mask_channel.values

    # apply masks to fNIRSdata
    M_array, M_elements = xrutils.apply_mask(fNIRSdata, M, 'nan', 'none')
    
    M = M.unstack('measurement').pint.quantify()
    M_array = M_array.unstack('measurement').pint.quantify()
    
    return M, M_array

#%%
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

    M_pad = M.copy()
    M_pad['samples'].values = M_pad['samples'].values + 12 # add 12 to the indices to account for padding

    mean_SNR_threshold = np.zeros(len(fNIRSdata.measurement))
        
    for m,measurement in enumerate(fNIRSdata.measurement):
     
        channel = fNIRSdata_pad.sel(measurement=measurement)
        channel_M = M_pad.sel(measurement=measurement)
        
        sig = channel.copy()
        sig.values = np.ones(channel.shape) # initialize array for baseline shift detection
        sig.values[np.where(channel_M.values == True)[0]] = 0 # set indices where motion is detected to 0
        
        # find locations where signal goes from clean -> motion -> clean
        temp = sig.diff('time')
        
        # create lists that indicate the start and finish of each motion segment
        lstMs = np.where(temp==-1)[0]
        lstMf = np.where(temp==1)[0]
        
        if len(lstMs) == 0:
            lstMs = [0]
        if len(lstMf) == 0:
            lstMf = len(sig)-1
        if lstMs[0] > lstMf[0]:
            lstMs = np.insert(lstMs, 0, 0)
        if lstMs[-1] > lstMf[-1]:
            lstMf.append(len(sig)-1)
        
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
                if sig.values[kk+1] == 0:
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
            mean_SNR_threshold[m] = np.mean(SNR_threshold[1:len(SNR_threshold)-1]) # get the mean signal to noise ratio for all segments longer than 3 seconds
        
        tinc = np.ones(nT_pad)
        # loop over each motion segment 
        for kk in range(len(lstMs)):
            
            if motion_kind[kk] > ssttdd_thresh: # if the difference between the start and finish of a segment is greater than the threshold, set the segment to 0
                tinc[lstMs[kk]:lstMf[kk]] = 0
                
            if ((lstMf[kk] - lstMs[kk]) > 0.1*fs) and ((lstMf[kk] - lstMs[kk]) < 0.49999*fs): # if the segment is longer than 0.1 seconds and shorter than 0.5 seconds, set the segment to 0
                tinc[lstMs[kk]:lstMf[kk]] = 0

            if lstMf[kk] - lstMs[kk] > fs: # if the segment is longer than the sampling frequency, set the segment to 0
                tinc[lstMs[kk]:lstMf[kk]] = 0

        tInc[:,m] = tinc

    # get SNR for all channels if it has not already been calculated 
    for ww, measurement in enumerate(fNIRSdata['measurement']):
        if np.isnan(mean_SNR_threshold[ww]) or mean_SNR_threshold[ww] == 0:
            mean_SNR_threshold[ww] = abs(fNIRSdata.sel(measurement=measurement).mean('time')/fNIRSdata.sel(measurement=measurement).std('time'))

    lent = mean_SNR_threshold.shape[0]//2-1 # get the number of channels 

    SNRvalue = 3

    # make sure the same mask is applied to both channels
    for ww in range(len(mean_SNR_threshold)//2-1):
        
        if mean_SNR_threshold[ww] < SNRvalue and mean_SNR_threshold[ww+lent] < SNRvalue: # if the SNR is less than the threshold for both channels, set channels to ones
            tInc[:,ww+lent] = np.ones(len(fNIRSdata_pad))
            tInc[:,ww] = np.ones(len(fNIRSdata_pad))
        
        elif mean_SNR_threshold[ww] > SNRvalue and mean_SNR_threshold[ww+lent] < SNRvalue: # if the SNR is greater than the threshold for the first channel and less than the threshold for the second channel, set the second channel to the first channel
            tInc[:,ww+lent] = tInc[:,ww]
        
        elif mean_SNR_threshold[ww] < SNRvalue and mean_SNR_threshold[ww+lent] > SNRvalue: # if the SNR is less than the threshold for the first channel and greater than the threshold for the second channel, set the first channel to the second channel
            tInc[:,ww] = tInc[:,ww+lent]
        
    # remove the padding
    tIncCh = tInc[extend:-extend,:].pint.quantify()

    tIncall = tIncCh[:,0]

    # get the union of the masks for all channels
    for kk in range(tInc.shape[1]):
        tIncall = tIncall * tIncCh[:,kk]

    tInc = tIncall
    tIncCh = tIncCh.unstack('measurement').pint.quantify()

    return tInc, tIncCh
