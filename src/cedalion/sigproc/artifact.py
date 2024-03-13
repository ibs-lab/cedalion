import numpy as np
import pandas as pd
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import xarray as xr
import cedalion.xrutils as xrutils
from cedalion import Quantity, units
from functools import reduce


@cdc.validate_schemas
def id_motion(fNIRSdata: cdt.NDTimeSeries, t_motion: Quantity, t_mask: Quantity,
              stdev_thresh: Quantity, amp_thresh: Quantity):
    """Identify motion artifacts in an fNIRS input data array.

    If any active data channel exhibits a signal change greater than std_thresh or
    amp_thresh, then a segment of data around that time point is marked 
    as a motion artifact.

    Based on Homer3 [1] v1.80.2 "hmR_MotionArtifact.m" and "hmR_MotionArtifactByChannel"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3
    Translated to cedalion using xarray functionality by AvL, 2024

    [1] Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009).
    "HomER: a review of time-series analysis methods for near-infrared spectroscopy of
    the brain". Appl Opt, 48(10), D280–D298. https://doi.org/10.1364/ao.48.00d280


    INPUTS:
    amplitudes:     NDTimeSeries, input fNIRS data array with at least time and channel
                    dimensions. Expectation is raw or optical density data.
    t_motion:       Quantity, time in seconds for motion artifact detection. Checks for
                    signal change indicative of a motion artifact over time range t_motion.
    t_mask:         Quantity, time in seconds to mask around motion artifacts. Mark data
                    over +/- tMask seconds around the identified motion artifact as a
                    motion artifact.
    stdev_thresh:   Quantity, threshold for std deviation of signal change. If the
                    signal d for any given active channel changes by more that
                    stdev_thresh * stdev(d) over the time interval tMotion, then this
                    time point is marked as a motion artifact. The standard deviation
                    is determined for each channel independently. Typical value ranges
                    from 5 to 20. Use a value of 100 or greater if you wish for this
                    condition to not find motion artifacts.
    amp_thresh:     Quantity, threshold for amplitude of signal change. If the signal d
                    for any given active channel changes by more that amp_thresh over
                    the time interval tMotion, then this time point is marked as a
                    motion artifact. Typical value ranges from 0.01 to 0.3 for optical
                    density units. Use a value greater than 100 if you wish for this
                    condition to not find motion artifacts.

    OUTPUS:
    ma_mask:   is an xarray that has at least the dimensions channel and time,
                    units are boolean. At each time point, "false" indicates data
                    included and "true" indicates motion artifacts.

    DEFAULT PARAMETERS:
    t_motion: 0.5
    t_mask: 1.0
    stdev_thresh: 50.0
    amp_thresh: 5.0
    global_flag: False
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
    for ii in range(1, t_motion_samples+1):
        # Shift the data by X samples to the left in the 'time' dimension
        shifted_data = fNIRSdata.shift(time=-ii,fill_value=0)
        # zero padding of original data where shifted data is shorter
        fNIRSdata0 =fNIRSdata.copy()
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
    art_ind = art_ind.where((max_diff>mc_thresh) | (max_diff>amp_thresh), False)

    # apply mask to data to mask points surrounding motion artifacts
    # convolution kernel for masking
    ckernel = np.ones(2*t_mask_samples + 1)

    # create mask: convolve the artifact indicators with the kernel to mask surrounding samples.
    # > 0 makes the result of the convolution again a boolean array
    ma_mask = xrutils.convolve(art_ind.astype(int), ckernel, 'time') > 0

    return ma_mask



@cdc.validate_schemas
def id_motion_refine(ma_mask: cdt.NDTimeSeries, operator: str):
    """Refines motion artifact mask to simplify and quantify motion artifacts.

    INPUTS:
    ma_mask:        motion artifact mask as generated by id_motion().
                    boolean xarray array with at least the dimensions channel and time
    operator:      string, operation to apply to the mask. Available operators:
        "by_channel":  collapses the mask along the amplitude/wavelenght/chromophore dimension
                    to provide a single motion artifact mask per channel (default) over time
        "all":      collapses the mask along all dimensions to provide a single motion
                    artifact marker for all channels over time
                    i.e. an artifact detected in any channel masks all channels.

    OUTPUS:
    ma_mask_new:    updated motion artifact mask, collapsed according to operator
    ma_info:        pandas dataframe that contains 1) channels with motion artifacts,
                    2) # of artifacts detected per channel 3)  fraction of artifacts/total time

    """

    # combine artifact masks (if multiple masks are provided).
    # Will result in a single mask containing all motion indicators
    mask = reduce(lambda x, y: x | y, ma_mask)


    # collapse motion artifact masks according to operator instruction
    if operator.lower() == "by_channel":
        # find whether "wavelength" or "concentration" exists as a dimension in ma_mask and collapse, otherwise assert an error
        if "wavelength" in ma_mask.dims:
            ma_mask_new = ma_mask.any(dim="wavelength")
        elif "concentration" in ma_mask.dims:
            ma_mask_new = ma_mask.any(dim="concentration")
        else:
            raise ValueError("ma_mask must have either 'wavelength' or 'concentration' as a dimension")

        ## --- extract motion artifact info --- ##
        # extract channels that had motion artifacts
        ch_wma = ma_mask_new.any(dim="time")
        ch_labels = ch_wma.where(ch_wma, drop=True).channel.values
        # for every channel in ch_label calculate the fraction of time points that are true over the total number of time points
        ch_frac = (ma_mask_new.sel(channel=ch_labels).sum(dim='time')/ma_mask_new.sizes['time']).to_series()
        # Count number of motion artifacts (transitions in the mask) for each channel
        transitions = ma_mask_new.astype(int).diff(dim='time')==1
        transitions_ct = transitions.sum(dim='time').to_series()
        # Store motion artifact info in a pandas dataframe
        ma_info = pd.DataFrame({
            'ma_fraction': ch_frac,
            'ma_count': transitions_ct
            }).reset_index()

    # collapse mask along all dimensions
    elif operator.lower() == "all":
        dims2collapse = [dim for dim in ma_mask.dims if dim != "time"]
        ma_mask_new = ma_mask.any(dim=dims2collapse)

        ## --- extract motion artifact info --- ##
        global_frac = (mask.sum(dim='time')/mask.sizes['time']).values
        # Count number of motion artifacts (transitions in the mask) for each channel
        transitions = mask.astype(int).diff(dim='time')==1
        transitions_ct = transitions.sum(dim='time').values
        # Store motion artifact info in a pandas dataframe
        ma_info = pd.DataFrame({
            'channel': ['all channels combined'],
            'ma_fraction': [global_frac],
            'ma_count': [transitions_ct]
            })

    else:
            raise ValueError(f"unsupported operator '{operator}'")

    return ma_mask_new, ma_info
