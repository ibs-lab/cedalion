import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity, units
from typing import List
from functools import reduce

from .frequency import freq_filter, sampling_rate


@cdc.validate_schemas
def sci(amplitudes: cdt.NDTimeSeries, window_length: Quantity, sci_thresh: Quantity):
    """Calculate the scalp-coupling index based on [1].

    INPUTS:
    amplitues:  NDTimeSeries, input fNIRS data xarray with time and channel dimensions.
    sci_thresh: Quantity, sci threshold (unitless).
                If mean(d)/std(d) < SNRthresh then it is excluded as an active channel
    OUTPUTS:
    sci:        xarray with coords from input NDTimeseries containing the scalp-coupling index
    sci_mask:   boolean mask xarray with coords from sci, true where sci_thresh is met


    [1] L. Pollonini, C. Olds, H. Abaya, H. Bortfeld, M. S. Beauchamp, and
        J. S. Oghalai, “Auditory cortex activation to natural speech and
        simulated cochlear implant speech measured with functional near-infrared
        spectroscopy,” Hearing Research, vol. 309, pp. 84–93, Mar. 2014, doi:
        10.1016/j.heares.2013.11.007.
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
    sci /= windows.std("window").prod("wavelength")

    # create sci mask and update accoording to sci_thresh
    sci_mask = xrutils.mask(sci, True)
    sci_mask = sci_mask.where(sci > sci_thresh, False)

    return sci, sci_mask


@cdc.validate_schemas
def snr(amplitudes: cdt.NDTimeSeries, snr_thresh: Quantity):
    """Calculates channel SNR of each channel and wavelength.

    Based on Homer3 [1] v1.80.2 "hmR_PruneChannels.m"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3

    [1] Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009).
     "HomER: a review of time-series analysis methods for near-infrared spectroscopy of the brain".
     Appl Opt, 48(10), D280–D298. https://doi.org/10.1364/ao.48.00d280

    INPUTS:
    amplitues:  NDTimeSeries, input fNIRS data xarray with time and channel dimensions.
    snr_thresh:  Quantity, SNR threshold (unitless).
                If mean(d)/std(d) < SNRthresh then it is excluded as an active channel
    OUTPUTS:
    snr:        xarray with coords from input NDTimeseries containing the ratio between 
                mean and std of the amplitude signal for all channels.
    snr_mask:   boolean mask xarray with coords from snr, true where snr_threshold is met

    DEFAULT PARAMETERS:
    amp_threshs: [1e4, 1e7]
    snr_thresh: 2

    """
    # calculate SNR
    snr = amplitudes.mean("time") / amplitudes.std("time")
    # create snr mask and update accoording to snr thresholds
    snr_mask = xrutils.mask(snr, True)
    snr_mask = snr_mask.where(snr > snr_thresh, False)

    return snr, snr_mask


@cdc.validate_schemas
def mean_amp(amplitudes: cdt.NDTimeSeries, amp_threshs: Quantity):
    """Calculate and threshold channels with mean(amplitudes) < amp_threshs(0) or > amp_threshs(1).

    Based on Homer3 [1] v1.80.2 "hmR_PruneChannels.m"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3

    [1] Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009).
     "HomER: a review of time-series analysis methods for near-infrared spectroscopy of the brain".
     Appl Opt, 48(10), D280–D298. https://doi.org/10.1364/ao.48.00d280

    INPUTS:
    amplitudes:  NDTimeSeries, input fNIRS data xarray with time and channel dimensions.
    amp_threshs:  Quantity, If mean(amplitudes) < amp_threshs(0) or > amp_threshs(1)
                then it is excluded as an active channel in amp_mask
    OUTPUTS:
    mean_amp:   xarray with coords from input NDTimeseries containing the mean amplitudes
    amp_mask:   boolean mask xarray with coords from mean_amp, true where amp_threshs are met

    DEFAULT PARAMETERS:
    amp_threshs: [1e4, 1e7]
    """
    # calculate mean amplitude
    mean_amp = amplitudes.mean("time")
    # create amplitude mask and update according to amp_range thresholds
    amp_mask = xrutils.mask(mean_amp, True)
    amp_mask = amp_mask.where((mean_amp > amp_threshs[0]) & (mean_amp < amp_threshs[1]), False)

    return mean_amp, amp_mask


@cdc.validate_schemas
def sd_dist(amplitudes: cdt.NDTimeSeries, geo3D: Quantity, sd_threshs: Quantity):
    """Calculate and threshold source-detector separations with <sd_threshs(0) or > sd_threshs(1).

    Based on Homer3 [1] v1.80.2 "hmR_PruneChannels.m"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3

    [1] Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009).
     "HomER: a review of time-series analysis methods for near-infrared spectroscopy of the brain".
     Appl Opt, 48(10), D280–D298. https://doi.org/10.1364/ao.48.00d280

    INPUTS:
    amplitues:  NDTimeSeries, input fNIRS data xarray with time and channel dimensions.
    geo3D:      Quantity, 3D coordinates of the channels
    sd_threshs: Quantity, in mm, cm or m. If source-detector separation <sd_threshs(0) or > sd_threshs(1)
                then it is excluded as an active channelin sd_mask
    OUTPUTS:
    sd_dist:    xarray with coords from input NDTimeseries containing the channel distances
    sd_mask:    boolean mask xarray with coords from ch_dists, true where sd_threshs are met

    DEFAULT PARAMETERS:
    sd_threshs: [0.0, 4.5]
    """

    # calculate channel distances
    sd_dist = xrutils.norm(
        geo3D.loc[amplitudes.source] - geo3D.loc[amplitudes.detector], dim="digitized"
        ).round(3)
    # create sd_mask and update according to sd_thresh thresholds
    sd_mask = xrutils.mask(sd_dist, True)
    sd_mask = sd_mask.where((sd_dist > sd_threshs[0]) & (sd_dist < sd_threshs[1]), False)

    return sd_dist, sd_mask



@cdc.validate_schemas
def prune_ch(amplitudes: cdt.NDTimeSeries, masks: List[cdt.NDTimeSeries], operator: str):
    """Prune channels from the the input data array using quality masks.

    INPUTS:
    amplitudes:     NDTimeSeries, input fNIRS data xarray with time and channel dimensions.
    masks:  list of boolean mask xarrays, with coordinates that are a subset of amplitudes
    operator:       string, operators for combination of masks before pruning data_array
        "all":          = logical AND, keeps a channel only if it is good across all masks/metrics
        "any":          = logical OR, keeps channel if it is good in any mask/metric

    OUTPUTS:
    amplitudes_pruned:   input data with channels pruned (dropped) according to quality masks
    prune_list:          list of pruned channels
    """

    # check if all dimensions in the all the masks are also existing in data_array
    for mask in masks:
        if not all(dim in amplitudes.dims for dim in mask.dims):
            raise ValueError("mask dimensions must be a subset of data_array dimensions")

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
    amplitudes, prune_list = xrutils.apply_mask(amplitudes, mask, "drop", dim_collapse="channel")


    return amplitudes, prune_list
