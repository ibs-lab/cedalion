"""Signal quality metrics and channel pruning."""

from functools import reduce
from typing import List

import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity, units
from cedalion.typing import NDTimeSeries

from .frequency import freq_filter, sampling_rate


@cdc.validate_schemas
def sci(amplitudes: NDTimeSeries, window_length: Quantity, sci_thresh: float):
    """Calculate the scalp-coupling index based on :cite:t:`Pollonini2014`.

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
    sci /= windows.std("window").prod("wavelength")

    # create sci mask and update accoording to sci_thresh
    sci_mask = xrutils.mask(sci, True)
    sci_mask = sci_mask.where(sci > sci_thresh, False)

    return sci, sci_mask


@cdc.validate_schemas
def snr(amplitudes: cdt.NDTimeSeries, snr_thresh: float = 2.0):
    r"""Calculates signal-to-noise ratio for each channel and other dimension.

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
    snr_mask = xrutils.mask(snr, True)
    snr_mask = snr_mask.where(snr > snr_thresh, False)

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
    amp_mask = xrutils.mask(mean_amp, True)
    amp_mask = amp_mask.where(
        (mean_amp > amp_range[0]) & (mean_amp < amp_range[1]), False
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
        geo3D.loc[amplitudes.source] - geo3D.loc[amplitudes.detector], dim="digitized"
    ).round(3)
    # create sd_mask and update according to sd_thresh thresholds
    sd_mask = xrutils.mask(sd_dist, True)
    sd_mask = sd_mask.where((sd_dist > sd_range[0]) & (sd_dist < sd_range[1]), False)

    return sd_dist, sd_mask


@cdc.validate_schemas
def prune_ch(
    amplitudes: cdt.NDTimeSeries, masks: List[cdt.NDTimeSeries], operator: str
):
    """Prune channels from the the input data array using quality masks.

    Args:
        amplitudes (:class:`NDTimeSeries`): input time series
        masks (:class:`List[NDTimeSeries]`) : list of boolean masks with coordinates
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
