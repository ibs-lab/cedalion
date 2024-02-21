import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.sigproc
import cedalion.xrutils as xrutils
from cedalion import Quantity, units

from .frequency import freq_filter, sampling_rate


@cdc.validate_schemas
def sci(amplitudes: cdt.NDTimeSeries, window_length: Quantity):
    """Calculate the scalp-coupling index.

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

    return sci


@cdc.validate_schemas
def snr_range(amplitudes: cdt.NDTimeSeries, snr_thresh: Quantity):
    """Calculates channel SNR of each channel and wavelength.

    INPUTS:
    amplitues:  NDTimeSeries, input fNIRS data xarray with time and channel dimensions.
    snr_thresh:  Quantity, SNR threshold (unitless).
                If mean(d)/std(d) < SNRthresh then it is excluded as an active channel
    OUTPUTS:
    snr:        ratio betwean mean and std of the amplitude signal for all channels.
    MeasList:   list of active channels that meet the conditions

    """
    # calculate SNR
    snr = amplitudes.mean("time") / amplitudes.std("time")
    # create MeasList and threshold it
    meas_list = list(amplitudes.coords["channel"].values)
    # for all active channels in the MeasList check if they meet the conditions 
    # or drop them from the list
    drop_list = snr.where(snr < snr_thresh).dropna(dim="channel").channel.values
    meas_list = [channel for channel in meas_list if channel not in drop_list]

    return snr, meas_list, drop_list


@cdc.validate_schemas
def amp_range(amplitudes: cdt.NDTimeSeries, amp_threshs: Quantity):
    """Identify and drop channels mean(amplitudes) < dRange(1) or > dRange(2).

    INPUTS:
    amplitues:  NDTimeSeries, input fNIRS data xarray with time and channel dimensions.
    amp_threshs:  Quantity, . If mean(d) < dRange(1) or > dRange(2)
                then it is excluded as an active channel
    OUTPUTS:
    MeasList:   list of active channels that meet the conditions
    """

    # create MeasList and threshold it
    meas_list = list(amplitudes.coords["channel"].values)
    # for all active channels in the MeasList check if they meet the conditions
    # or drop them from the list
    drop_list = amplitudes.mean("time").where(
        (amplitudes.mean("time") < amp_threshs[0]) |
        (amplitudes.mean("time") > amp_threshs[1])
        ).dropna(dim="channel").channel.values
    meas_list = [channel for channel in meas_list if channel not in drop_list]

    return meas_list, drop_list


@cdc.validate_schemas
def sd_range(amplitudes: cdt.NDTimeSeries, geo3D: Quantity, sd_threshs: Quantity):
    """Identify and drop channels with a source-detector separation <sd_threshs(0) or > sd_threshs(1).

    INPUTS:
    amplitues:  NDTimeSeries, input fNIRS data xarray with time and channel dimensions.
    geo3D:      Quantity, 3D coordinates of the channels
    sd_threshs: Quantity, in mm, cm or m. If source-detector separation <sd_threshs(0) or > sd_threshs(1)
                then it is excluded as an active channel
    OUTPUTS:
    ch_dist:    channel distances
    MeasList:   list of active channels that meet the conditions
    """

    # calculate channel distances
    ch_dist = xrutils.norm(
        geo3D.loc[amplitudes.source] - geo3D.loc[amplitudes.detector], dim="digitized"
        ).round(3)
    # create MeasList and threshold it
    meas_list = list(amplitudes.coords["channel"].values)
    # for all active channels in the MeasList check if they meet the conditions 
    # or drop them from the list
    drop_list = ch_dist.where((ch_dist < sd_threshs[0]) | (ch_dist > sd_threshs[1])
                              ).dropna(dim="channel").channel.values
    meas_list = [channel for channel in meas_list if channel not in drop_list]

    return ch_dist, meas_list,drop_list



@cdc.validate_schemas
def prune(data: cdt.NDTimeSeries, geo3D: Quantity, snr_thresh: Quantity,
          amp_threshs: Quantity, sd_threshs: Quantity):
    """Prune channels from the measurement list.

    Prunging criteria are signal strength, standard deviation (SNR), or channel distances.
    TODO: Include SCI/PSP etc.

    Based on Homer3 [1] v1.80.2 "hmR_PruneChannels.m"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3

    [1] Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009).
     "HomER: a review of time-series analysis methods for near-infrared spectroscopy of the brain".
     Appl Opt, 48(10), D280–D298. https://doi.org/10.1364/ao.48.00d280

    INPUTS:
    data:       NDTimeSeries, input fNIRS data xarray with time and channel dimensions.
    geo3D:      Quantity, 3D coordinates of the channels
    snr_thresh:  Quantity, SNR threshold (unitless).
                If mean(d)/std(d) < SNRthresh then it is excluded as an active channel
    amp_threshs:  Quantity, . If mean(d) < dRange(1) or > dRange(2)
                then it is excluded as an active channel
    sd_threshs: Quantity, in cm. If source-detector separation <SDrange(1) or > SDrange(2)
                then it is excluded as an active channel
    OUTPUTS:
    meas_list:   list of active channels that meet the conditions

    DEFAULT PARAMETERS:
    amp_threshs: [1e4, 1e7]
    snr_thresh: 2
    sd_threshs: [0.0, 4.5]
    """

    # create MeasList with all channels active
    meas_list = list(data.coords["channel"].values)

    # SNR thresholding
    snr, meas_list_snr, drop_list_snr = snr_range(data, snr_thresh)
    # keep only the channels in MeasList that are also in MeasList_snr
    meas_list = [channel for channel in meas_list if channel in meas_list_snr]

    # Source Detector Separation thresholding
    ch_dist, meas_list_sd, drop_list_sd = sd_range(data, geo3D, sd_threshs)
    # keep only the channels in MeasList that are also in MeasList_sd
    meas_list = [channel for channel in meas_list if channel in meas_list_sd]

    # Amplitude thresholding
    meas_list_amp, drop_list_amp = amp_range(data, amp_threshs)
    # keep only the channels in MeasList that are also in MeasList_amp
    meas_list = [channel for channel in meas_list if channel in meas_list_amp]

    # FIXME / TODO
    # SCI thresholding

    # drop the channels in data that are not in MeasList
    data = data.sel(channel=meas_list)
    # also return a list of all dropped channels
    drop_list = list(set(drop_list_snr) | set(drop_list_sd) | set(drop_list_amp))

    return data, drop_list
