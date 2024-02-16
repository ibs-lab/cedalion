import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.sigproc
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
def snr(amplitudes: cdt.NDTimeSeries):
    """Calculates channel SNR of each channel and wavelength as the ratio betwean mean and std 
    of the amplitude signal    
    """
    snr = amplitudes.mean("time") / amplitudes.std("time")

    return snr


#@cdc.validate_schemas
def prune(data: cdt.NDTimeSeries, SNRThresh: Quantity, dRange: Quantity, SDrange: Quantity):
    """Prune channels from the measurement list if their signal is too weak, too strong, or their 
    standard deviation is too great. This function updates MeasListAct based on whether data 'd' 
    meets these conditions as specified by'dRange' and 'SNRthresh'.

    Based on Homer3 [1] v1.80.2 "hmR_PruneChannels.m"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3

    [1] Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009).
     "HomER: a review of time-series analysis methods for near-infrared spectroscopy of the brain". 
     Appl Opt, 48(10), D280–D298. https://doi.org/10.1364/ao.48.00d280
    
    INPUTS:
    amplitues:  NDDataSetSchema, input fNIRS data array with at least time and channel dimensions and geo3D coordinates.
    SNRthresh:  Quantity, SNR threshold (unitless). 
                If mean(d)/std(d) < SNRthresh then it is excluded as an active channel
    dRange:     Quantity, in mm if mean(d) < dRange(1) or > dRange(2) then it is excluded as an active channel
    SDrange - will prune channels with a source-detector separation <SDrange(1) or > SDrange(2)

    OUTPUTS:    input NDDataSetSchema with a the Measurement List "MeasList" reduced by pruned channels 
    """

    # create list of active channels if it does not exist yet
    if not hasattr(data, "MeasList"):
        # add measurement list to dat xarray object. it countains the full list of channels
        data['MeasList'] = list(data.coords['channel'].values)
  
    # for all active channels in the MeasList check if they meet the conditions or drop them from the list

    ## find channels with bad snr
    # calculate SNR of channel amplitudes and find channels with SNR below threshold. Then remove them from the MeasList
    dat_snr = snr(data.amp) 
    drop_list = dat_snr.where(dat_snr < SNRThresh).dropna(dim="channel").channel.values   
    data = data.drop_sel(MeasList=drop_list)
    
    ##



    ##
 

    return data