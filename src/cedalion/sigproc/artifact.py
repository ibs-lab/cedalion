import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity, units


@cdc.validate_schemas
def id_motion(fNIRSdata: cdt.NDTimeSeries, t_motion: Quantity, t_mask: Quantity,
              stdev_thresh: Quantity, amp_thresh: Quantity, global_flag: bool
              ):
    """Identify motion artifacts in an fNIRS input data array.

    If any active data channel exhibits a signal change greater than std_thresh or
    amp_thresh, then a segment of data around that time point is marked 
    as a motion artifact.

    Based on Homer3 [1] v1.80.2 "hmR_MotionArtifact.m" and "hmR_MotionArtifactByChannel"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3

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
    global_flag:    Boolean, flag to indicate whether identified motion artefacts in each
                    channel should be masked per channel or globally across channels.

    OUTPUS:
    channel_mask:   is an xarray that has at least the dimensions channel and time,
                    units are boolean. At each time point, "true" indicates  data
                    included and "false" indicates motion artifacts.

    PARAMETERS:
    t_motion: 0.5
    t_mask: 1.0
    stdev_thresh: 50.0
    amp_thresh: 5.0
    global_flag: False
    """

    # TODO assert OD units, otherwise issue a warning

    # initialize a channel mask
    channel_mask = xrutils.mask(fNIRSdata, True)

    # calculate the "std_diff", the standard deviation of the approx 1st derivative of
    # each channel over time
    std_diff = fNIRSdata.diff(dim="time").std(dim="time")

    # calulate max_diff across channels for different time delays
    



    return fNIRSdata
