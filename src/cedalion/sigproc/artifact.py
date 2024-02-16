import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import Quantity, units


@cdc.validate_schemas
def id_motion(fNIRSdata: cdt.NDTimeSeries, 
              tMotion: Quantity, tMask: Quantity, STDEVthresh: Quantity, AMPthresh: Quantity):
    """Identify motion artifacts in an fNIRS input data array. If any active
    data channel exhibits a signal change greater than std_thresh or amp_thresh, 
    then a segment of data around that time point is marked as a motion artifact.

    Based on Homer3 [1] v1.80.2 "hmR_MotionArtifact.m"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3

    [1] Huppert, T. J., Diamond, S. G., Franceschini, M. A., & Boas, D. A. (2009).
     "HomER: a review of time-series analysis methods for near-infrared spectroscopy of the brain". 
     Appl Opt, 48(10), D280–D298. https://doi.org/10.1364/ao.48.00d280
    
    
    INPUTS:
    amplitudes:     NDTimeSeries, input fNIRS data array with at least time and channel dimensions. 
                    Expectation is data in optical density units.
    tMotion:        Quantity, time in seconds for motion artifact detection. Checks for signal change indicative
                    of a motion artifact over time range tMotion. 
    tMask:          Quantity, time in seconds to mask around motion artifacts. Mark data over +/- tMask seconds 
                    around the identified motion artifact as a motion artifact. 
    STDEVthresh:    Quantity, threshold for std deviation of signal change. If the signal d for any given active 
                    channel changes by more that stdev_thresh * stdev(d) over the time interval tMotion, then
                    this time point is marked as a motion artifact. The standard deviation is determined for 
                    each channel independently. Typical value ranges from 5 to 20. Use a value of 100 or greater 
                    if you wish for this condition to not find motion artifacts.
    AMPthresh:      Quantity, threshold for amplitude of signal change. If the signal d for any given active channel 
                    changes by more that amp_thresh over the time interval tMotion, then this time point is marked 
                    as a motion artifact. Typical value ranges from 0.01 to 0.3 for optical density units. 
                    Use a value greater than 100 if you wish for this condition to not find motion artifacts.

    OUTPUS: 
    fNIRSdata:      NDTimeSeries, input fNIRS data array with an additional dimension "artefact_mask" that is a boolean
                    array with "true" for data included and "false" to indicate motion artifacts.

    PARAMETERS:
    tMotion: 0.5
    tMask: 1.0
    STDEVthresh: 50.0
    AMPthresh: 5.0

    """

 

    return sci
