import numpy as np
from functools import reduce
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity, units
import xarray as xr
from .quality import snr, sci, psp, GVTD
from scipy import signal
import pint
import pandas as pd
import logging
import time
from numpy.typing import ArrayLike

logger = logging.getLogger("cedalion")

#%% GETTING METRICS
def get_data_metrics(amplitudes: cdt.NDTimeSeries):
    
    snr_val = get_snr(amplitudes)
    sci_val, sci_mask = get_sci(amplitudes)
    psp_val, psp_mask = get_psp(amplitudes)
    gvtd_val = get_gvtd(amplitudes)
    
    scixpsp_mask = sci_mask and psp_mask
    
    metric_dict = {'SNR': snr_val,
                'SCI': sci,
                'PSP': psp, 
                'SCI_mask': sci_mask,
                'PSP_mask': psp_mask,
                'SCIxPSP_mask': scixpsp_mask,
                'GVTD': gvtd_val
        }
    
    return metric_dict

def get_snr(amplitudes: cdt.NDTimeSeries, snr_thresh: float = 2.0):
    snr_val, snr_mask = snr(amplitudes, snr_thresh)
    return snr_val

def get_psp(amplitudes: cdt.NDTimeSeries, window_length: Quantity = 5*units.s, psp_thresh: float = 0.1):
    psp_val, psp_mask = psp(amplitudes, window_length, psp_thresh)
    return psp_val, psp_mask

def get_sci(amplitudes: cdt.NDTimeSeries, window_length: Quantity = 5*units.s, sci_thresh: float = 0.8):
    sci_val, sci_mask = sci(amplitudes, window_length, sci_thresh)
    return sci_val, sci_mask

def get_gvtd(amplitudes: cdt.NDTimeSeries):
    gvtd = GVTD(amplitudes)
    return gvtd

#%% PLOTTING METRICS
def plot_metrics_on_probe():
    pass

def plot_timeseries():
    pass

def plot_timeseries_all_channels():
    pass

#%% GENERATE REPORTS
def generate_report_single_run():
    pass

def generate_report_group():
    pass









