import numpy as np
from functools import reduce
import cedalion
from cedalion import io
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity, units
import cedalion.plots as plots
import xarray as xr
from .quality import snr, sci, psp, gvtd
from scipy import signal
import pint
import pandas as pd
import logging
import time
import math
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import matplotlib.gridspec as gridspec
import os 
import matplotlib.colors as clrs
import matplotlib
logger = logging.getLogger("cedalion")

#%% GETTING METRICS
def get_data_metrics(amplitudes,
                     snr_threshold,
                     psp_threshold, 
                     sci_threshold):
    

    snr_val = get_snr(amplitudes)
    sci_val, sci_mask = get_sci(amplitudes)
    psp_val, psp_mask = get_psp(amplitudes)
    gvtd_val = get_gvtd(amplitudes)
        
    scixpsp_mask =  sci_mask & psp_mask #xr.where(sci_mask & psp_mask == False, sci_mask, psp_mask)
    scixpsp_mask = ~scixpsp_mask
    perc_channel = scixpsp_mask.sum('time')/scixpsp_mask.shape[1]
    
    metric_dict = {'SNR': snr_val,
                'SCI': sci_val,
                'PSP': psp_val, 
                'SCI_mask': sci_mask,
                'PSP_mask': psp_mask,
                'SCIxPSP_mask': scixpsp_mask,
                'GVTD': gvtd_val,
                'perc_channel_thresholded': perc_channel
        }
    
    return metric_dict

def get_snr(amplitudes: cdt.NDTimeSeries, snr_thresh: float = 2.0):
    snr_val, snr_mask = snr(amplitudes, snr_thresh)
    return snr_val

def get_psp(amplitudes: cdt.NDTimeSeries, window_length: Quantity = 5*units.s, psp_thresh: float = 0.1):
    psp_val, psp_mask = psp(amplitudes, window_length, psp_thresh)
    return psp_val, psp_mask

def get_sci(amplitudes: cdt.NDTimeSeries, window_length: Quantity = 5*units.s, sci_thresh: float = 0.7):
    sci_val, sci_mask = sci(amplitudes, window_length, sci_thresh)
    return sci_val, sci_mask

def get_gvtd(amplitudes: cdt.NDTimeSeries):
    gvtd_val = gvtd(amplitudes)
    return gvtd_val



#%% PLOTTING TIMESERIES METRICS

def plot_timeseries(GVTD, stim, ax, savePath=None):
    
    ax.plot(GVTD.time,GVTD,'k')
    ax.set(xlabel='Time (s)',ylabel='GVTD',title='GVTD')
    ax.set_xlim([GVTD.time[0], GVTD.time[-1]])
    if not stim.empty:
        plot_stim_marks(stim)

    if savePath != None:
        plt.savefig(savePath+'GVTD_timecourse.png', dpi=500)
        
    pass

def plot_stim_marks(stim):
    ### add markers for events
    colours = ['b', 'r', 'g', 'y', 'c']
    
    trial_types = stim['trial_type'].unique()
    
    colour_dict = {trial : colours[i] for i, trial in enumerate(trial_types)}
    
    for idx, onset in enumerate(stim['onset']):
        # Use the index to get the corresponding element in the 'duration' column
        stim_type = stim.at[idx, 'trial_type']
        plt.axvline(x=onset, color=colour_dict[stim_type], linestyle='--' )

    pass

def plot_timeseries_all_channels(scixpsp_mask, stim, ax, title, savePath = None):
    
    colors=["black","white"]
    cmap = clrs.ListedColormap(colors)
    
    c = ax.pcolor(scixpsp_mask.time,np.arange(len(scixpsp_mask.channel)), scixpsp_mask, cmap=cmap)
    # ax.set_xlim([0,scixpsp_mask.time[-1]])
    ax.set_yticks([])
    ax.set_ylabel('Channels')
    plt.colorbar(c, location='bottom', ticks = [0,1], shrink = 0.1, pad = 0.2)
    ax.set_title(title)
    
    plot_stim_marks(stim)

    plt.tight_layout()
    if savePath != None: 
        plt.savefig(savePath+'scixpsp_thresholded.png', dpi=500)
    return

#%% GENERATE REPORTS
def generate_report_single_run(snirfObj, title=None, savePath=None):
    '''
    - get the metrics
    - setup the figure
    - plot the montages
    - plot the timeseries
    - save the figure
    - return the metrics
    '''    
    
    metric_dict = get_data_metrics(snirfObj['amp'])
    wavelengths = snirfObj.wavelengths
    
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(20,10))
    if title == None:
        fig.suptitle('Signal Quality Evaluation')
    else:
        fig.suptitle(title)
    
    gs = gridspec.GridSpec(3, 3, height_ratios=[8,2,1],figure=fig)
    
    threshold_col = metric_dict['SNR'] < 5
    colors=["red","yellow","green"]
    nodes = [0.0, 0.5, 1.0]
    cmap = clrs.LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))        

    ax1 = fig.add_subplot(gs[0,0])
    plots.scalp_plot(snirfObj, metric_dict['SNR'].sel(wavelength = wavelengths[0]), 
                          ax1, colormap=cmap, vmin=0, vmax=25, 
                          threshold_ind=5, threshold_col = threshold_col.sel(wavelength=wavelengths[0]),
                          title='SNR: $\lambda$ = ' + str(wavelengths[0]), remove_short=True)
    
    ax2 = fig.add_subplot(gs[0,1])
    plots.scalp_plot(snirfObj, metric_dict['SNR'].sel(wavelength = wavelengths[1]), 
                          ax2, colormap=cmap, vmin=0, vmax=25, 
                          threshold_ind=5, threshold_col = threshold_col.sel(wavelength=wavelengths[1]),
                          title='SNR: $\lambda$ = ' + str(wavelengths[1]), remove_short=True)

    #TODO need to generate this metric 
    threshold_col = metric_dict['perc_channel_thresholded'] < 0.6
    ax3 = fig.add_subplot(gs[0,2])
    plots.scalp_plot(snirfObj, metric_dict['perc_channel_thresholded'], 
                          ax3, colormap=cmap, vmin=0, vmax=1, 
                          threshold_ind=0.6, threshold_col = threshold_col, 
                          title='% of time SCI x PSP above thresholds', remove_short=True)

    ax4 = fig.add_subplot(gs[1,:])
    plot_timeseries_all_channels(metric_dict['SCIxPSP_mask'], snirfObj.stim, 
                                 ax4, title = 'SCI x PSP')   
    
    ax5 = fig.add_subplot(gs[2,:], sharex=ax4)
    plot_timeseries(metric_dict['GVTD'], snirfObj.stim, ax5)  
    
    plt.tight_layout()
    if savePath != None:
         dirname = os.path.dirname(savePath)
         if not os.path.exists(dirname):
             os.makedirs(dirname)
         plt.savefig(savePath)

    return metric_dict

# def generate_report_group():
#     '''
#     - run the report for each run of each subject
#     - save the metrics that are returned
#     - compile the metrics across subjects
#     - save compiled report
#     '''
    
    
    
    
    
    
    
    
#     pass









