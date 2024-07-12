import numpy as np
from functools import reduce
import cedalion
from cedalion import io
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
import math
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import matplotlib.gridspec as gridspec
import os 
import matplotlib.colors as clrs
import matplotlib
logger = logging.getLogger("cedalion")

#%% GETTING METRICS
def get_data_metrics(amplitudes):
    

    snr_val = get_snr(amplitudes)
    sci_val, sci_mask = get_sci(amplitudes)
    psp_val, psp_mask = get_psp(amplitudes)
    gvtd_val = get_gvtd(amplitudes)
    
    masks = [sci_mask, psp_mask]
    
    scixpsp_mask = xr.where(sci_mask & psp_mask == False, sci_mask, psp_mask)

    perc_channel = 1 - scixpsp_mask.sum('time')/scixpsp_mask.shape[1]
    
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
    gvtd = GVTD(amplitudes)
    return gvtd

#%% PLOTTING MONTAGE METRICS

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def convert_optodePos3D_to_circular2D(pos, tranformation_matrix, norm_factor):
    pos = np.append(pos, np.ones((pos.shape[0],1)), axis=1)
    pos_sphere = np.matmul(pos,tranformation_matrix)
    pos_sphere_norm = np.sqrt(np.sum(np.square(pos_sphere), axis=1))
    pos_sphere_norm = pos_sphere_norm.reshape(-1,1)
    pos_sphere = np.divide(pos_sphere,pos_sphere_norm)
    azimuth, elevation, r = cart2sph(pos_sphere[:,0], pos_sphere[:,1], pos_sphere[:,2])
    elevation = math.pi/2-elevation;
    x, y = pol2cart(azimuth,elevation)
    x = x/norm_factor
    y = y/norm_factor
    return x, y

def get_transformation_matrix(geo3d):
    
    landmark_system_df = pd.read_excel('10-5-System_Mastoids_EGI129.xlsx') 

    landmarks = geo3d.loc[geo3d.type == io.snirf.PointType.LANDMARK]
    landmark_probe_lst =landmarks.label.values
    landmark_df_np = landmark_system_df['Label'].tolist()
    
    index_mapping = [landmark_df_np.index(item) if item in landmark_df_np else None for item in landmark_probe_lst]
   
    probe_landmark_pos3D = np.hstack([landmarks, np.ones([landmarks.shape[0],1])])
    circular_landmark_pos3D = landmark_system_df[['X','Y','Z']].loc[index_mapping]
 
    landmarks2D_theta = (landmark_system_df['Theta']*2*math.pi/360).to_numpy()
    landmarks2D_phi = ((90-landmark_system_df['Phi'])*2*math.pi/360).to_numpy()
    x,y = pol2cart(landmarks2D_theta, landmarks2D_phi)
    
    norm_factor = max(np.sqrt(np.add(np.square(x),np.square(y))))
    temp = np.linalg.inv(np.matmul(np.transpose(probe_landmark_pos3D),probe_landmark_pos3D))
    transformation_matrix = np.matmul(temp,np.matmul(np.transpose(probe_landmark_pos3D),circular_landmark_pos3D))        
    
    return transformation_matrix.values, norm_factor

def remove_short(data, metric, geo3d, threshold_col):
    
    skipped_channels = []
    skipped_detectors = []
    skipped_metrics = []
    nMeas = len(data.channel)
    for u in range(nMeas):
        
        sourceIndex =  data.source[u]
        detectorIndex =  data.detector[u]
        
        dist = xrutils.norm(geo3d.loc[data.source[u]] - geo3d.loc[data.detector[u]], dim="pos")
        if dist < 10:
                skipped_channels.append([sourceIndex, detectorIndex])
                skipped_detectors.append(detectorIndex)
                skipped_metrics.append(u)

    # if the metrics/threshold_col given include those for short channels, remove them from the array 
    if len(metric) == nMeas//2:
        metric = np.delete(metric,skipped_metrics)

    if type(threshold_col) == list:
        if len(threshold_col) == nMeas//2:
            threshold_col = np.delete(threshold_col,skipped_metrics)

    return metric, skipped_detectors, skipped_channels, threshold_col


    

def plot_metrics_on_probe(snirfObj, metric, ax, colormap=plt.cm.bwr, title='DQR', threshold_ind = None, threshold_col = None, saturation=None, vmin=0, vmax=1, savePath = None, remove_short=0):
    
    geo3d = snirfObj.geo3d
    sources = geo3d.loc[geo3d.type == io.snirf.PointType.SOURCE]
    detectors = geo3d.loc[geo3d.type == io.snirf.PointType.DETECTOR]
    
    transformation_matrix, norm_factor = get_transformation_matrix(geo3d)

    data = snirfObj.data[0]
    nMeas = len(data.channel)
    
    if remove_short == 1: # then remove any channels that are less than 10mm 
        metric, skipped_detectors, threshold_col, skipped_channels = remove_short(data, metric, geo3d, threshold_col)
    else:
        skipped_channels = []
        skipped_detectors = []
    #### scale indices #####
    sourcePos2DX , sourcePos2DY = convert_optodePos3D_to_circular2D(sources, transformation_matrix, norm_factor)
    detectorPos2DX , detectorPos2DY = convert_optodePos3D_to_circular2D(detectors, transformation_matrix, norm_factor)
    
    scale = 1.3
    sourcePos2DX = sourcePos2DX*scale
    detectorPos2DX = detectorPos2DX*scale
    sourcePos2DY = sourcePos2DY*scale
    detectorPos2DY = detectorPos2DY*scale
        
    #### plot the positions on the unit circle ####
    t = np.linspace(0, 2 * np.pi, 100)
    head_x = [math.sin(i) for i in t]
    head_y = [math.cos(i) for i in t]
        
    
    #### plot the channels according to the metrics ####
    norm = clrs.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(cmap=colormap,norm=norm)
    fontDict_src = dict(color='r', fontweight = 'bold', fontstretch= 'expanded',fontsize = 7)
    fontDict_det = dict(color='b', fontweight = 'bold', fontstretch= 'expanded',fontsize = 7)
    
    i =0
    for u in range(nMeas):
        sourceIndex =  data.source[u]
        detectorIndex =  data.detector[u]
        
        # skip the short_channels 
        if [sourceIndex, detectorIndex] in skipped_channels:
            continue
        
        
        iS = int(sourceIndex.to_numpy().tolist()[1:])
        iD = int(detectorIndex.to_numpy().tolist()[1:])
        x = [sourcePos2DX[iS-1], detectorPos2DX[iD-1]]
        y = [sourcePos2DY[iS-1], detectorPos2DY[iD-1]]
        

        try:
            assert(threshold_col == None)
        except:
            if threshold_col[i] == 1: #metric[u] < threshold: 
                linestyle = '-'
                alpha = 0.4
            else:
                linestyle = '-'
                alpha = 1
        else:
            linestyle = '-'
            alpha=1
        
        try:
            assert(saturation == None)
        except:
            if saturation[i] == 1:
                color = '0.7'
                alpha = 1
            else:
                color = colormap(norm(metric[i]))
        else:
            color = colormap(norm(metric[i]))
            
        ax.plot(x,y, color=color,linestyle=linestyle, linewidth = 2, alpha=alpha)
        # ax.text(sourcePos2DX[sourceIndex-1], sourcePos2DY[sourceIndex-1],str(sourceIndex),fontdict=fontDict_src) # bbox=dict(color = 'r',boxstyle = "round, pad=0.3", alpha=0.05))
        # ax.text(detectorPos2DX[detectorIndex-1], detectorPos2DY[detectorIndex-1], str(detectorIndex),fontdict=fontDict_det) # bbox=dict(color='b',boxstyle = "round, pad=0.3", alpha=0.05))
        i+=1
    
    ax.plot(head_x,head_y,'k')
    for u in range(len(sourcePos2DX)):
        ax.plot(sourcePos2DX[u] , sourcePos2DY[u], 'r.', markersize=8)
        
    for u in range(len(detectorPos2DX)):
        if u+1 in skipped_detectors:
            continue
        ax.plot(detectorPos2DX[u] , detectorPos2DY[u], 'b.',markersize=8)
     
    if threshold_ind != None:
        ticks = [vmin, (vmin+vmax)//2, threshold_ind, vmax]
    else:   
        ticks = [vmin, (vmin+vmax)//2, vmax]
        
    ax.plot(0, 1 , marker="^",markersize=16)
    plt.colorbar(sm,shrink =0.6, ticks=ticks)
    ax.set_title(title)
    plt.tight_layout()
    plt.axis('equal')
    plt.axis('off')
    
    if savePath is not None: 
        plt.savefig(savePath, dpi=500)
    
    pass

#%% PLOTTING TIMESERIES METRICS

def plot_timeseries(GVTD, stim, ax, savePath=None):
    
    ax.plot(GVTD.time,GVTD,'k')
    ax.set(xlabel='Time (s)',ylabel='GVTD',title='GVTD')
    ax.set_xlim([GVTD.time[0], GVTD.time[-1]])
    plot_stim_marks(stim)

    if savePath != None:
        plt.savefig(savePath+'GVTD_timecourse.png', dpi=500)
        
    pass

def plot_stim_marks(stim):
    
    #TODO match cedalion dataframe structure 
    ### add markers for events
    colours = ['b', 'r', 'g', 'y', 'c']
    
    trial_types = stim.trial_type.unique()
    
    for i,trial in enumerate(trial_types):
        tmp = stim[stim.trial_type.isin([trial])]
        nStims = len(tmp)
        for n in range(nStims):
            plt.axvline(x=tmp['onset'][n], color=colours[i], lw=1)
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
    
    metric_dict = get_data_metrics(snirfObj.data[0])
    wavelengths = snirfObj.data[0].wavelength
    
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
    plot_metrics_on_probe(snirfObj, metric_dict['SNR'].sel(wavelength = wavelengths[0]), 
                          ax1, colormap=cmap, vmin=0, vmax=25, 
                          threshold_ind=5, threshold_col = threshold_col.sel(wavelength=wavelengths[0]),
                          title='SNR: $\lambda$ = ' + str(wavelengths[0].values), remove_short=remove_short)
    
    ax2 = fig.add_subplot(gs[0,1])
    plot_metrics_on_probe(snirfObj, metric_dict['SNR'].sel(wavelength = wavelengths[1]), 
                          ax2, colormap=cmap, vmin=0, vmax=25, 
                          threshold_ind=5, threshold_col = threshold_col.sel(wavelength=wavelengths[1]),
                          title='SNR: $\lambda$ = ' + str(wavelengths[1].values), remove_short=remove_short)

    #TODO need to generate this metric 
    threshold_col = metric_dict['perc_channel_thresholded'] < 0.6
    ax3 = fig.add_subplot(gs[0,2])
    plot_metrics_on_probe(snirfObj, metric_dict['perc_channel_thresholded'], 
                          ax3, colormap=cmap, vmin=0, vmax=1, 
                          threshold_ind=0.6, threshold_col = threshold_col, 
                          title='% of time SCI x PSP above thresholds', remove_short=remove_short)

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

def generate_report_group():
    '''
    - run the report for each run of each subject
    - save the metrics that are returned
    - compile the metrics across subjects
    - save compiled report
    '''
    
    
    
    
    
    
    pass









