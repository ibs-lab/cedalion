#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 10:32:38 2024

@author: lauracarlton
"""

import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity, units

import numpy as np
import xarray as xr
import pint
import matplotlib.pyplot as plt
import scipy.signal as signal
import os.path
from tqdm import tqdm 

xr.set_options(display_expand_data=False)

@cdc.validate_schemas
def tInc_baselineshift_Ch_nirs(fNIRSdata: cdt.NDTimeSeries):
    
    """Identify baseline shifts in an fNIRS timeseries data array
    
    If data channel has baseline shift, then timepoints surrounding shift are marked as a baseline shift
    
    Based on Homer3 [1] v1.80.2 "hmrR_tInc_baselineshift_Ch_Nirs.m"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3
    
    INPUTS:
        fNIRSdata: NDTimeSeries, input fNIRS data array with at least time and channel dimensions. Data should be in optical density units
    
    OUTPUTS:
        tInc: xarray that has at least dimensions channels and time, units are boolean. Each time point is marked either True to indicuate data
        is included and False to indicate baseline shift
    
    """




    fs =  fNIRSdata.cd.sampling_rate

    t = fNIRSdata.time
    extend = int(np.round(12*fs))
    
    fNIRSdata = fNIRSdata.stack(measurement = ['channel', 'wavelength']).sortby('wavelength')
    SNR_thre = np.zeros(fNIRSdata.shape[1])
    
    fNIRSdata_padded = fNIRSdata.pad(time = extend, mode='edge')
    
    tInc = xrutils.mask(fNIRSdata_padded, True)
    
    M = detect_outliers(fNIRSdata, t_motion=1)






@cdc.validate_schemas
def detect_outliers(fNIRSdata : cdt.NDTimeSeries, t_motion: Quantity):
    
    T_MOTION = 1
    fs =  fNIRSdata.cd.sampling_rate

    window_size = int(np.round(fs*T_MOTION))
    
    M = xrutils.mask(fNIRSdata, True)
    M_std = xrutils.mask(fNIRSdata, True)
    M_grad = xrutils.mask(fNIRSdata, True)
    
    IQR_GRAD_THRESH = 1.5
    IQR_STD_THRESH = 2
    
    # want to loop through all channels
    for ii, measurement in enumerate(fNIRSdata['measurement']):
    # select one channel for now instead of looping through all of them 
    
    
        channel = fNIRSdata.sel(measurement=measurement)
        M_std_channel = M_std.sel(measurement=measurement)
        M_grad_channel = M_grad.sel(measurement=measurement)
        
        # filter data between 0, 2 
        channel = channel.cd.freq_filter(0, 2, butter_order=4)
        
        # filter data between 0 0.5
        channel_lpf = channel.cd.freq_filter(0, 0.5, butter_order=4)
        
        # detect outliers in std variations of the signal 
        windows = channel_lpf.rolling(time=window_size).construct("window", stride=1)
        sigSTD = windows.std("window")
        
        quants_std = sigSTD.quantile([0.25,0.5,0.75])
        IQR_std = quants_std[2] - quants_std[0]
        high_std = quants_std[2] + IQR_std*IQR_STD_THRESH
        low_std = quants_std[0] - IQR_std*IQR_STD_THRESH
        
        # detect outliers in gradient of the signal
        grad = channel.copy()
        grad.values = signal.convolve(channel.values, [-1,0,1], mode='same')
        
        quants_grad = grad.quantile([0.25,0.5,0.75])
        IQR_grad = quants_grad[2] - quants_grad[0]
        high_grad = quants_grad[2] + IQR_grad*IQR_GRAD_THRESH
        low_grad = quants_grad[0] - IQR_grad*IQR_GRAD_THRESH
        
        # union of all outliers
        condition = (sigSTD[:-window_size] > high_std) | (sigSTD[:-window_size] < low_std)
        M_std_channel[condition] = False
    
        condition = (grad > high_grad) | (grad < low_grad)
        M_grad_channel[condition] = False
        
        if len(M_std) != 0:
            M_std_channel = np.round(window_size/2) + M_std_channel
            
        
        # combine grad and std outliers -> M
        if len(M_std) != 0 and len(M_grad) != 0:
            set1 = set(M_std)
            set2 = set(M_grad)
            M = np.array(list(set1.symmetric_difference(set2))).astype(int)
        
        elif len(M_std) != 0:
            M = M_std.astype(int)
        elif len(M_grad) != 0:
            M = M_grad.astype(int)
    
    return M
    
    
    
    

def detect_baselineshift(fNIRSdata:cdt.NDTimeSeries, M:cdt.NDTimeSeries):
    
        #% BASELINE SHIFT MOTION DETECTION
        if len(M) != 0:  # only need to do it if outliers were detected 
            M = M+extend
            sig = np.ones(len(s1_pad))
            sig[M] = 0
            
            # find locations
            temp = np.diff(sig)
    
            condition = np.where(temp == 1)[0]
            meanpL = s1_pad[condition]
            
            condition = np.where(temp == -1)[0]
            meanpH = s1_pad[condition]
            motion_kind = abs(meanpH.values-meanpL.values)
            
    
        #% FIND BASELINE SHIFTS BY COMPARING MOTION AMPLTIDUES W HR AMPLITUDE 
        
            # find the baseline shifts by comparing motion amplitudes with heart rate amplitude
            snoise = s1_pad.copy()
    
            s1_pad = s1_pad.cd.freq_filter(0, 2, butter_order=4)
            
            tt=0
            a=[]
            b=[]
            sigtemp=[]
            signoise=[]
            
            
            for ii in np.arange(len(s1_pad)-1):
            
                if sig[ii] == 1:
                    a.append(s1_pad.values[ii])
                    b.append(snoise.values[ii])
                    if sig[ii+1] == 0:
                        sigtemp.append(a)
                        signoise.append(b)
                        a=[]
                        b=[]
                        tt=tt+1
            
            
            Nthresh = int(np.round(0.5*fs))
            ssttdd = []
            
            tempo2=[]
            for ii in range(tt):
                tempo = sigtemp[ii]
                if len(tempo)>Nthresh:
                    for jj in range(len(tempo)-Nthresh):
                        tempo2.append((abs(tempo[jj+Nthresh]-tempo[jj])))
                        
                ssttdd.extend(tempo2) 
                tempo2 = []
            
            
            ssttdd = np.array(ssttdd)
        
            ssttdd_thresh = np.quantile(ssttdd,0.5)
            pointS = np.where(temp<0)[0]
            pointE = np.where(temp>0)[0]
            countnoise = 0
            SNR_Thresh = []
            for kk in range(len(signoise)):
                if len(signoise[kk]) > 3*fs:
                    countnoise =countnoise+1
                    dmean = np.mean(signoise[kk]);
                    dstd = np.std(signoise[kk]);
                    SNR_Thresh.append(abs(dmean)/(dstd+1e-20))
                
    
            SNR_thre[ww] = np.mean(SNR_Thresh[1:len(SNR_Thresh)-1])
                
            
            sig2 = np.ones(len(s1_pad))
            for ii in range(len(pointS)):
                
                if motion_kind[ii] > ssttdd_thresh:
                    sig2[pointS[ii]:pointE[ii]] = 0
                
                
                if (pointE[ii] - pointS[ii]) > 0.1*fs and (pointE[ii] - pointS[ii]) < 0.49999*fs:
                    sig2[pointS[ii]:pointE[ii]] = 0
                
                if pointE[ii] - pointS[ii] > fs:
                    sig2[pointS[ii]:pointE[ii]] = 0
        
            tInc[:,ww] = sig2
        else:
            tInc[:,ww] =np.ones(len(t))
        ww=ww+1
        ## end loop for that channel
        
        
    
    
