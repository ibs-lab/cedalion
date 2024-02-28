#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:29:01 2024

@author: lauracarlton
"""

import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils
import numpy as np
import xarray as xr
import pint
import matplotlib.pyplot as plt
import scipy.signal as signal
import os.path
from tqdm import tqdm 
from scipy.interpolate import UnivariateSpline, CubicSpline

xr.set_options(display_expand_data=False)

#%% LOAD THE DATA 
p = 0.99

DATADIR = "/Users/lauracarlton/Library/CloudStorage/GoogleDrive-lcarlton@bu.edu/My Drive/fNIRS/Data/OLD_FT/BIDS-NIRS-Tapping/"
elements = cedalion.io.read_snirf(DATADIR + "sub-01/nirs/sub-01_task-tapping_nirs.snirf")

amp = elements[0].data[0]
dod1 =  -np.log( amp / amp.mean("time"))
dod = dod1.stack(measurement = ['channel', 'wavelength']).sortby('wavelength')

t = dod.time

fs =  dod.cd.sampling_rate

#TODO
# needs tInc from baselineshift function 

dtSHORT = 0.3
dtLONG = 3 

ml = elements[0].measurement_lists[0]

dodSpline = dod.copy()

# tInc = np.ones(dod.shape)

ww = 0
for measurement in tqdm(dod['measurement']):
    
    tInc_chan = tIncCh[:,ww]
    lstMA = np.where(tInc_chan==0)[0]
    
    # s1 = dod.sel(measurement=measurement)
    if len(lstMA) != 0:
        
        lstMs = np.where(np.diff(tInc_chan) == -1)[0] # starting indices of motion artefacts
        lstMf = np.where(np.diff(tInc_chan) == 1)[0] # ending indices of motion artefacts

    # Case where there's a single MA segment, that either starts at the
       # beginning or ends at the end of the total time duration
        if lstMf is None or len(lstMf) == 0:
            lstMf = tInc_chan.shape
        
        if lstMs is None or len(lstMs) == 0:
            lstMs = 0
        
        # if any motion artefact segment either starts at the beginning or ends at the end of the total time
        if lstMs[0] > lstMf[0]:
            temp = [0]
            temp.extend(lstMs)
            lstMs = temp
        
        if lstMs[-1] > lstMf[-1]:
            lstMf.extend(tInc_chan.shape)
        
        #TODO - handle lists with zero length
        
        lstMl = lstMf - lstMs # length of motion artefact segments
        
        nbMA = len(lstMl) # number of motion artefact segments
        
        # do spline interpolation on each motion artefact segment - only including channels in active meas list
        for ii in range(nbMA):
            
            lst = np.arange(lstMs[ii], lstMf[ii])
            
            
            if len(lst)>3:
                splInterp = CubicSpline(t[lst], dod[lst,ww])#UnivariateSpline(t[lst], dod[lst,ww], k=3, s=p)
                # a = splInterp(lst)
                
                dodSpline.values[lst,ww] = dod[lst,ww] - splInterp(t[lst])
    

    
    #% for the first MA segment: shift to previous no MA segment if it exits
    lst = np.arange(lstMs[0], lstMf[0])
    SegCurrLength = lstMl[0]
    
    if SegCurrLength < dtSHORT * fs:
        windCurr = SegCurrLength
    elif SegCurrLength < dtLONG * fs:
        windCurr = int(np.floor(dtSHORT * fs))
    else:
        windCurr = int(np.floor(SegCurrLength / 10))
    
    if lstMs[0] > 1:
        SegPrevLength = len(np.arange(1, lstMs[0]))
        
        if SegPrevLength < dtSHORT * fs:
            windPrev = SegPrevLength
        elif SegPrevLength < dtLONG * fs:
            windPrev = int(np.floor(dtSHORT * fs))
        else:
            windPrev = int(np.floor(SegPrevLength / 10))
        
        meanPrev = np.mean(dodSpline[slice(lst[0] - windPrev, lst[0] - 1), ww])
        meanCurr = np.mean(dodSpline[slice(lst[0], lst[0] + windCurr - 1), ww])
        
        dodSpline[lst, ww] = dodSpline[lst, ww] - meanCurr + meanPrev
    
    else:
        if len(lstMs) > 1:
            SegNextLength = len(np.arange(lstMf[0], lstMs[1]))
        else:
            SegNextLength = len(np.arange(lstMf[0], tInc_chan.shape[0]))
    
        if SegNextLength < dtSHORT * fs:
            windNext = SegNextLength
        elif SegNextLength < dtLONG * fs:
            windNext = int(np.floor(dtSHORT * fs))
        else:
            windNext = int(np.floor(SegNextLength / 10))
        
        meanCurr = np.mean(dodSpline[slice(lst[-1] - windCurr, lst[-1] - 1), ww])
        meanNext = np.mean(dodSpline[slice(lst[-1], lst[-1] + windNext), ww])
        
        dodSpline[lst, ww] = dodSpline[lst, ww] - meanCurr + meanNext

    #% intermediate segments
    for kk in range(nbMA-1):
        # no motion
        lst = np.arange(lstMf[kk], lstMs[kk+1])
        SegPrevLength = lstMl[kk]
        SegCurrLength = len(lst)
        
        windPrev = (
            SegPrevLength if SegPrevLength < dtSHORT * fs
            else int(np.floor(dtSHORT * fs)) if SegPrevLength < dtLONG * fs
            else int(np.floor(SegPrevLength / 10))
        )
        
        windCurr = (
            SegCurrLength if SegCurrLength < dtSHORT * fs
            else int(np.floor(dtSHORT * fs)) if SegCurrLength < dtLONG * fs
            else int(np.floor(SegCurrLength / 10))
        )
    
        meanPrev = np.mean(dodSpline[slice(lst[0] - windPrev, lst[0]), ww])
        meanCurr = np.mean(dodSpline[slice(lst[0], lst[0] + windCurr), ww])
    
        dodSpline[lst, ww] = dodSpline[lst, ww] - meanCurr + meanPrev
    
        # motion
        lst = slice(lstMs[kk], lstMf[kk])
        SegPrevLength = SegCurrLength
        SegCurrLength = lstMl[kk]
    
        windPrev = (
            SegPrevLength if SegPrevLength < dtSHORT * fs
            else int(np.floor(dtSHORT * fs)) if SegPrevLength < dtLONG * fs
            else int(np.floor(SegPrevLength / 10))
        )
        
        windCurr = (
            SegCurrLength if SegCurrLength < dtSHORT * fs
            else int(np.floor(dtSHORT * fs)) if SegCurrLength < dtLONG * fs
            else int(np.floor(SegCurrLength / 10))
        )
    
        meanPrev = np.mean(dodSpline[slice(lst.start - windPrev, lst.start), ww])
        meanCurr = np.mean(dodSpline[slice(lst.start, lst.start + windCurr), ww])
    
        dodSpline[lst, ww] = dodSpline[lst, ww] - meanCurr + meanPrev
    
    #% last not MA segment
    if lstMf[-1] < len(t):
        lst = np.arange(lstMf[-1] - 1, len(t))
        SegPrevLength = lstMl[-1]
        SegCurrLength = len(lst)
        
        windPrev = (
            SegPrevLength if SegPrevLength < dtSHORT * fs
            else int(np.floor(dtSHORT * fs)) if SegPrevLength < dtLONG * fs
            else int(np.floor(SegPrevLength / 10))
        )
        
        windCurr = (
            SegCurrLength if SegCurrLength < dtSHORT * fs
            else int(np.floor(dtSHORT * fs)) if SegCurrLength < dtLONG * fs
            else int(np.floor(SegCurrLength / 10))
        )
    
        meanPrev = np.mean(dodSpline[slice(lst[0] - windPrev, lst[0]), ww])
        meanCurr = np.mean(dodSpline[slice(lst[0], lst[0] + windCurr), ww])
    
        dodSpline[lst, ww] = dodSpline[lst, ww] - meanCurr + meanPrev


    ww=ww+1

#%% 

#%%
fig, ax = plt.subplots(1,1)
ax.plot(dod[:,0])
# ax.plot(splInterp(lst))
ax.plot(dodSpline[:,0])




