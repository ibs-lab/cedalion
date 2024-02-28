#!/usr/bin/env python
# coding: utf-8


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

xr.set_options(display_expand_data=False)


#%% LOAD THE DATA 

DATADIR = "/Users/lauracarlton/Library/CloudStorage/GoogleDrive-lcarlton@bu.edu/My Drive/fNIRS/Data/OLD_FT/BIDS-NIRS-Tapping/"
elements = cedalion.io.read_snirf(DATADIR + "sub-01/nirs/sub-01_task-tapping_nirs.snirf")

amp = elements[0].data[0]
dod1 =  -np.log( amp / amp.mean("time"))

#%%  hmrR_tInc_baselineshift_Ch_Nirs

# need to first correct for baseline shifts
IQR_GRAD_THRESH = 1.5
IQR_STD_THRESH = 2
T_MOTION = 1

fs =  dod1.cd.sampling_rate
win_size = int(np.round(fs*T_MOTION))

t = dod1.time
extend = int(np.round(12*fs))

dod = dod1.stack(measurement = ['channel', 'wavelength']).sortby('wavelength')
SNR_thre = np.zeros(dod.shape[1])
tInc = np.zeros([dod.shape[0] + extend*2, dod.shape[1]])

ww=0
# want to loop through all channels
for measurement in tqdm(dod['measurement']):
# select one channel for now instead of looping through all of them 
# s1 = dod.sel(channel='S1D1', wavelength=850) #, time=dod.time[np.arange(10000)])
    s1 = dod.sel(measurement=measurement)
    
    # filter data between 0, 2 
    s1 = s1.cd.freq_filter(0, 2, butter_order=4)
    
    # filter data between 0 0.5
    s2 = s1.cd.freq_filter(0, 0.5, butter_order=4)
    
    #% DETECTING OUTLIERS
    # detect outliers in std variations of the signal 
    windows = s2.rolling(time=win_size).construct("window", stride=1)
    sigSTD = windows.std("window")
    
    quants_std = sigSTD.quantile([0.25,0.5,0.75])
    IQR_std = quants_std[2] - quants_std[0]
    high_std = quants_std[2] + IQR_std*IQR_STD_THRESH
    low_std = quants_std[0] - IQR_std*IQR_STD_THRESH
    #%
    
    # detect outliers in gradient of the signal
    grad = s1.copy()
    grad.values = signal.convolve(s1.values, [-1,0,1], mode='same')
    
    quants_grad = grad.quantile([0.25,0.5,0.75])
    IQR_grad = quants_grad[2] - quants_grad[0]
    high_grad = quants_grad[2] + IQR_grad*IQR_GRAD_THRESH
    low_grad = quants_grad[0] - IQR_grad*IQR_GRAD_THRESH
    
    # union of all outliers
    condition = (sigSTD[:-win_size] > high_std) | (sigSTD[:-win_size] < low_std)
    M_std = np.where(condition)[0]

    condition = (grad > high_grad) | (grad < low_grad)
    M_grad = np.where(condition)[0]

    
    if len(M_std) != 0:
        M_std = np.round(win_size/2) + M_std
        
    
    # combine grad and std outliers -> M
    if len(M_std) != 0 and len(M_grad) != 0:
        set1 = set(M_std)
        set2 = set(M_grad)
        M = np.array(list(set1.symmetric_difference(set2))).astype(int)
    
    elif len(M_std) != 0:
        M = M_std.astype(int)
    elif len(M_grad) != 0:
        M = M_grad.astype(int)
    
    
    #% extend s1 
    s1_pad = s1.pad(time = extend, mode='edge')

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
    
    
#%% calculate SNR for all channels
    


for ww in range(np.shape(dod)[1]):
    if np.isnan(SNR_thre[ww]) or SNR_thre[ww] == 0:
        SNR_thre[ww] = abs(dod[ww].mean('time')/dod[ww].std('time'))
        # dmean = np.mean(dod[:,ww])
        # dstd = np.std(dod[:,ww])
        # SNR_thre[ww] = abs(dmean)/dstd    


#%% extract noisy channels from baseline-shift motion correction procedure
    
lent = SNR_thre.shape[0]//2-1

SNRvalue=3

for ww in range(len(SNR_thre)//2-1):
    
    if SNR_thre[ww] < SNRvalue and SNR_thre[ww+lent] < SNRvalue:
        tInc[:,ww+lent] = np.ones(len(s1_pad))
        tInc[:,ww] = np.ones(len(s1_pad))
    
    elif SNR_thre[ww] > SNRvalue and SNR_thre[ww+lent] < SNRvalue:
        tInc[:,ww+lent] = tInc[:,ww]
    
    elif SNR_thre[ww] < SNRvalue and SNR_thre[ww+lent] > SNRvalue:
        tInc[:,ww] = tInc[:,ww+lent]
    


tIncCh = tInc[extend:-extend,:]
tInc = tInc[extend:-extend,:]
tIncall = tInc[:,0]

for kk in range(tInc.shape[1]):
    tIncall = tIncall * tInc[:,kk]

tInc = tIncall
    
        
        
#TODO - 
# - maintain dataArray structure better 
# - validate with matlab 
# - check filtering with Eike 

        
        
        
        
        
        
        
        
