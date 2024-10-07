#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:50:05 2024

@author: lauracarlton
"""

from cedalion.sigproc import data_quality_report as dqr

import cedalion.nirs as nirs
import cedalion.io as io
import os
import SQE_metrics as sqm
from cedalion import units
from cedalion.sigproc.frequency import freq_filter, sampling_rate
import numpy as np 
import scipy.signal as signal 
import cedalion.xrutils as xrutils 
import cedalion.sigproc.quality as quality 
import pdb

#%%
subjID = 'sub-01'
# rootDir_data = "/projectnb/nphfnirs/ns/lcarlton/DATA/MAFC_raw/"
rootDir_data = "/Users/lauracarlton/Documents/DATA/MAFC_raw/"
subj_temp = subjID + '/' + subjID + '_task-MA_run-01_nirs.snirf'
file_name = os.path.join(rootDir_data, subj_temp)

elements = io.read_snirf(file_name)

amplitudes = elements[0]['amp']

#%% SQE implementation
from snirf import Snirf
import pickle 
# pdb.set_trace()
snirf_obj = Snirf(file_name)
sci_val, psp_val, mask = sqm.sci_psp(snirf_obj, mode='montage', ax=None)
sqe_output = {'sci_val':sci_val, 
              'psp_val':psp_val, 
              'mask': mask}
with open('sqe_metrics_output.pkl', 'wb') as f:
    pickle.dump(sqe_output, f)
    

#%% cedalion sci implementation 

sci_ced, sci_mask_ced = quality.sci(amplitudes)

with open('ced_sci_output.pkl', 'wb') as f:
    pickle.dump(sci_ced, sci_mask_ced, f)

#%% cedalion implementation 


window_length = 5*units.s
psp_thresh = 0.1
cardiac_fmin = 0.5 * units.Hz
cardiac_fmax = 2.5 * units.Hz

amp = freq_filter(amplitudes, cardiac_fmin, cardiac_fmax, butter_order=4)
amp = amp.pint.dequantify()

amp = (amp - amp.mean("time")) / amp.std("time")
wavelengths = amp.wavelength
# convert window_length to samples
nsamples = (window_length * sampling_rate(amp)).to_base_units()
nsamples = int(np.ceil(nsamples))

# This creates a new DataArray with a new dimension "window", that is
# window_len_samples large. The time dimension will contain the time coordinate of
# the first sample in the window. Setting the stride size to the same value as the
# window length will result in non-overlapping windows.
windows = amp.rolling(time=nsamples).construct("window", stride=nsamples)
windows = windows.dropna('time')
# windows = windows.assign_coords({'window':np.arange()})

fs = amp.cd.sampling_rate


# Vectorized signal extraction and correlation
# sig = windows.transpose('channel', 'time', 'wavelength','window')
pdb.set_trace()
psp_xr = amp.isel(time=windows.samples.values, wavelength=0)
  
for window in windows.time:
    
    for chan in windows.channel:
        
            sig_temp = windows.sel(channel=chan, time=window)
                    
            similarity = np.correlate(sig_temp.sel(wavelength=wavelengths[0]).values, sig_temp.sel(wavelength=wavelengths[1]).values, 'full')
            similarity = similarity / nsamples
            
            # lags = np.arange(-nsamples + 1, nsamples)
            # similarity_unbiased = similarity / (nsamples - np.abs(lags))
            
            # similarity_norm = (nsamples * similarity_unbiased) / np.sqrt(np.sum(np.abs(sig_temp.sel(wavelength=wavelengths[0]).values)**2) * np.sum(np.abs(sig_temp.sel(wavelength=wavelengths[0]).values)**2))
            # similarity_norm[np.isnan(similarity_norm)] = 0
        
        
            # f, pxx = signal.periodogram(similarity_norm.T, window=signal.hamming(len(similarity_norm)), nfft=len(similarity_norm), fs=fs,  scaling='density')
            nfft = max(256, 2**int(np.ceil(np.log2(len(similarity))))) 
            f, pxx = signal.periodogram(similarity.T, window='hamming', nfft=nfft, fs=fs, scaling='density')
            # f, pxx = signal.welch(similarity.T,nfft=nfft, fs=fs, scaling='density')

            psp_xr.sel(channel=chan, time=window).values = np.max(pxx)            
    
    

# Apply threshold mask
psp_mask = xrutils.mask(psp_xr, True)
psp_mask = psp_mask.where(psp_xr < psp_thresh, False)


#%%

metric_dict = dqr.get_data_metrics(elements[0]['amp'])



#%%
# pdb.set_trace()
S = dqr.generate_report_single_run(elements[0])


#%%
pdb.set_trace()

psp_xr, psp_mask = quality.psp(amplitudes)

#%%
pdb.set_trace()

sci_xr, sci_mask = quality.sci(amplitudes, 5*units.s, 0.8)
