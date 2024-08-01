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

#%%
import os
subjID = 'sub-01'
rootDir_data = "/projectnb/nphfnirs/ns/lcarlton/DATA/MAFC_raw/"

subj_temp = subjID + '/nirs/' + subjID + '_task-squats_run-01_nirs.snirf'
file_name = os.path.join(rootDir_data, subj_temp)

elements = io.read_snirf(file_name)

amp = elements[0].data[0]

#%%
# import pdb 
# pdb.set_trace()
metric_dict = dqr.get_data_metrics(amp)

#%% 
from SQE_metrics import GVTD as GVTD_old
from snirf import Snirf
snirf_obj = Snirf(file_name)
gvtd_old = GVTD_old(snirf_obj, ax=None)

#%% 
import matplotlib.pyplot as plt 
fig,ax = plt.subplots(1,1)
title='SCI x PSP'
dqr.plot_timeseries_all_channels(MD['SCIxPSP_mask'], elements[0].stim, ax, title)
#%%
fig,ax = plt.subplots(1,1)
dqr.plot_timeseries(MD['GVTD'].squeeze(), elements[0].stim, ax)  

#%%
import pdb
pdb.set_trace()

MD = dqr.generate_report_single_run(elements[0])

#%%
from snirf import Snirf
pdb.set_trace()
snirf_obj = Snirf(file_name)
sci_val, psp_val, mask = sqm.sci_psp(snirf_obj, mode='montage', ax=None)
#%% 
from cedalion.sigproc.quality import sci, psp
from cedalion import Quantity, units
pdb.set_trace()

SCI, SCI_mask = sci(elements[0].data[0], 5*units.s, 0.8)
PSP, PSP_mask = psp(elements[0].data[0], 5*units.s, 0.1)



