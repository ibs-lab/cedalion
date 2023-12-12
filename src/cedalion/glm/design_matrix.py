import numpy as np
import pandas as pd
from scipy.stats import norm
import xarray as xr
import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils
import warnings
warnings.filterwarnings("ignore", message="The unit of the quantity is stripped")

def make_design_matrix(y : xr.DataArray, s: xr.DataArray, trange : list, idxBasis: str, paramsBasis, driftOrder: int):
    """
    Generate the HRF regressors (as part of the design matrix)
    inputs:
        y: xarray.DataArray of the data (time x chromo x channels)
        s: xarray.DataArray of the stimulus (time x conditions)
        trange: list of the time range of the HRF regressors [pre, post] (in seconds relative to stimulus onset, for example [-5, 15])
        idxBasis: string indicating the type of basis function to use ("gaussians", "gamma", "gamma_deriv", "afni_gamma", "individual")
        paramsBasis: parameters for the basis function (for example for "gaussians" it is [gms, gstd] where 
                    gms is the mean of the Gaussians and gstd is the standard deviation)
        driftOrder: order of the polynomial drift regressors to add to the design matrix
    outputs:
        A: xarray.DataArray of the design matrix (time x regressors x chromo)
    """

    t = y.time
    cond_names = s.condition.values
    dt = 1 / y.cd.sampling_rate
    nPre = round(trange[0] / dt)
    nPost = round(trange[1] / dt)
    nTpts = y.shape[0]
    tHRF = np.arange(nPre * dt, (nPost + 1) * dt, dt)
    ntHRF = len(tHRF)
    nT = len(t)


    # prune good stim
    # handle case of conditions with 0 trials
    lstCond = np.where(np.sum(s > 0, axis=0) > 0)[0]
    nCond = len(lstCond)  # could also be s.shape[1]
    onset = np.zeros((nT, nCond))
    nTrials = np.zeros(nCond)

    for iCond in range(nCond):
        lstT = np.where(s[:, lstCond[iCond]] == 1)[0]
        lstp = np.where((lstT + nPre >= 0) & (lstT + nPost < nTpts))[0]  # subtract 1 due to 0-indexing in Python
        lst = lstT[lstp]
        nTrials[iCond] = len(lst)
        onset[lst + nPre, iCond] = 1


    ###########################################################################
    # Construct the basis functions

    # Gaussians
    if idxBasis == "gaussians":
        gms = paramsBasis[0]
        gstd = paramsBasis[1]
        
        nB = int((trange[1] - trange[0]) / gms) - 1
        tbasis = np.zeros((ntHRF, nB))
        for b in range(nB):
            tbasis[:, b] = np.exp(-(tHRF - (trange[0] + (b+1) * gms))**2 / (2 * gstd**2))
            tbasis[:, b] = tbasis[:, b] / np.max(tbasis[:, ])  # Normalize to 1

    # Modified Gamma
    elif idxBasis == "gamma":
        params_len = len(paramsBasis)
        if params_len == 3:
            nConc = 1
        elif params_len == 6:
            nConc = 2

        nB = 1
        tbasis = np.zeros((ntHRF, nB, nConc))
        for iConc in range(nConc):
            tau = paramsBasis[iConc*3]
            sigma = paramsBasis[iConc*3 + 1]
            T = paramsBasis[iConc*3 + 2]
            
            tbasis[:, 0, iConc] = (np.exp(1) * (tHRF - tau)**2 / sigma**2) * np.exp(-(tHRF - tau)**2 / sigma**2)
            lstNeg = np.where(tHRF < 0)[0]
            tbasis[lstNeg, 0, iConc] = 0
            
            if tHRF[0] < tau:
                tbasis[:int((tau - tHRF[0]) / dt), 0, iConc] = 0
            
            if T > 0:
                for ii in range(nB):
                    foo = np.convolve(tbasis[:, ii, iConc], np.ones(int(T / dt))) / int(T / dt)
                    tbasis[:, ii, iConc] = foo[:ntHRF]

    # Modified Gamma and Derivative
    elif idxBasis == "gamma_deriv":
        params_len = len(paramsBasis)
        if params_len == 3:
            nConc = 1
        elif params_len == 6:
            nConc = 2

        nB = 2
        tbasis = np.zeros((ntHRF, nB, nConc))
        for iConc in range(nConc):
            tau = paramsBasis[iConc * 3]
            sigma = paramsBasis[iConc * 3 + 1]
            T = paramsBasis[iConc * 3 + 2]
            
            tbasis[:, 0, iConc] = (np.exp(1) * (tHRF - tau)**2 / sigma**2) * np.exp(-(tHRF - tau)**2 / sigma**2)
            tbasis[:, 1, iConc] = 2 * np.exp(1) * ((tHRF - tau) / sigma**2 - (tHRF - tau)**3 / sigma**4) * np.exp(-(tHRF - tau)**2 / sigma**2)
            
            if tHRF[0] < tau:
                tbasis[:int((tau - tHRF[0]) / dt), :2, iConc] = 0
            
            if T > 0:
                for ii in range(nB):
                    foo = np.convolve(tbasis[:, ii, iConc], np.ones(int(T / dt))) / int(T / dt)
                    tbasis[:, ii, iConc] = foo[:ntHRF]

    # AFNI Gamma function
    elif idxBasis == "afni_gamma":
        params_len = len(paramsBasis)
        if params_len == 3:
            nConc = 1
        elif params_len == 6:
            nConc = 2

        nB = 1
        tbasis = np.zeros((ntHRF, nB, nConc))
        for iConc in range(nConc):
            p = paramsBasis[iConc * 3]
            q = paramsBasis[iConc * 3 + 1]
            T = paramsBasis[iConc * 3 + 2]
            
            tbasis[:, 0, iConc] = (tHRF / (p * q))**p * np.exp(p - tHRF / q)
            
            if T > 0:
                foo = np.convolve(tbasis[:, 0, iConc], np.ones(int(T / dt))) / int(T / dt)
                tbasis[:, 0, iConc] = foo[:ntHRF]

    # Individualized basis function for each channel from a previously estimated HRF
    elif idxBasis == "individual":
        nConc = 2  # HbO and HbR separate basis
        nB = 1
        tbasis = np.zeros((paramsBasis.shape[2], ntHRF, nB, nConc))
        for iConc in range(nConc):
            for iCh in range(paramsBasis.shape[2]):
                tbasis[iCh, :, 0, iConc] = paramsBasis[:, iConc, iCh]

    ###########################################################################
    # Construct design matrix

    if idxBasis != "individual":
        dA = np.zeros((nT, nB * nCond, 2))
        for iConc in range(2):
            iC = -1
            for iCond in range(nCond):
                for b in range(nB):
                    iC += 1
                    if len(tbasis.shape) == 2:
                        clmn = np.convolve(onset[:, iCond], tbasis[:, b])
                    else:
                        clmn = np.convolve(onset[:, iCond], tbasis[:, b, iConc])
                    clmn = clmn[:nT]
                    dA[:, iC, iConc] = clmn


    elif idxBasis == "individual":
        dA = np.zeros((nT, nB * nCond, 2, paramsBasis.shape[2]))
        for iConc in range(2):
            for iCh in range(paramsBasis.shape[2]):
                iC = -1
                for iCond in range(nCond):
                    for b in range(nB):
                        iC += 1
                        clmn = np.convolve(onset[:, iCond], np.squeeze(tbasis[iCh, :, b, iConc]))
                        clmn = clmn[:nT]
                        dA[:, iC, iConc, iCh] = clmn

    ###########################################################################
    # Add drift regressors
    xDrift = np.ones((nT, driftOrder + 1))

    for ii in range(1, driftOrder + 1):
        xDrift[:, ii] = np.arange(1, nT + 1) ** (ii)
        xDrift[:, ii] = xDrift[:, ii] / xDrift[-1, ii]


    ###########################################################################
    # Stack drift and HRF regressors

    A = np.zeros((dA.shape[0], dA.shape[1] + xDrift.shape[1], 2))

    for iConc in range(2):  # Looping through the two concentrations
        A[:, :, iConc] = np.hstack([dA[:, :, iConc], xDrift])

    ###########################################################################
    # To xarray

    regressor_names = []
    # HRF regressors
    for iCond in range(nCond):
        for b in range(nB):
            regressor_names.append("HRF " + cond_names[iCond] + " " + str(b + 1))
    # Drift regressors
    for i in range(driftOrder + 1):
        regressor_names.append('Drift ' + str(i))

    A = xr.DataArray(A, dims=['time', 'regressor', 'chromo'], coords={'time': y.time, 'regressor': regressor_names, 'chromo': ['HbO', 'HbR']})
    A = A.pint.quantify("micromolar")
    #print(A.pint.units)
    return A



def get_ss_regressors(y: xr.DataArray, geo3d: xr.DataArray, ssMethod = 'closest' ,ssTresh: float = 1.5):
    """
    Get short separation channels for each long channel
    inputs:
        y: xarray.DataArray of the data (time x chromo x channels)
        geo3d: xarray.DataArray of the 3D geometry (number of sources/detectors x dim pos)
        ssMethod: method for determining short separation channels ("nearest", "correlation", "average")
        ssTresh: threshold for short separation channels (in cm)
    outputs:
        ss: xarray.DataArray of short separation channels (channels x chromo)
    """

    # Calculate source-detector distances for each channel
    dists = xrutils.norm(geo3d.loc[y.source] - geo3d.loc[y.detector], dim="pos").pint.to("mm").round(2)
    
    # Identify short channels
    short_channels = dists.channel[dists < ssTresh * cedalion.units.cm]

    if ssMethod == "closest":
        middle_pos = (geo3d.loc[y.source] + geo3d.loc[y.detector]) / 2
        return closest_short_channel(y, short_channels, middle_pos)
    elif ssMethod == "corr":
        return max_corr_short_channel(y, short_channels)


def closest_short_channel(y, short_channels, middle_positions):
    # Initialize array to store closest short channel for each channel
    closest_short = xr.DataArray([["______", "______"] for i in range(len(y.channel))], dims=["channel", "chromo"], coords={"channel": y.channel, "chromo": y.chromo})
    # For each channel, find the closest short channel
    for ch in y.channel:
        # Compute distances from this channel's middle position to all short channel middle positions
        distances_to_short = xrutils.norm(middle_positions.loc[short_channels.channel] - middle_positions.loc[ch], dim="pos")
        # Find the closest short channel
        closest_short_channel = short_channels.channel[(distances_to_short).argmin()]
        # Store the result
        closest_short.loc[ch, 'HbO'] = closest_short_channel
        closest_short.loc[ch, 'HbR'] = closest_short_channel

    return closest_short


def max_corr_short_channel(y, short_channels):
    # Get indices of short channels
    lstSS = [np.where(y.channel == ch)[0][0] for ch in short_channels]
    
    # HbO
    dc = y[:, 0, :].squeeze()
    dc = (dc - np.mean(dc, axis=0)) / np.std(dc, axis=0)
    cc1 = np.dot(dc.T, dc) / len(dc)

    # HbR
    dc = y[:, 1, :].squeeze()
    dc = (dc - np.mean(dc, axis=0)) / np.std(dc, axis=0)
    cc2 = np.dot(dc.T, dc) / len(dc)

    iNearestSS = np.zeros((cc1.shape[0], 2), dtype=int)
    for iML in range(cc1.shape[0]):
        iNearestSS[iML, 0] = lstSS[np.argmax(cc1[iML, lstSS])]
        iNearestSS[iML, 1] = lstSS[np.argmax(cc2[iML, lstSS])]    
     
    # transform to xarray
    channel_iSS = [y.channel[i] for i in iNearestSS]
    highest_corr_short = xr.DataArray(channel_iSS, dims=["channel", "chromo"], coords={"channel": y.channel, "chromo": y.chromo})

    return highest_corr_short
    
