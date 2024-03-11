
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
import xarray as xr
import cedalion.typing as cdt
from cedalion import Quantity, units
import cedalion.dataclasses as cdc
from .artifact import detect_outliers, detect_baselineshift

@cdc.validate_schemas
def motionCorrectSpline(fNIRSdata:cdt.NDTimeSeries, tIncCh:cdt.NDTimeSeries): #, mlAct:cdt.NDTimeSeries):
    """
    Apply motion correction using spline interpolation to fNIRS data.
    
    Based on Homer3 [1] v1.80.2 "hmrR_tInc_baselineshift_Ch_Nirs.m"
    Boston University Neurophotonics Center
    https://github.com/BUNPC/Homer3

    Inputs:
        fNIRSdata (cdt.NDTimeSeries): The fNIRS data to be motion corrected.
        tIncCh (cdt.NDTimeSeries): The time series indicating the presence of motion artifacts.

    Returns:
        dodSpline (cdt.NDTimeSeries): The motion-corrected fNIRS data.
    """
    dtShort = 0.3
    dtLong = 3
    
    fNIRSdata = fNIRSdata.stack(measurement = ['channel', 'wavelength']).sortby('wavelength').pint.dequantify()
    tIncCh = tIncCh.stack(measurement = ['channel', 'wavelength']).sortby('wavelength').pint.dequantify()

    fs =  fNIRSdata.cd.sampling_rate 
    t = fNIRSdata['time']

    dodSpline = fNIRSdata.copy()

    for m,measurement in enumerate(fNIRSdata.measurement):
        
        tInc_channel = tIncCh.sel(measurement=measurement)
        channel = fNIRSdata.sel(measurement=measurement)
        dodSpline_chan = channel.copy()

        # get list of start and finish of each motion artifact segment
        lstMA = np.where(tInc_channel == 0)[0]
        if len(lstMA) != 0:
            temp = np.diff(tInc_channel.values.astype(int))
            lstMs = np.where(temp==-1)[0]
            lstMf = np.where(temp==1)[0]
            
            if len(lstMs) == 0:
                lstMs = [0]
            if len(lstMf) == 0:
                lstMf = len(channel)-1
            if lstMs[0] > lstMf[0]:
                lstMs = np.insert(lstMs, 0, 0)
            if lstMs[-1] > lstMf[-1]:
                lstMf.append(len(channel)-1)
            
            nbMA = len(lstMs)
            lstMl = lstMf - lstMs
            
            # apply spline interpolation to each motion artifact segment
            for ii in range(nbMA):
                idx = np.arange(lstMs[ii],lstMf[ii])
                
                if len(idx) > 3:
                    splInterp_obj = UnivariateSpline(t[idx], channel[idx])
                    splInterp = splInterp_obj(t[idx])
                    
                    dodSpline_chan[idx] = channel[idx] - splInterp

       
            # reconstruct the timeseries by shifting the motion artifact segments to the previous or next non-motion artifact segment
            # for the first MA segment - shirf to previous noMA segment if it exists otherwise shift to next noMA segment
            idx = np.arange(lstMs[0], lstMf[0])
            SegCurrLength = lstMl[0]
            windCurr = compute_window(SegCurrLength, dtShort, dtLong, fs)
            
            if lstMs[0] > 0:
                SegPrevLength = lstMs[0]
                windPrev = compute_window(SegPrevLength, dtShort, dtLong, fs)
                meanPrev = np.mean(dodSpline_chan[idx[0]-windPrev:idx[0]])                      
                meanCurr = np.mean(dodSpline_chan[idx[0]:idx[0]+windCurr])
                dodSpline_chan[idx] = dodSpline_chan[idx] - meanCurr + meanPrev    
            else:
                if nbMA > 1:
                    SegNextLength = lstMs[1]- lstMf[0]
                else:
                    SegNextLength = len(dodSpline_chan) - lstMf[0]
                
                windNext = compute_window(SegNextLength, dtShort, dtLong, fs)
                meanNext = np.mean(dodSpline_chan[idx[-1]:idx[-1]+windNext])                      
                meanCurr = np.mean(dodSpline_chan[idx[-1]-windCurr:idx[-1]])
                dodSpline_chan[idx] = dodSpline_chan[idx] - meanCurr + meanNext
            
            # intermediate segments
            for kk in range(nbMA-1):
                
                # no motion 
                idx = np.arange(lstMf[kk], lstMs[kk+1])
                SegPrevLength = lstMl[kk]
                SegCurrLength = len(idx)
                
                windPrev = compute_window(SegPrevLength, dtShort, dtLong, fs)
                windCurr = compute_window(SegCurrLength, dtShort, dtLong, fs)
    
                meanPrev = np.mean(dodSpline_chan[idx[0]-windPrev:idx[0]])                      
                meanCurr = np.mean(channel[idx[0]:idx[0]+windCurr])
                
                dodSpline_chan[idx] = channel[idx] - meanCurr + meanPrev
                
                # motion 
                idx = np.arange(lstMs[kk+1], lstMf[kk+1])
    
                SegPrevLength = SegCurrLength
                SegCurrLength = lstMl[kk+1]
                
                windPrev = compute_window(SegPrevLength, dtShort, dtLong, fs)
                windCurr = compute_window(SegCurrLength, dtShort, dtLong, fs)
    
                meanPrev = np.mean(dodSpline_chan[idx[0]-windPrev:idx[0]])                      
                meanCurr = np.mean(dodSpline_chan[idx[0]:idx[0]+windCurr])
                
                dodSpline_chan[idx] = dodSpline_chan[idx] - meanCurr + meanPrev
    
            
            # last not MA segment
            if lstMf[-1] < len(dodSpline_chan):
                idx = np.arange(lstMf[-1], len(dodSpline_chan))
                SegPrevLength = lstMl[-1]
                SegCurrLength = len(idx)
                
                windPrev = compute_window(SegPrevLength, dtShort, dtLong, fs)
                windCurr = compute_window(SegCurrLength, dtShort, dtLong, fs)
                        
                meanPrev = np.mean(dodSpline_chan[idx[0]-windPrev:idx[0]])                      
                meanCurr = np.mean(channel[idx[0]:idx[0]+windCurr])
                
                dodSpline_chan[idx] = channel[idx] - meanCurr + meanPrev
            
        dodSpline[:,m] = dodSpline_chan
    
    dodSpline = dodSpline.unstack('measurement').pint.quantify()
    return dodSpline

#%%
def compute_window(SegLength:cdt.NDTimeSeries, dtShort:Quantity, dtLong:Quantity, fs:Quantity):
    """
    Computes the window size based on the segment length, short time interval, long time interval, and sampling frequency.

    Inputs:
        SegLength (cdt.NDTimeSeries): The length of the segment.
        dtShort (Quantity): The short time interval.
        dtLong (Quantity): The long time interval.
        fs (Quantity): The sampling frequency.

    Returns:
        wind: The computed window size.
    """
    if SegLength < dtShort * fs:
        wind = SegLength
    elif SegLength < dtLong * fs:
        wind = np.floor(dtShort * fs)
    else:
        wind = np.floor(SegLength / 10)
    return int(wind)

#%%
@cdc.validate_schemas
def motionCorrectSplineSG(fNIRSdata:cdt.NDTimeSeries, framesize_sec:Quantity = 10 ): #, mlAct:cdt.NDTimeSeries):
    """
    Apply motion correction using spline interpolation and Savitzky-Golay filter to fNIRS data.

    Inputs:
        fNIRSdata (cdt.NDTimeSeries): The fNIRS data to be motion corrected.
        framesize_sec (Quantity): The size of the sliding window in seconds for the Savitzky-Golay filter. Default is 10 seconds.

    Returns:
        dodSplineSG (cdt.NDTimeSeries): The motion-corrected fNIRS data after applying spline interpolation and Savitzky-Golay filter.
    """
    
    fs =  fNIRSdata.cd.sampling_rate
    
    M, M_array = detect_outliers(fNIRSdata, 1)
    
    tInc,tIncCh = detect_baselineshift(fNIRSdata, M)
    
    fNIRSdata_lpf2 = fNIRSdata.cd.freq_filter(0, 2, butter_order=4)
    extend = int(np.round(12*fs)) # extension for padding
    
    # pad fNIRSdata and tIncCh for motion correction 
    fNIRSdata_lpf2_pad = fNIRSdata_lpf2.pad(time=extend, mode='edge')
    
    tIncCh_pad = tIncCh.pad(time=extend, mode='edge')
    
    dodSpline = motionCorrectSpline(fNIRSdata_lpf2_pad, tIncCh_pad)
    
    # remove padding
    dodSpline = dodSpline[extend:-extend,:]
    dodSpline = dodSpline.stack(measurement = ['channel', 'wavelength']).sortby('wavelength').pint.dequantify()

    # apply SG filter
    K = 3
    framesize_sec = int(np.round(framesize_sec*fs))
    if framesize_sec//2 == 0:
        framesize_sec = framesize_sec + 1

    dodSplineSG = xr.apply_ufunc(savgol_filter, dodSpline.T, framesize_sec, K).T
    
    dodSplineSG = dodSplineSG.unstack('measurement').pint.quantify()
    
    return dodSplineSG



