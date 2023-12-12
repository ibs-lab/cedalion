import numpy as np
import pandas as pd
from nilearn.glm.first_level import run_glm as nilearn_run_glm
import xarray as xr
import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils
import warnings
warnings.filterwarnings("ignore", message="moin")

def solve_glm(y, design_matrix, ss_regressors, noise_model='ols'):
    """
    Solve the GLM for a given design matrix and data.
    inputs:
        data: xarray.DataArray containing the data (time x chromo x channels)
        design_matrix: design matrix used for the GLM (time x regressors x chromo)
        ss: xarray.DataArray containing the short channel regressors (channels x chromo)
        noise_model: noise model used for the GLM (default = 'ols')
    return: 
        thetas: xarray.DataArray estimated parameters of the GLM (regressors x chromo x channels)
    """

    thetas = xr.DataArray(np.zeros((design_matrix.regressor.size + 1, y.chromo.size, y.channel.size)), dims=['regressor', 'chromo', 'channel'], 
                          coords={'regressor': design_matrix.regressor.values.tolist() + ['SS'], 'chromo': y.chromo, 'channel': y.channel})

    predicted = xr.DataArray(np.zeros((y.time.size, y.chromo.size, y.channel.size)), dims=['time', 'chromo', 'channel'],
                                coords={'time': y.time, 'chromo': y.chromo, 'channel': y.channel})

    predicted_hrf = xr.DataArray(np.zeros((y.time.size, y.chromo.size, y.channel.size)), dims=['time', 'chromo', 'channel'],
                                coords={'time': y.time, 'chromo': y.chromo, 'channel': y.channel})

    hrf_regs = design_matrix.regressor.str.startswith('HRF').values.tolist()
    hrf_regs += [False] # SS regressor

    for chromo in y.chromo:
        ss_chromo = ss_regressors.sel(chromo=chromo)
        for ss in np.unique(ss_chromo.values):
            # get channels to which the current ss regressor is assigned
            ch = ss_chromo.where(ss_chromo == ss, drop=True).channel
            # add ss regressor to design matrix
            ss_arr = xr.DataArray(np.expand_dims(y.sel(channel=ss, chromo=chromo), 1), dims=['time', 'regressor'], coords={'time': y.time, 'regressor': ['SS']})
            ss_arr = ss_arr.pint.quantify("micromolar") / abs(ss_arr).max()
            dm = xr.concat([design_matrix.sel(chromo=chromo), ss_arr], dim='regressor')
            # solve GLM
            labels, glm_est = nilearn_run_glm(y.sel(channel=ch, chromo=chromo).values, dm.values, noise_model=noise_model)
            thetas_temp = xr.DataArray(glm_est[labels[0]].theta, dims = ("regressor", "channel"), coords = {"regressor" : dm.regressor, "channel" : ch})
            # store results
            thetas.loc[{'regressor': dm.regressor, 'chromo': chromo, 'channel': ch}] = thetas_temp
            predicted.loc[{'time': y.time, 'chromo': chromo, 'channel': ch}] = glm_est[labels[0]].predicted
            predicted_hrf.loc[{'time': y.time, 'chromo': chromo, 'channel': ch}] = dm.sel(regressor=hrf_regs).values @ thetas_temp.sel(regressor=hrf_regs).values

    predicted = predicted.pint.quantify(y.pint.units)
    predicted_hrf = predicted_hrf.pint.quantify(y.pint.units)
    return thetas, predicted, predicted_hrf




def get_HRFs(predicted_hrf: xr.DataArray, stim: pd.DataFrame, id_stim: int = 0 ,HRFmin: int = -2, HRFmax: int = 15, plot_HRFs=False):
    """
    Get HRFs for each condition and channel estimated by the GLM.
    inputs:
        predicted_hrf: xarray.DataArray containing the predicted HRFs (time x chromo x channels)
        stim: pandas.DataFrame containing the stimulus information
        id_stim: id of the stimulus block for which the HRFs are estimated (default = 0)
        HRFmin: minimum time of the HRF (default = -2)
        HRFmax: maximum time of the HRF (default = 15)
        plot_HRFs: plot the estimated HRFs (default = False)
    
    return: 
        hrfs: xarray.DataArray containing HRFs for every condition and every channel (time x chromo x channels x conditions)
    """

    # get id_stim-th stim onset of each condition
    stim_onsets = stim.groupby("trial_type").onset.nth(id_stim).values
    conds = stim.trial_type.unique()
    stim_onsets = xr.DataArray(stim_onsets, dims="condition", coords={"condition" : conds})

    # get time axis for HRFs:
    dt = 1/predicted_hrf.cd.sampling_rate
    time_hrf = np.arange(HRFmin, HRFmax+dt, dt)

    hrfs = xr.DataArray(np.zeros((time_hrf.size, predicted_hrf.chromo.size, predicted_hrf.channel.size, conds.size)),
                    dims=["time", "chromo", "channel", "condition"],
                    coords={"time": time_hrf, "chromo": predicted_hrf.chromo, "channel": predicted_hrf.channel, "condition": conds})

    for chromo in predicted_hrf.chromo:
        for cond in conds:
            # select HRFs for current chromophore and condition
            hrf = predicted_hrf.sel(chromo=chromo, 
                                    time=slice(stim_onsets.sel(condition = cond) + HRFmin, stim_onsets.sel(condition = cond) + HRFmax))
            # store HRFs
            hrfs.loc[{"time": time_hrf, "chromo": hrf.chromo, "condition": cond}] = hrf.values

    # remove baseline
    hrfs = hrfs - hrfs.sel(time=slice(HRFmin, 0)).mean(dim="time")
    # add units
    hrfs = hrfs.pint.quantify(predicted_hrf.pint.units)

    return hrfs