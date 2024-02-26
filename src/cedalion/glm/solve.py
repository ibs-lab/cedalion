import numpy as np
import pandas as pd
from nilearn.glm.first_level import run_glm as nilearn_run_glm
import xarray as xr


def solve_glm(
    y: xr.DataArray,
    design_matrix: xr.DataArray,
    local_regressors={"HbO": {}, "HbR": {}},
    noise_model="ols",
):
    """Solve the GLM for a given design matrix, additional regressors (channel dependent) and data.

    inputs:
        y: xarray.DataArray containing the data (time x chromo x channels)
        design_matrix: design matrix used for the GLM (time x regressors x chromo)
        local_regressors: dictionary containing additional regressors for each chromophore
                        local_regressors['data']: xarray.DataArray containing the additional regressors (regressor x chromo x channels)
                        local_regressors['HbO']: dictionary containing the additional regressors for HbO
                            local_regressors['HbO']['regressor_1']: list of HbO channels assigned to regressor_1
                        local_regressors['HbR']: dictionary containing the additional regressors for HbR
                            local_regressors['HbR']['regressor_1']: list of HbR channels assigned to regressor_1
        noise_model: noise model used for the GLM (default = 'ols')
    return:
        thetas: xarray.DataArray estimated parameters of the GLM (regressors x chromo x channels)
        predicted: xarray.DataArray predicted data (time x chromo x channels)
        predicted_hrf: xarray.DataArray predicted HRFs (time x chromo x channels)
    """

    thetas = xr.DataArray(
        np.zeros((design_matrix.regressor.size, y.chromo.size, y.channel.size)),
        dims=["regressor", "chromo", "channel"],
        coords={
            "regressor": design_matrix.regressor.values.tolist(),
            "chromo": y.chromo,
            "channel": y.channel,
        },
    )

    predicted = xr.DataArray(
        np.zeros((y.time.size, y.chromo.size, y.channel.size)),
        dims=["time", "chromo", "channel"],
        coords={"time": y.time, "chromo": y.chromo, "channel": y.channel},
    )

    predicted_hrf = xr.DataArray(
        np.zeros((y.time.size, y.chromo.size, y.channel.size)),
        dims=["time", "chromo", "channel"],
        coords={"time": y.time, "chromo": y.chromo, "channel": y.channel},
    )

    # save units
    units = y.pint.units
    y = y.pint.dequantify()
    design_matrix = design_matrix.pint.dequantify()
    if "data" in local_regressors:
        local_regressors["data"] = local_regressors["data"].pint.dequantify()

    # process local_regressors so that
    # each channel is assigned to not more than one regressor
    local_regressors["HbO"] = process_regressors(
        local_regressors["HbO"], y.channel.values
    )
    local_regressors["HbR"] = process_regressors(
        local_regressors["HbR"], y.channel.values
    )

    hrf_regs = design_matrix.regressor.sel(
        regressor=design_matrix.regressor.str.startswith("HRF")
    )

    for chromo in y.chromo.values:
        for assigned_regs, ch in local_regressors[chromo].items():
            # add assigned regressor to design matrix
            if assigned_regs != ():
                add_reg = local_regressors["data"].sel(
                    regressor=np.array(assigned_regs), chromo=chromo
                )
                # add_reg = add_reg.pint.quantify("micromolar") / abs(add_reg).max()
                dm = xr.concat(
                    [design_matrix.sel(chromo=chromo), add_reg], dim="regressor"
                )
            else:
                dm = design_matrix.sel(chromo=chromo)
            # solve GLM for current chromophore and channels
            thetas_temp = run_glm(
                y.sel(channel=ch, chromo=chromo), dm, noise_model=noise_model
            )
            # store results
            thetas.loc[{"chromo": chromo, "channel": ch}] = thetas_temp.sel(
                regressor=thetas.regressor
            )
            predicted.loc[{"time": y.time, "chromo": chromo, "channel": ch}] = (
                dm.values @ thetas_temp.values
            )
            predicted_hrf.loc[{"time": y.time, "chromo": chromo, "channel": ch}] = (
                dm.sel(regressor=hrf_regs).values
                @ thetas_temp.sel(regressor=hrf_regs).values
            )

    predicted = predicted.pint.quantify(units)
    predicted_hrf = predicted_hrf.pint.quantify(units)

    return thetas, predicted, predicted_hrf


def process_regressors(regressors, all_channels):
    # Identify all unique channels and which regressors they belong to
    channel_to_regressors = {}
    for regressor, channels in regressors.items():
        for channel in channels:
            if channel in channel_to_regressors:
                channel_to_regressors[channel].append(regressor)
            else:
                channel_to_regressors[channel] = [regressor]

    # Include extra_channels in the mapping without assigning them yet
    for channel in all_channels:
        if channel not in channel_to_regressors:
            channel_to_regressors[channel] = []

    # Prepare final assignments structure
    final_assignments = {regressor: [] for regressor in regressors}

    # Assign channels to regressors or tuples, resolving conflicts
    for channel, assigned_regressors in channel_to_regressors.items():
        if len(assigned_regressors) == 1:
            # Channel is unique to one regressor
            final_assignments[assigned_regressors[0]].append(channel)
        elif len(assigned_regressors) > 1:
            # Channel is shared; create or update a tuple entry
            assigned_regressors.sort()
            tuple_key = tuple(assigned_regressors)
            if tuple_key not in final_assignments:
                final_assignments[tuple_key] = [channel]
            else:
                final_assignments[tuple_key].append(channel)

    # Assign leftover channels to a new regressor (empty tuple) if they exist
    assigned_channels = set(
        [channel for channels in final_assignments.values() for channel in channels]
    )
    leftover_channels = [
        channel for channel in all_channels if channel not in assigned_channels
    ]
    if leftover_channels:
        final_assignments[tuple()] = leftover_channels

    # Clean up the structure before returning
    for key in list(final_assignments.keys()):
        if not final_assignments[key]:
            del final_assignments[key]  # Remove empty lists to clean up final output
        else:
            final_assignments[key].sort()  # Sort channels for readability

    return final_assignments


def run_glm(y, dm, noise_model="ols"):
    """Run the GLM for a given design matrix and data.

    inputs:
        data: xarray.DataArray containing the data (time x chromo x channels)
        dm: design matrix used for the GLM (time x regressor x chromo)
        noise_model: noise model used for the GLM (default = 'ols')
    return:
        thetas: xarray.DataArray estimated parameters of the GLM (regressors x chromo x channels)
    """

    y = y.pint.dequantify()
    dm = dm.pint.dequantify()

    # if y contains chromo dimension, run GLM for each chromophore separately
    if "chromo" in y.dims:
        chromo_results = []
        for chromo in y.chromo:
            thetas_temp = run_glm(
                y.sel(chromo=chromo), dm.sel(chromo=chromo), noise_model=noise_model
            )
            thetas_temp = thetas_temp.assign_coords(chromo=chromo)
            chromo_results.append(thetas_temp)
        return xr.concat(chromo_results, dim="chromo")
    # if y contains only 1 channel, expand dims
    if "channel" not in y.dims:
        labels, glm_est = nilearn_run_glm(
            np.expand_dims(y, 1), dm, noise_model=noise_model
        )
        thetas = xr.DataArray(
            np.squeeze(glm_est[labels[0]].theta),
            dims=("regressor"),
            coords={"regressor": dm.regressor},
        )
    else:
        labels, glm_est = nilearn_run_glm(y, dm, noise_model=noise_model)
        thetas = xr.DataArray(
            glm_est[labels[0]].theta,
            dims=("regressor", "channel"),
            coords={"regressor": dm.regressor, "channel": y.channel},
        )

    return thetas


def get_HRFs(
    predicted_hrf: xr.DataArray,
    stim: pd.DataFrame,
    id_stim: int = 0,
    HRFmin: int = -2,
    HRFmax: int = 15,
):
    """Get HRFs for each condition and channel estimated by the GLM.

    inputs:
        predicted_hrf: xarray.DataArray containing the predicted HRFs (time x chromo x channels)
        stim: pandas.DataFrame containing the stimulus information
        id_stim: id of the stimulus block for which the HRFs are estimated (default = 0)
        HRFmin: minimum relative time of the HRF (default = -2)
        HRFmax: maximum relative time of the HRF (default = 15)
    return:
        hrfs: xarray.DataArray containing HRFs for every condition and every channel (time x chromo x channels x conditions)
    """

    unit = predicted_hrf.pint.units
    predicted_hrf = predicted_hrf.pint.dequantify()

    # get id_stim-th stim onset of each condition
    stim_onsets = stim.groupby("trial_type").onset.nth(id_stim).values
    conds = stim.trial_type.unique()
    stim_onsets = xr.DataArray(
        stim_onsets, dims="condition", coords={"condition": conds}
    )

    # get time axis for HRFs:
    dt = 1 / predicted_hrf.cd.sampling_rate
    time_hrf = np.arange(HRFmin, HRFmax + dt, dt)

    hrfs = xr.DataArray(
        np.zeros(
            (
                time_hrf.size,
                predicted_hrf.chromo.size,
                predicted_hrf.channel.size,
                conds.size,
            )
        ),
        dims=["time", "chromo", "channel", "condition"],
        coords={
            "time": time_hrf,
            "chromo": predicted_hrf.chromo,
            "channel": predicted_hrf.channel,
            "condition": conds,
        },
    )

    for chromo in predicted_hrf.chromo:
        for cond in conds:
            # select HRFs for current chromophore and condition
            hrf = predicted_hrf.sel(
                chromo=chromo,
                time=slice(
                    stim_onsets.sel(condition=cond) + HRFmin,
                    stim_onsets.sel(condition=cond) + HRFmax,
                ),
            )
            # change time axis to relative time
            hrf = hrf.assign_coords(time=time_hrf)
            # store HRFs
            hrfs.loc[{"time": time_hrf, "chromo": chromo, "condition": cond}] = hrf

    # remove baseline
    hrfs = hrfs - hrfs.sel(time=slice(HRFmin, 0)).mean(dim="time")
    # add units
    hrfs = hrfs.pint.quantify(unit)

    return hrfs
