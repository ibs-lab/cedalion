"""Solve the GLM model."""

from __future__ import annotations
from collections import defaultdict

import numpy as np
import xarray as xr

import cedalion.typing as cdt
import cedalion.xrutils as xrutils
#import cedalion.dataclasses.statistics
from cedalion.models.glm.design_matrix import DesignMatrix
#import statsmodels.regression
import statsmodels.api
import pandas as pd
from scipy.linalg import toeplitz

from tqdm import tqdm
from joblib import Parallel, delayed, parallel_config
import cedalion.math.ar_irls


def _channel_fit(y, x, noise_model="ols", ar_order=30):
    available_models = ["ols", "rls", "wls", "ar_irls", "gls", "glsar"]

    if noise_model not in available_models:
        raise ValueError(
            f"unsupported noise_model '{noise_model}'. Please select one"
            f"of these: {', '.join(available_models)}"
        )

    if noise_model == "ols":
        reg_result = statsmodels.api.OLS(y, x).fit()
    elif noise_model == "rls":
        reg_result = statsmodels.api.RecursiveLS(y, x).fit()
    elif noise_model == "wls":
        reg_result = statsmodels.api.WLS(y, x).fit()
    elif noise_model == "ar_irls":
        reg_result = cedalion.math.ar_irls.ar_irls_GLM(y, x, pmax=ar_order)
    elif noise_model == "gls":
        ols_resid = statsmodels.api.OLS(y, x).fit().resid
        resid_fit = statsmodels.api.OLS(
            ols_resid[1:],
            statsmodels.api.add_constant(ols_resid[:-1]),
        ).fit()
        rho = resid_fit.params[1]
        order = toeplitz(range(len(ols_resid)))
        reg_result = statsmodels.api.GLS(y, x, sigma=rho**order).fit()
    elif noise_model == "glsar":
        reg_result = statsmodels.api.GLSAR(y, x, ar_order).iterative_fit(4)
    else:
        raise NotImplementedError()

    return reg_result


def fit(
    ts: cdt.NDTimeSeries,
    design_matrix: DesignMatrix,
    noise_model: str = "ols",
    ar_order: int = 30,
    max_jobs: int = -1,
    verbose: bool = False,
):
    """Fit design matrix to data.

    Args:
        ts: the time series to be modeled
        design_matrix: DataArray with dims time, regressor, chromo
        noise_model: specifies the linear regression model

            - ols: ordinary least squares
            - rls: recursive least squares
            - wls: weighted least squares
            - ar_irls: autoregressive iteratively reweighted least squares
              (:cite:t:`Barker2013`)
            - gls: generalized least squares
            - glsar: generalized least squares with autoregressive covariance structure

        ar_order: order of the autoregressive model
        max_jobs: controls the number of jobs in parallel execution. Set to -1 for
            all available cores. Set it to 1 to disable parallel execution.
        verbose: display progress information if True.

    Returns:
        thetas as a DataArray

    """

    # FIXME: unit handling?
    # shoud the design matrix be dimensionless? -> thetas will have units
    ts = ts.pint.dequantify()

    dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")


    reg_results = xr.DataArray(
        np.empty((ts.sizes["channel"], ts.sizes[dim3_name]), dtype=object),
        dims=("channel", dim3_name),
        coords=xrutils.coords_from_other(ts.isel(time=0), dims=("channel", dim3_name))
    )

    for (
        dim3,
        group_channels,
        group_design_matrix,
    ) in design_matrix.iter_computational_groups(ts):
        group_y = ts.sel({"channel": group_channels, dim3_name: dim3}).transpose(
            "time", "channel"
        )

        # pass x as a DataFrame to statsmodel to make it aware of regressor names
        x = pd.DataFrame(
            group_design_matrix.values, columns=group_design_matrix.regressor.values
        )

        if(max_jobs==1):
            for chan in tqdm(group_y.channel.values, disable=not verbose):
                result = _channel_fit(group_y.loc[:, chan], x, noise_model, ar_order)
                reg_results.loc[chan, dim3] = result
        else:
            args_list=[]
            for chan in group_y.channel.values:
                args_list.append([group_y.loc[:, chan], x, noise_model, ar_order])

            with parallel_config(backend='threading', n_jobs=max_jobs):
                batch_results = tqdm(
                    Parallel(return_as="generator")(
                        delayed(_channel_fit)(*args) for args in args_list
                    ),
                    total=len(args_list)
                )

            for chan, result in zip(group_y.channel.values, batch_results):
                reg_results.loc[chan, dim3] = result

    #try:
    #    coloring_matrix=np.linalg.cholesky(np.corrcoef(np.array(resid)))
    #    coloring_matrix=xr.DataArray(data=coloring_matrix,dims=['channel','type'],
    #          coords={'channel':df.channel,'type':df.type})
    #except np.linalg.LinAlgError:
    #    coloring_matrix = None

    description = ""
    if noise_model=='ols':
        description='OLS model via statsmodels.regression'
    elif noise_model=='rls':
        description='Recursive LS model via statsmodels.regression'
    elif noise_model=='gls':
        description='Generalized LS model via statsmodels.regression'
    elif noise_model=='glsar':
        description='Generalized LS AR-model via statsmodels.regression'
    elif noise_model =="ar_irls":
        description='AR_IRLS' # FIXME

    reg_results.attrs["description"] = description

    return reg_results



def predict(
    ts: cdt.NDTimeSeries,
    thetas: xr.DataArray,
    design_matrix: DesignMatrix,
) -> cdt.NDTimeSeries:
    """Predict time series from design matrix and thetas.

    Args:
        ts (cdt.NDTimeSeries): The time series to be modeled.
        thetas (xr.DataArray): The estimated parameters.
        design_matrix (xr.DataArray): DataArray with dims time, regressor, chromo
        channel_wise_regressors (list[xr.DataArray]): Optional list of design matrices,
        with additional channel dimension.

    Returns:
        prediction (xr.DataArray): The predicted time series.
    """

    dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")

    prediction = defaultdict(list)

    for (
        dim3,
        group_channels,
        group_design_matrix,
    ) in design_matrix.iter_computational_groups(ts):
        # (dim3, channel, regressor)
        t = thetas.sel({"channel": group_channels, dim3_name: [dim3]})
        prediction[dim3].append(xr.dot(group_design_matrix, t, dim="regressor"))

    # concatenate channels
    prediction = [xr.concat(v, dim="channel") for v in prediction.values()]

    # concatenate dim3
    prediction = xr.concat(prediction, dim=dim3_name)

    return prediction


def predict_with_uncertainty(
    ts: cdt.NDTimeSeries,
    fit_results: xr.DataArray,
    design_matrix: DesignMatrix,
    regressors: list[str],
    nsample: int = 10,
):
    """Predict time series from design matrix, fitted parameters and uncertainties.

    Using the covariance matrix of the fitted parameters and a multivariate
    normal distribution,  `nsample` set of parameters are sampled around the best
    fit value. From this ensemble of predictions the mean and std are returned.

    Args:
        ts: The time series to be modeled.
        fit_results: xr.DataArray of statsmodels fit results (from glm.fit)
        design_matrix: design matrix, probably from
            glm.design_matrix.hrf_extract_regressors
        regressors: list of regressors to compute
        nsample : the size of the sampled parameter set

    Returns:
        prediction (xr.DataArray): The predicted time series.
    """

    cov = fit_results.sm.cov_params().sel(
        regressor_r=regressors.values,
        regressor_c=regressors.values,
    )

    betas = fit_results.sm.params.sel(regressor=regressors)

    sampled_betas = (
        xr.zeros_like(betas).expand_dims({"sample": nsample}, axis=-1).copy()
    )
    for i_ch in range(sampled_betas.shape[0]):
        for i_cr in range(sampled_betas.shape[1]):
            sampled_betas[i_ch, i_cr, :, :] = np.random.multivariate_normal(
                betas[i_ch, i_cr, :],
                cov[i_ch, i_cr, :, :],
                size=nsample,
            ).T

    pred = predict(ts, sampled_betas, design_matrix)
    pred_mean = pred.mean("sample")
    pred_std = pred.std("sample")

    return pred_mean, pred_std
