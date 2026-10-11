"""Solve the GLM model."""

from __future__ import annotations
from collections import defaultdict

import numpy as np
import xarray as xr

import cedalion.typing as cdt
import cedalion.dataclasses as cdc
import cedalion.xrutils as xrutils
from cedalion import cite
from cedalion.models.glm.design_matrix import DesignMatrix

import statsmodels.api
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed, parallel_config
import cedalion.math.ar_irls


def _channel_fit(y, x, noise_model="ols", ar_order=30, mask=None):
    """Fit a single channel's time series with the specified linear noise model.

    Args:
        y: Dependent variable (pandas Series or 1-D array).
        x: Design matrix (pandas DataFrame or 2-D array).
        noise_model: Regression model to use.  One of ``"ols"``, ``"rls"``,
            ``"wls"``, ``"ar_irls"``, ``"gls"``, ``"glsar"``.
        ar_order: AR model order used by ``"ar_irls"`` and ``"glsar"``.
        mask: Optional 1-D boolean array. Samples that are ``False`` are ignored in
            the fit. If ``None``, all samples are used.

    Returns:
        Fitted statsmodels regression result object.

    Raises:
        ValueError: If ``noise_model`` is not one of the supported options.
        NotImplementedError: If ``noise_model`` is ``"glsar"`` and the mask contains
            gaps between unmasked samples.
    """
    available_models = ["ols", "rls", "wls", "ar_irls", "gls", "glsar"]

    if noise_model not in available_models:
        raise ValueError(
            f"unsupported noise_model '{noise_model}'. Please select one"
            f"of these: {', '.join(available_models)}"
        )

    if mask is None:
        mask = np.ones(len(x), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    if noise_model == "ar_irls":
        return cedalion.math.ar_irls.ar_irls_GLM(y, x, pmax=ar_order, mask=mask)

    # Masked samples are dropped. Labeling y with the index of x lets per-sample
    # results (e.g. resid) keep the positions of the samples in the time series.
    y = pd.Series(np.asarray(y), index=x.index)

    if noise_model == "ols":
        reg_result = statsmodels.api.OLS(y[mask], x[mask]).fit()
    elif noise_model == "rls":
        # The state space model treats NaNs in y as missing observations, which,
        # unlike dropped samples, keeps the time index intact.
        reg_result = statsmodels.api.RecursiveLS(
            y.where(mask),
            x,
            initialization="known",
            initial_state=np.zeros(x.shape[1]),
            initial_state_cov=np.eye(x.shape[1]) * 1e6,
        ).fit()
    elif noise_model == "wls":
        reg_result = statsmodels.api.WLS(y[mask], x[mask]).fit()
    elif noise_model == "gls":
        y, x = y[mask], x[mask]
        idx = np.flatnonzero(mask)
        ols_resid = statsmodels.api.OLS(y, x).fit().resid.values  # need a numpy array
        # estimate the lag-1 autocorrelation only from pairs of adjacent samples
        adjacent = np.diff(idx) == 1
        resid_fit = statsmodels.api.OLS(
            ols_resid[1:][adjacent],
            statsmodels.api.add_constant(ols_resid[:-1][adjacent]),
        ).fit()
        rho = resid_fit.params[1]
        # AR(1) covariance from the lags between the original sample positions,
        # so that gaps left by masked samples are accounted for.
        order = np.abs(idx[:, None] - idx[None, :])
        reg_result = statsmodels.api.GLS(y, x, sigma=rho**order).fit()
    elif noise_model == "glsar":
        idx = np.flatnonzero(mask)
        if idx[-1] - idx[0] + 1 != len(idx):
            raise NotImplementedError(
                "noise_model 'glsar' does not support masks with gaps between "
                "unmasked samples. Use 'ar_irls' instead."
            )
        reg_result = statsmodels.api.GLSAR(y[mask], x[mask], ar_order).iterative_fit(4)
    else:
        raise NotImplementedError()

    return reg_result


def _prepare_mask(ts: cdt.NDTimeSeries, mask: xr.DataArray | None) -> xr.DataArray:
    """Validate the mask passed to :func:`fit` and broadcast it to the shape of ts.

    Args:
        ts: The (dequantified) time series to be modeled.
        mask: Boolean mask or ``None``.

    Returns:
        A boolean mask with the same dimensions and shape as ``ts``.

    Raises:
        TypeError: If the mask is not boolean.
        ValueError: If the mask's dimensions or coordinates don't match ``ts``, or if
            ``ts`` contains non-finite values that are not masked.
    """
    if mask is None:
        mask = xr.ones_like(ts, dtype=bool)
    else:
        if mask.dtype != bool:
            raise TypeError(f"mask must be boolean but has dtype '{mask.dtype}'.")

        if "time" not in mask.dims:
            raise ValueError("mask must have a 'time' dimension.")

        extra_dims = [d for d in mask.dims if d not in ts.dims]
        if extra_dims:
            raise ValueError(
                f"mask has dimensions {extra_dims} that the time series does not "
                f"have. Collapse them first, e.g. with mask.all({extra_dims})."
            )

        try:
            _, mask = xr.align(ts, mask, join="exact")
        except ValueError as e:
            raise ValueError(
                "The coordinates of mask and time series don't match."
            ) from e

        mask = mask.broadcast_like(ts).transpose(*ts.dims)

    unmasked_nonfinite = ~np.isfinite(ts) & mask
    if unmasked_nonfinite.any():
        spatial_dim = cdc.get_spatial_dimension(ts)
        other_dims = [d for d in ts.dims if d != spatial_dim]
        bad = unmasked_nonfinite.any(other_dims)
        bad_labels = ", ".join(map(str, ts[spatial_dim].values[bad.values]))
        raise ValueError(
            "The time series contains NaNs or infs that are not masked "
            f"({spatial_dim}: {bad_labels}). "
            "To exclude them from the fit, pass mask=np.isfinite(ts)."
        )

    return mask


def _check_design_matrix_is_finite(design_matrix: DesignMatrix):
    """Raise a ValueError if any regressor contains NaNs or infs."""

    arrays = [] if design_matrix.common is None else [design_matrix.common]
    arrays.extend(design_matrix.channel_wise)

    for array in arrays:
        array = array.pint.dequantify()
        other_dims = [d for d in array.dims if d != "regressor"]
        bad = (~np.isfinite(array)).any(other_dims)
        if bad.any():
            raise ValueError(
                "The design matrix must be finite, but these regressors contain NaNs "
                f"or infs: {', '.join(map(str, array.regressor.values[bad.values]))}. "
                "To exclude time points from the fit, use the mask argument instead."
            )


def fit(
    ts: cdt.NDTimeSeries,
    design_matrix: DesignMatrix,
    noise_model: str = "ols",
    ar_order: int = 30,
    max_jobs: int = -1,
    verbose: bool = False,
    mask: xr.DataArray | None = None,
):
    """Fit design matrix to data.

    Args:
        ts: the time series to be modeled
        design_matrix: the design matrix. It must not contain NaNs or infs.
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
        mask: optional boolean array that marks which samples of ``ts`` enter the
            fit. ``True`` (:data:`cedalion.sigproc.quality.CLEAN`) means the sample
            is used, ``False`` (``TAINTED``) means it is ignored. The mask must have
            a time dimension. Its dimensions must be a subset of ``ts.dims``, and
            missing dimensions are broadcast: a mask with only a time dimension
            masks the same segments everywhere, a mask with dimensions time and
            channel (or parcel, vertex) and optionally chromo (or wavelength)
            masks segments individually. Masked samples may contain NaNs, unmasked
            samples must be finite. To exclude all NaNs and infs in ``ts``, pass
            ``mask=np.isfinite(ts)``. The design matrix is not masked, so
            :func:`predict` still yields predictions for masked segments.

    Returns:
        thetas as a DataArray

    Raises:
        ValueError: If the design matrix or the unmasked samples of ``ts`` contain
            NaNs or infs, if the mask doesn't match ``ts``, or if a channel (or
            parcel, vertex) has too few unmasked samples to estimate all regressors.

    """

    cite("Barker2013")
    # FIXME: unit handling?
    # shoud the design matrix be dimensionless? -> thetas will have units
    ts = ts.pint.dequantify()

    spatial_dim = cdc.get_spatial_dimension(ts)

    dim3_name = xrutils.other_dim(design_matrix.common, "time", "regressor")

    _check_design_matrix_is_finite(design_matrix)
    mask = _prepare_mask(ts, mask)

    n_regressors = len(design_matrix.regressors)
    too_few = (mask.sum("time") <= n_regressors).any(
        [d for d in mask.dims if d not in ("time", spatial_dim)]
    )
    if too_few.any():
        bad_labels = ", ".join(map(str, ts[spatial_dim].values[too_few.values]))
        raise ValueError(
            f"Too few unmasked samples to estimate {n_regressors} regressors "
            f"({spatial_dim}: {bad_labels})."
        )


    reg_results = xr.DataArray(
        np.empty((ts.sizes[spatial_dim], ts.sizes[dim3_name]), dtype=object),
        dims=(spatial_dim, dim3_name),
        coords=xrutils.coords_from_other(ts.isel(time=0), dims=(spatial_dim, dim3_name))
    )

    for (
        dim3,
        group_channels,
        group_design_matrix,
    ) in design_matrix.iter_computational_groups(ts):

        group_y = ts.sel({spatial_dim: group_channels, dim3_name: dim3}).transpose(
            "time", spatial_dim
        )
        group_mask = mask.sel(
            {spatial_dim: group_channels, dim3_name: dim3}
        ).transpose("time", spatial_dim)
        x = pd.DataFrame(
            group_design_matrix.values, columns=group_design_matrix.regressor.values
        )

        if max_jobs == 1:
            for chan in tqdm(group_y[spatial_dim].values, disable=not verbose):
                result = _channel_fit(
                    group_y.loc[:, chan],
                    x,
                    noise_model,
                    ar_order,
                    group_mask.loc[:, chan].values,
                )
                reg_results.loc[chan, dim3] = result
        else:
            args_list = []
            for chan in group_y[spatial_dim].values:
                args_list.append(
                    [
                        group_y.loc[:, chan],
                        x,
                        noise_model,
                        ar_order,
                        group_mask.loc[:, chan].values,
                    ]
                )

            with parallel_config(backend='threading', n_jobs=max_jobs):
                batch_results = tqdm(
                    Parallel(return_as="generator")(
                        delayed(_channel_fit)(*args) for args in args_list
                    ),
                    total=len(args_list)
                )

            for chan, result in zip(group_y[spatial_dim].values, batch_results):
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

    spatial_dim = cdc.get_spatial_dimension(ts)

    prediction = defaultdict(list)

    for (
        dim3,
        group_channels,
        group_design_matrix,
    ) in design_matrix.iter_computational_groups(ts):
        # (dim3, channel, regressor)
        t = thetas.sel({spatial_dim: group_channels, dim3_name: [dim3]})
        prediction[dim3].append(xr.dot(group_design_matrix, t, dim="regressor"))

    # concatenate channels
    prediction = [xr.concat(v, dim=spatial_dim) for v in prediction.values()]

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
