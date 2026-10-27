"""Utilities for GLM-based hemodynamic response estimation."""

from pathlib import Path

import xarray as xr

import cedalion.io
import cedalion.models.glm as glm
import cedalion.nirs
import cedalion.nirs.cw
from cedalion.physunits import parse_quantity


def estimate_hrf_from_glm(
    betas: xr.DataArray,
    covariance: xr.DataArray,
    basis: xr.DataArray,
    trial_type: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Estimate a baseline-corrected HRF and its pointwise variance.

    Args:
        betas: Fitted GLM coefficients with a ``regressor`` dimension.
        covariance: Coefficient covariance matrix with ``regressor_r`` and
            ``regressor_c`` dimensions.
        basis: HRF basis functions with ``time`` and ``component`` dimensions.
        trial_type: Stimulus condition whose HRF coefficients are selected.

    Returns:
        A tuple containing the baseline-corrected HRF estimate and its
        pointwise variance, both as ``xr.DataArray`` objects.

    Notes:
        Coefficient covariance is propagated through the same linear
        baseline correction applied to the HRF estimate.
    """
    components = basis.component.values
    if len(components) == 0:
        raise ValueError("HRF basis must contain at least one component.")

    if len(components) == 1:
        regressors = [f"HRF {trial_type}"]
    else:
        regressors = [f"HRF {trial_type} {component}" for component in components]

    missing = [name for name in regressors if name not in betas.regressor.values]
    if missing:
        raise ValueError(f"Missing HRF regressors: {missing}")

    if not bool((basis.time < 0).any()):
        raise ValueError("HRF basis must include pre-stimulus time points.")

    selected_betas = betas.sel(regressor=regressors)
    selected_covariance = covariance.sel(
        regressor_r=regressors,
        regressor_c=regressors,
    )

    centered_basis = basis - basis.sel(time=basis.time < 0).mean("time")
    centered_basis = centered_basis.rename({"component": "regressor"})
    centered_basis = centered_basis.assign_coords(regressor=regressors)

    hrf = xr.dot(selected_betas, centered_basis, dim="regressor")

    basis_c = centered_basis.rename({"regressor": "regressor_c"})
    basis_r = centered_basis.rename({"regressor": "regressor_r"})

    variance = xr.dot(
        xr.dot(selected_covariance, basis_c, dim="regressor_c"),
        basis_r,
        dim="regressor_r",
    )

    return hrf, variance


def estimate_hrf(
    input_snirf: Path | str,
    output_hrf: Path | str,
    t_pre: str,
    t_post: str,
    t_delta: str,
    t_std: str,
    noise_model: str,
    ts_name: str = "conc",
    trial_types: list[str] | None = None,
    drift_order: int = 3,
    distance_threshold: str = "20 mm",
    use_short_separation: bool = True,
    ar_order: int = 30,
    max_jobs: int = 1,
) -> None:
    """Estimate channel-space HRFs from one preprocessed SNIRF recording.

    Args:
        input_snirf: Path to the preprocessed SNIRF file.
        output_hrf: Destination for the NetCDF results.
        t_pre: Time before stimulus onset included in the HRF basis.
        t_post: Time after stimulus onset included in the HRF basis.
        t_delta: Spacing between Gaussian basis functions.
        t_std: Width parameter of the Gaussian basis functions.
        noise_model: Noise model passed to ``glm.fit``.
        ts_name: Name of a concentration or optical-density time series.
            Optical density is converted to concentration using DPF 1.0
            and the Prahl spectrum, matching preprocessing defaults.
        trial_types: Stimulus conditions to estimate. Defaults to all conditions.
        drift_order: Maximum order of the Legendre drift regressors.
        distance_threshold: Maximum source-detector distance for short channels.
        use_short_separation: Whether to include the mean short-channel regressor.
        ar_order: AR-order parameter passed to ``glm.fit``. For
            ``"ar_irls"``, this is the maximum order considered by BIC.
        max_jobs: Maximum number of parallel GLM fitting jobs.

    Returns:
        None. Writes a NetCDF dataset containing estimated HRFs and their
        pointwise variances.

    Raises:
        ValueError: If requested conditions or required channels are unavailable.
    """
    rec = cedalion.io.read_snirf(input_snirf)[0]
    ts = rec[ts_name]
    input_is_optical_density = "wavelength" in ts.dims

    if input_is_optical_density:
        if not (ts_name == "od" or ts_name.startswith("od_")):
            raise ValueError(
                "Wavelength-based HRF inputs must be optical-density "
                "time series named 'od' or starting with 'od_'."
            )

        dpf = xr.DataArray(
            [1.0] * ts.sizes["wavelength"],
            dims=("wavelength",),
            coords={"wavelength": ts.wavelength.values},
        )
        ts = cedalion.nirs.cw.od2conc(ts, rec.geo3d, dpf, spectrum="prahl")

    available_conditions = list(rec.stim["trial_type"].unique())
    selected_conditions = (
        available_conditions if trial_types is None else list(trial_types)
    )

    if not selected_conditions:
        raise ValueError("At least one stimulus condition is required.")

    missing_conditions = sorted(set(selected_conditions) - set(available_conditions))
    if missing_conditions:
        raise ValueError(f"Stimulus conditions not found: {missing_conditions}")

    stim = rec.stim.loc[rec.stim["trial_type"].isin(selected_conditions)].copy()

    basis_function = glm.GaussianKernels(
        t_pre=parse_quantity(t_pre),
        t_post=parse_quantity(t_post),
        t_delta=parse_quantity(t_delta),
        t_std=parse_quantity(t_std),
    )

    long_ts, short_ts = cedalion.nirs.split_long_short_channels(
        ts,
        rec.geo3d,
        distance_threshold=parse_quantity(distance_threshold),
    )

    if long_ts.sizes["channel"] == 0:
        raise ValueError("No long-separation channels are available.")

    design_matrix = glm.design_matrix.hrf_regressors(
        ts, stim, basis_function
    ) & glm.design_matrix.drift_legendre_regressors(ts, order=drift_order)

    if use_short_separation:
        if short_ts.sizes["channel"] == 0:
            raise ValueError(
                "Short-separation regression was requested, but no "
                "short-separation channels are available."
            )
        design_matrix = (
            design_matrix & glm.design_matrix.average_short_channel_regressor(short_ts)
        )

    results = glm.fit(
        long_ts,
        design_matrix,
        noise_model=noise_model,
        ar_order=ar_order,
        max_jobs=max_jobs,
    )

    betas = results.sm.params
    covariance = results.sm.cov_params()
    basis = basis_function(ts)

    estimates = []
    variances = []

    for condition in selected_conditions:
        hrf, variance = estimate_hrf_from_glm(
            betas=betas,
            covariance=covariance,
            basis=basis,
            trial_type=condition,
        )
        estimates.append(hrf.rename(time="reltime"))
        variances.append(variance.rename(time="reltime"))

    condition_axis = xr.IndexVariable("trial_type", selected_conditions)

    dataset = xr.Dataset(
        {
            "hrf_estimate": xr.concat(estimates, dim=condition_axis),
            "hrf_variance": xr.concat(variances, dim=condition_axis),
        }
    )

    concentration_units = ts.pint.units
    if concentration_units is not None:
        dataset["hrf_estimate"].attrs["units"] = str(concentration_units)
        dataset["hrf_variance"].attrs["units"] = f"({concentration_units}) ** 2"

    dataset.attrs.update(
        source_timeseries=ts_name,
        noise_model=noise_model,
        basis_function="GaussianKernels",
        t_pre=t_pre,
        t_post=t_post,
        t_delta=t_delta,
        t_std=t_std,
        drift_order=drift_order,
        distance_threshold=distance_threshold,
        use_short_separation=str(use_short_separation).lower(),
    )

    if noise_model == "ar_irls":
        dataset.attrs["ar_order_max"] = ar_order

    if input_is_optical_density:
        dataset.attrs["od_to_conc_dpf"] = 1.0
        dataset.attrs["od_to_conc_spectrum"] = "prahl"

    output_path = Path(output_hrf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output_path)
