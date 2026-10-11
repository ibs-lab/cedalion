import pytest
import numpy as np
import pandas as pd
import xarray as xr

import cedalion
import cedalion.math.ar_irls
import cedalion.dataclasses as cdc
import cedalion.models.glm as glm
from cedalion.models.glm.basis_functions import (
    AFNIGamma,
    GammaDeriv,
    GaussianKernels,
    GaussianKernelsWithTails,
)

from cedalion import units
from dataclasses import dataclass



@dataclass
class SyntheticData:
    ts: object
    stim: object
    dms: object
    dm: object
    signal_dim: str
    signal_coord: object
    expected_amplitudes: dict
    noise_model: str

rng = np.random.default_rng(0)

@pytest.fixture
def synthetic_rec(request):

    fs = 10.0 * cedalion.units.Hz # sampling rate
    T = 40 * cedalion.units.s # time series length
    chromo = ["HbO", "HbR"] # two chromophores
    nsample = int(T * fs)  # number of samples

    config = request.param
    signal_dim = config["signal_dim"]
    drift_order = config["drift_order"]
    noise_model = config["noise_model"]
    scale_stima = config["scale_stima"]
    offset_stima = config["offset_stima"]
    linear_drift = config["linear_drift"]
    basis_function = config["basis_function"]
    hrf_regressor = config["hrf_regressor"]
    signal_coord = "S1D1"
    signal_coords = [signal_coord]

    # For future generalization once the main GLM code supports non-channel spatial dims:
    # spatial_coord_arg = {signal_dim: signal_coords}

    ts = cdc.build_timeseries(
         rng.normal(0, 0.05, (nsample, len(signal_coords), len(chromo))),
         dims = ["time", "channel", "chromo"],
        # Future general form:
        #  dims = ["time", signal_dim, "chromo"],
         time = np.arange(nsample) / fs.magnitude,
         channel = signal_coords,
        #  **spatial_coord_arg,
         value_units = units.uM,
         time_units = units.s,
         other_coords={"chromo": chromo},
        )


    if signal_dim != "channel":
        ts = ts.rename({"channel": signal_dim})


    stim = pd.DataFrame({
        "onset": [10],
        "trial_type": ["StimA"],
        })
    stim["value"] = [1.0]
    stim["duration"] = 10.

    dms = (
    glm.design_matrix.hrf_regressors(ts, stim, basis_function)
    & glm.design_matrix.drift_regressors(ts, drift_order=drift_order)
)

    dm = dms.common


    signal = (
    scale_stima * dm.sel(regressor=hrf_regressor, chromo="HbO").pint.quantify("uM")
    + offset_stima * cedalion.units.uM
)

    if drift_order >= 1:
        signal += (
            linear_drift
            * dm.sel(regressor="Drift 1", chromo="HbO").pint.quantify("uM")
    )

    ts.loc[{signal_dim: signal_coord, "chromo": "HbO"}] += signal


    expected_amplitudes = {
    hrf_regressor: {
        "HbO": scale_stima,
        "HbR": 0.0
    },
    "Drift 0": {
        "HbO": offset_stima,
        "HbR": 0.0
    },
 }

    if drift_order >= 1:
        expected_amplitudes["Drift 1"] = {
            "HbO": linear_drift,
            "HbR": 0.0,
    }

    if hrf_regressor == "HRF StimA gamma":
        expected_amplitudes["HRF StimA gamma_deriv"] = {
            "HbO": 0.0,
            "HbR": 0.0,
    }


    return SyntheticData(
        ts = ts,
        stim = stim,
        dms = dms,
        dm = dm,
        signal_coord = signal_coord,
        signal_dim = signal_dim,
        expected_amplitudes = expected_amplitudes,
        noise_model = noise_model,
    )


# Each basis config includes the regressor name that receives the injected signal.
# Multi-component bases need an explicit component regressor name.
BASIS_CONFIGS = [
    {
        "basis_function": glm.Gamma(tau=0 * units.s, sigma=5 * units.s),
        "hrf_regressor": "HRF StimA",
    },
    {
        "basis_function": glm.DiracDelta(),
        "hrf_regressor": "HRF StimA",
    },
    {
        "basis_function": AFNIGamma(p=8.6, q=0.547 * units.s),
        "hrf_regressor": "HRF StimA",
    },
    {
        "basis_function": GammaDeriv(tau=0 * units.s, sigma=5 * units.s),
        "hrf_regressor": "HRF StimA gamma",
    },
    {
    "basis_function": GaussianKernels(
        t_pre=0 * units.s,
        t_post=20 * units.s,
        t_delta=10 * units.s,
        t_std=3 * units.s,
    ),
    "hrf_regressor": "HRF StimA 0",
    },
    {
    "basis_function": GaussianKernelsWithTails(
        t_pre=0 * units.s,
        t_post=30 * units.s,
        t_delta=10 * units.s,
        t_std=3 * units.s,
    ),
    "hrf_regressor": "HRF StimA",
    },
]

# SIGNAL_DIMS = ["channel", "parcel", "vertex"]
SIGNAL_DIMS = ["channel"]
DRIFT_ORDERS = [0, 1, 2]

# Only include noise models that currently support this test path.
# "gls" currently fails inside glm.fit with index-alignment errors.
# "rls" returns result objects that are not compatible with result.sm.params.
NOISE_MODELS = ["ols", "glsar", "wls", "ar_irls", "gls", "rls"]

@pytest.mark.parametrize(
    "synthetic_rec",
    [
        {
            "signal_dim": signal_dim,
            "drift_order": drift_order,
            "noise_model": noise_model,
            "basis_function": basis_cfg["basis_function"],
            "hrf_regressor": basis_cfg["hrf_regressor"],
            "scale_stima": 1.25,
            "offset_stima": 0.5,
            "linear_drift": 0.3,
        }
        for signal_dim in SIGNAL_DIMS
        for drift_order in DRIFT_ORDERS
        for noise_model in NOISE_MODELS
        for basis_cfg in BASIS_CONFIGS
    ],
    indirect=True,
)

def test_glm_recovers_known_amplitudes(synthetic_rec):
    """GLM fit should recover injected HRF and drift amplitudes from synthetic data."""
    result = glm.fit(synthetic_rec.ts,
                     synthetic_rec.dms,
                     synthetic_rec.noise_model,
                     max_jobs = 1,
                    )


    params = result.sm.params

    for regressor, chromo_values in synthetic_rec.expected_amplitudes.items():
        for chromo, expected in chromo_values.items():
            beta = float(
                params.sel(
                    {
                        synthetic_rec.signal_dim: synthetic_rec.signal_coord,
                        "chromo": chromo,
                        "regressor": regressor,
                    }
                )
            )
            assert np.isclose(beta, expected, atol=0.2)


# --- masking ---------------------------------------------------------------------

EXPECTED_MASKING = {"HRF StimA": 1.25, "Drift 0": 0.5}
MASKING_SPATIAL_LABELS = {
    "channel": ["S1D1", "S2D1"],
    "parcel": ["A", "B"],
    "vertex": [0, 1],
}


def _masking_rec(spatial_dim="channel"):
    """Two channels (or parcels, vertices) with the same injected HRF and offset."""
    fs = 10.0
    nsample = 600
    ts = cdc.build_timeseries(
        rng.normal(0, 0.05, (nsample, 2, 2)),
        dims=["time", "channel", "chromo"],
        time=np.arange(nsample) / fs,
        channel=["S1D1", "S2D1"],
        value_units=units.uM,
        time_units=units.s,
        other_coords={"chromo": ["HbO", "HbR"]},
    )

    stim = pd.DataFrame(
        {
            "onset": [10.0, 35.0],
            "trial_type": ["StimA", "StimA"],
            "value": [1.0, 1.0],
            "duration": [10.0, 10.0],
        }
    )

    dms = glm.design_matrix.hrf_regressors(
        ts, stim, glm.Gamma(tau=0 * units.s, sigma=5 * units.s)
    ) & glm.design_matrix.drift_regressors(ts, drift_order=0)

    signal = (
        EXPECTED_MASKING["HRF StimA"]
        * dms.common.sel(regressor="HRF StimA").pint.quantify("uM")
        + EXPECTED_MASKING["Drift 0"] * units.uM
    )
    ts = (ts + signal).transpose(*ts.dims)

    if spatial_dim != "channel":
        ts = ts.rename(channel=spatial_dim).assign_coords(
            {spatial_dim: MASKING_SPATIAL_LABELS[spatial_dim]}
        )

    return ts, dms


def _segment(ts, t0, t1):
    return ((ts.time >= t0) & (ts.time < t1)).values


def _corrupt(ts, label, t0, t1, value):
    """Overwrite a HbO segment at one spatial label and return a mask excluding it."""
    spatial_dim = cdc.get_spatial_dimension(ts)
    segment = _segment(ts, t0, t1)
    ts = ts.copy()
    ts.loc[{spatial_dim: label, "chromo": "HbO", "time": segment}] = value
    mask = ~(
        xr.DataArray(segment, dims="time", coords={"time": ts.time})
        & (ts[spatial_dim] == label)
        & (ts.chromo == "HbO")
    )
    return ts, mask


def _assert_expected_masking_params(params):
    for regressor, expected in EXPECTED_MASKING.items():
        np.testing.assert_allclose(
            params.sel(regressor=regressor).values, expected, atol=0.2
        )


@pytest.mark.parametrize("spatial_dim", ["channel", "parcel", "vertex"])
@pytest.mark.parametrize("noise_model", ["ols", "wls", "rls", "gls", "ar_irls"])
def test_glm_mask_excludes_segments(noise_model, spatial_dim):
    labels = MASKING_SPATIAL_LABELS[spatial_dim]
    ts, dms = _masking_rec(spatial_dim)
    ts, mask1 = _corrupt(ts, labels[0], 20, 25, 1e4 * units.uM)
    ts, mask2 = _corrupt(ts, labels[1], 40, 45, np.nan * units.uM)

    result = glm.fit(ts, dms, noise_model, max_jobs=1, mask=mask1 & mask2)

    _assert_expected_masking_params(result.sm.params)


@pytest.mark.parametrize("spatial_dim", ["channel", "parcel", "vertex"])
def test_glm_time_and_spatial_mask_is_broadcast_over_chromo(spatial_dim):
    labels = MASKING_SPATIAL_LABELS[spatial_dim]
    ts, dms = _masking_rec(spatial_dim)
    ts, _ = _corrupt(ts, labels[0], 20, 25, 1e4 * units.uM)

    segment = xr.DataArray(_segment(ts, 20, 25), dims="time", coords={"time": ts.time})
    mask = ~(segment & (ts[spatial_dim] == labels[0]))
    assert set(mask.dims) == {"time", spatial_dim}

    result = glm.fit(ts, dms, "ols", max_jobs=1, mask=mask)

    _assert_expected_masking_params(result.sm.params)


def test_glm_corrupted_segment_biases_unmasked_fit():
    """Sanity check that the artifact in test_glm_mask_excludes_segments matters."""
    ts, dms = _masking_rec()
    ts, mask = _corrupt(ts, "S1D1", 20, 25, 1e4 * units.uM)

    params = glm.fit(ts, dms, "ols", max_jobs=1).sm.params

    deviations = [
        abs(params.sel(channel="S1D1", chromo="HbO", regressor=reg).item() - expected)
        for reg, expected in EXPECTED_MASKING.items()
    ]
    assert max(deviations) > 1


@pytest.mark.parametrize("noise_model", NOISE_MODELS)
def test_glm_all_true_mask_matches_no_mask(noise_model):
    ts, dms = _masking_rec()
    mask = xr.ones_like(ts.pint.dequantify(), dtype=bool)

    params = glm.fit(ts, dms, noise_model, max_jobs=1).sm.params
    params_masked = glm.fit(ts, dms, noise_model, max_jobs=1, mask=mask).sm.params

    np.testing.assert_allclose(params_masked.values, params.values)


def test_glm_time_only_mask_is_broadcast():
    ts, dms = _masking_rec()
    mask_time = xr.DataArray(
        ~_segment(ts, 20, 25), dims="time", coords={"time": ts.time}
    )
    mask_full = mask_time.broadcast_like(ts.pint.dequantify())

    params_time = glm.fit(ts, dms, "ols", max_jobs=1, mask=mask_time).sm.params
    params_full = glm.fit(ts, dms, "ols", max_jobs=1, mask=mask_full).sm.params

    np.testing.assert_allclose(params_time.values, params_full.values)


@pytest.mark.parametrize("spatial_dim", ["channel", "parcel", "vertex"])
def test_glm_unmasked_nans_raise(spatial_dim):
    label = MASKING_SPATIAL_LABELS[spatial_dim][1]
    ts, dms = _masking_rec(spatial_dim)
    ts, _ = _corrupt(ts, label, 40, 45, np.nan * units.uM)

    with pytest.raises(ValueError, match=rf"not masked \({spatial_dim}: {label}\)"):
        glm.fit(ts, dms, "ols", max_jobs=1)

    params = glm.fit(ts, dms, "ols", max_jobs=1, mask=np.isfinite(ts)).sm.params
    _assert_expected_masking_params(params)


def test_glm_design_matrix_must_be_finite():
    ts, dms = _masking_rec()
    dms = dms.copy()
    dms.common.loc[{"regressor": "HRF StimA", "time": dms.common.time[5]}] = np.nan

    with pytest.raises(ValueError, match="design matrix must be finite.*HRF StimA"):
        glm.fit(ts, dms, "ols", max_jobs=1)


def test_glm_invalid_masks_raise():
    ts, dms = _masking_rec()
    mask = xr.ones_like(ts.pint.dequantify(), dtype=bool)

    with pytest.raises(TypeError, match="boolean"):
        glm.fit(ts, dms, "ols", max_jobs=1, mask=mask.astype(int))

    with pytest.raises(ValueError, match="'time' dimension"):
        glm.fit(ts, dms, "ols", max_jobs=1, mask=mask.all("time"))

    with pytest.raises(ValueError, match="wavelength"):
        glm.fit(
            ts,
            dms,
            "ols",
            max_jobs=1,
            mask=mask.expand_dims(wavelength=[760.0, 850.0]),
        )

    with pytest.raises(ValueError, match="coordinates"):
        glm.fit(ts, dms, "ols", max_jobs=1, mask=mask.isel(time=slice(1, None)))

    too_few = mask.copy()
    too_few.loc[{"channel": "S2D1", "chromo": "HbR"}] = False
    too_few[:2, 1, 1] = True
    with pytest.raises(ValueError, match=r"Too few unmasked.*\(channel: S2D1\)"):
        glm.fit(ts, dms, "ols", max_jobs=1, mask=too_few)


def test_glm_glsar_mask():
    ts, dms = _masking_rec()

    leading = xr.DataArray(~_segment(ts, 0, 5), dims="time", coords={"time": ts.time})
    params = glm.fit(ts, dms, "glsar", max_jobs=1, mask=leading).sm.params
    _assert_expected_masking_params(params)

    gap = xr.DataArray(~_segment(ts, 20, 25), dims="time", coords={"time": ts.time})
    with pytest.raises(NotImplementedError, match="glsar"):
        glm.fit(ts, dms, "glsar", max_jobs=1, mask=gap)


@pytest.mark.parametrize("spatial_dim", ["channel", "parcel", "vertex"])
def test_glm_mask_resid_aligned_with_samples(spatial_dim):
    """Residuals of channels with different masks must refer to the same samples."""
    labels = MASKING_SPATIAL_LABELS[spatial_dim]
    ts, dms = _masking_rec(spatial_dim)
    ts, mask1 = _corrupt(ts, labels[0], 20, 25, 1e4 * units.uM)
    ts, mask2 = _corrupt(ts, labels[1], 40, 45, 1e4 * units.uM)
    mask = mask1 & mask2

    result = glm.fit(ts, dms, "ols", max_jobs=1, mask=mask)
    resid = result.sm.resid
    expected = ts.pint.dequantify() - glm.predict(ts, result.sm.params, dms)
    expected = expected.where(mask).transpose(*resid.dims[:2], "time")

    np.testing.assert_allclose(resid.values, expected.values)


def test_valid_after_filter():
    mask = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1], dtype=bool)
    valid = cedalion.math.ar_irls._valid_after_filter(mask, p=2)
    expected = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)

    np.testing.assert_array_equal(valid, expected)
