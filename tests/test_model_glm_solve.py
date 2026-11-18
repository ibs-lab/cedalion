import pytest
import numpy as np
import pandas as pd

import cedalion
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
