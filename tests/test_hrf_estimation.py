"""Tests for GLM-based hemodynamic response estimation."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cedalion.dataclasses as cdc
from cedalion import units
from cedalion.sigproc.hrf_estimation import estimate_hrf, estimate_hrf_from_glm


def test_hrf_variance_accounts_for_baseline_correction():
    """Propagate coefficient covariance through baseline subtraction."""
    regressors = ["HRF motor 00", "HRF motor 01"]

    betas = xr.DataArray(
        [[[2.0, 3.0]]],
        dims=("channel", "chromo", "regressor"),
        coords={
            "channel": ["S1D11"],
            "chromo": ["HbO"],
            "regressor": regressors,
        },
    )

    covariance = xr.DataArray(
        [[[[4.0, 0.0], [0.0, 9.0]]]],
        dims=("channel", "chromo", "regressor_r", "regressor_c"),
        coords={
            "channel": ["S1D11"],
            "chromo": ["HbO"],
            "regressor_r": regressors,
            "regressor_c": regressors,
        },
    )

    basis = xr.DataArray(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]],
        dims=("time", "component"),
        coords={
            "time": [-1.0, 0.0, 1.0],
            "component": ["00", "01"],
        },
    )

    hrf, variance = estimate_hrf_from_glm(betas, covariance, basis, trial_type="motor")

    np.testing.assert_allclose(hrf.values.squeeze(), [0.0, 1.0, 5.0])
    np.testing.assert_allclose(variance.values.squeeze(), [0.0, 13.0, 13.0])


def test_hrf_selection_excludes_overlapping_trial_names():
    """Select exact condition regressors and retain covariance terms."""
    regressors = [
        "HRF motor 00",
        "HRF motor 01",
        "HRF motor extra 00",
    ]

    betas = xr.DataArray(
        [[[2.0, 3.0, 100.0]]],
        dims=("channel", "chromo", "regressor"),
        coords={
            "channel": ["S1D11"],
            "chromo": ["HbO"],
            "regressor": regressors,
        },
    )

    covariance = xr.DataArray(
        [[[[4.0, 1.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 25.0]]]],
        dims=("channel", "chromo", "regressor_r", "regressor_c"),
        coords={
            "channel": ["S1D11"],
            "chromo": ["HbO"],
            "regressor_r": regressors,
            "regressor_c": regressors,
        },
    )

    basis = xr.DataArray(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0]],
        dims=("time", "component"),
        coords={"time": [-1.0, 0.0, 1.0], "component": ["00", "01"]},
    )

    hrf, variance = estimate_hrf_from_glm(betas, covariance, basis, trial_type="motor")

    np.testing.assert_allclose(hrf.values.squeeze(), [0.0, 1.0, 5.0])
    np.testing.assert_allclose(variance.values.squeeze(), [0.0, 11.0, 15.0])


def test_single_component_hrf():
    """Reconstruct an HRF with one basis component."""
    betas = xr.DataArray(
        [[[2.0]]],
        dims=("channel", "chromo", "regressor"),
        coords={
            "channel": ["S1D11"],
            "chromo": ["HbO"],
            "regressor": ["HRF motor"],
        },
    )
    covariance = xr.DataArray(
        [[[[9.0]]]],
        dims=("channel", "chromo", "regressor_r", "regressor_c"),
        coords={
            "channel": ["S1D11"],
            "chromo": ["HbO"],
            "regressor_r": ["HRF motor"],
            "regressor_c": ["HRF motor"],
        },
    )
    basis = xr.DataArray(
        [[1.0], [3.0], [5.0]],
        dims=("time", "component"),
        coords={"time": [-1.0, 0.0, 1.0], "component": ["00"]},
    )

    hrf, variance = estimate_hrf_from_glm(betas, covariance, basis, trial_type="motor")

    np.testing.assert_allclose(hrf.values.squeeze(), [0.0, 4.0, 8.0])
    np.testing.assert_allclose(variance.values.squeeze(), [0.0, 36.0, 144.0])

    with pytest.raises(ValueError, match="Missing HRF regressors"):
        estimate_hrf_from_glm(betas, covariance, basis, trial_type="unknown")


@pytest.mark.parametrize("ts_name", ["conc", "od_03"])
def test_estimate_hrf_writes_netcdf(tmp_path, monkeypatch, ts_name):
    """Fit a synthetic recording and serialize the resulting HRF."""
    rng = np.random.default_rng(42)
    recording = cdc.Recording()
    recording["conc"] = cdc.build_timeseries(
        rng.normal(0.0, 0.02, size=(240, 2, 2)),
        dims=["time", "channel", "chromo"],
        time=np.arange(240) / 10.0,
        channel=["long", "short"],
        value_units=units.uM,
        time_units=units.s,
        other_coords={"chromo": ["HbO", "HbR"]},
    )
    recording["od_03"] = cdc.build_timeseries(
        rng.normal(0.0, 0.02, size=(240, 2, 2)),
        dims=["time", "channel", "wavelength"],
        time=np.arange(240) / 10.0,
        channel=["long", "short"],
        value_units=units.dimensionless,
        time_units=units.s,
        other_coords={"wavelength": [760, 850]},
    )

    recording.stim = pd.DataFrame(
        {
            "onset": [5.0, 11.0, 17.0],
            "duration": [1.0, 1.0, 1.0],
            "value": [1.0, 1.0, 1.0],
            "trial_type": ["motor", "motor", "motor"],
        }
    )

    monkeypatch.setattr(
        "cedalion.sigproc.hrf_estimation.cedalion.io.read_snirf",
        lambda path: [recording],
    )
    monkeypatch.setattr(
        "cedalion.sigproc.hrf_estimation.cedalion.nirs.split_long_short_channels",
        lambda ts, geo3d, distance_threshold: (
            ts.sel(channel=["long"]),
            ts.sel(channel=["short"]),
        ),
    )

    conversion_calls = []

    def fake_od2conc(ts, geo3d, dpf, spectrum):
        conversion_calls.append((dpf, spectrum))
        return recording["conc"]

    monkeypatch.setattr(
        "cedalion.sigproc.hrf_estimation.cedalion.nirs.cw.od2conc",
        fake_od2conc,
    )

    output_path = tmp_path / "analysis" / "hrf.nc"
    estimate_hrf(
        input_snirf=tmp_path / "input.snirf",
        output_hrf=output_path,
        t_pre="2 s",
        t_post="4 s",
        t_delta="1 s",
        t_std="1 s",
        noise_model="ols",
        ts_name=ts_name,
        trial_types=["motor"],
        drift_order=1,
        max_jobs=1,
    )

    assert output_path.is_file()

    with xr.open_dataset(output_path) as dataset:
        assert set(dataset.data_vars) == {"hrf_estimate", "hrf_variance"}
        assert dataset.channel.values.tolist() == ["long"]
        assert dataset.chromo.values.tolist() == ["HbO", "HbR"]
        assert dataset.trial_type.values.tolist() == ["motor"]
        assert dataset.attrs["source_timeseries"] == ts_name
        assert dataset.attrs["basis_function"] == "GaussianKernels"
        assert dataset.attrs["t_pre"] == "2 s"
        assert dataset.attrs["t_post"] == "4 s"
        assert dataset.attrs["t_delta"] == "1 s"
        assert dataset.attrs["t_std"] == "1 s"
        assert dataset.attrs["drift_order"] == 1
        assert dataset.attrs["distance_threshold"] == "20 mm"
        assert dataset.attrs["use_short_separation"] == "true"
        assert "ar_order_max" not in dataset.attrs

        if ts_name == "od_03":
            assert dataset.attrs["od_to_conc_dpf"] == 1.0
            assert dataset.attrs["od_to_conc_spectrum"] == "prahl"
        else:
            assert "od_to_conc_dpf" not in dataset.attrs
        assert dataset.hrf_estimate.attrs["units"] == "micromolar"
        assert dataset.hrf_variance.attrs["units"] == "(micromolar) ** 2"
        assert np.isfinite(dataset.hrf_estimate.values).all()
        assert np.isfinite(dataset.hrf_variance.values).all()
        assert (dataset.hrf_variance.values >= -1e-10).all()

    assert len(conversion_calls) == (1 if ts_name == "od_03" else 0)
    if conversion_calls:
        dpf, spectrum = conversion_calls[0]
        np.testing.assert_allclose(dpf.values, [1.0, 1.0])
        assert spectrum == "prahl"


def test_estimate_hrf_rejects_amplitude_input(tmp_path, monkeypatch):
    """Reject amplitude data instead of treating it as optical density."""
    recording = cdc.Recording()
    recording["amp"] = cdc.build_timeseries(
        np.ones((10, 1, 2)),
        dims=["time", "channel", "wavelength"],
        time=np.arange(10) / 10.0,
        channel=["S1D1"],
        value_units=units.dimensionless,
        time_units=units.s,
        other_coords={"wavelength": [760, 850]},
    )

    monkeypatch.setattr(
        "cedalion.sigproc.hrf_estimation.cedalion.io.read_snirf",
        lambda path: [recording],
    )

    with pytest.raises(ValueError, match="must be optical-density"):
        estimate_hrf(
            input_snirf=tmp_path / "input.snirf",
            output_hrf=tmp_path / "hrf.nc",
            t_pre="2 s",
            t_post="10 s",
            t_delta="1 s",
            t_std="1 s",
            noise_model="ols",
            ts_name="amp",
        )
