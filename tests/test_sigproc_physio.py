import pytest
import cedalion.sigproc.physio as physio
import cedalion.data
from cedalion import units
import xarray as xr
import numpy as np

np.random.seed(42)


@pytest.fixture
def rec():
    rec = cedalion.data.get_snirf_test_data()[0]
    return rec


def test_global_component_subtract(rec):
    ts = rec["amp"]

    ts_weights = ts.sum("time")
    ts_weights[:,:] = np.random.uniform(
        0.01, 1, size=(ts.sizes["channel"], ts.sizes["wavelength"])
    )

    for k in [0, 1, 2]:
        correct, global_comp = physio.global_component_subtract(
            ts, ts_weights=None, k=k
        )

        correct, global_comp = physio.global_component_subtract(
            ts, ts_weights=ts_weights, k=k
        )


def test_compute_Hglobal_from_PCA_warns_for_non_vertex_spatial_dim(rec):
    data = rec["amp"].isel(time=slice(0, 20), channel=slice(0, 4))
    smoothing_kernel = np.eye(data.sizes["channel"], dtype=np.float32)

    with pytest.warns(UserWarning, match="vertex"):
        result = physio.compute_Hglobal_from_PCA(
            data,
            smoothing_kernel,
            spatial_dim="channel",
            spectral_dim="wavelength",
        )

    assert result.dims == data.dims


def test_get_spatial_smoothing_kernel_reuses_distance_array(monkeypatch):
    distances = np.array(
        [
            [0.0, 2.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )

    expected = np.exp(-(distances.copy() ** 2) / 4.0)
    expected[expected < 1e-3] = 0
    expected /= expected.sum(axis=1, keepdims=True)

    monkeypatch.setattr(
        physio,
        "cdist",
        lambda vertices_a, vertices_b: distances,
    )

    vertices = np.zeros((2, 3), dtype=np.float32)
    result = physio.get_spatial_smoothing_kernel(vertices, sigma_mm=2.0)

    assert result is distances
    np.testing.assert_allclose(result, expected)


def test_get_spatial_smoothing_kernel_converts_quantified_positions_to_mm():
    rng = np.random.default_rng(0)
    pos_mm = rng.normal(scale=50.0, size=(6, 3))

    pos_m = xr.DataArray(
        pos_mm / 1000.0, dims=("label", "digitized")
    ).pint.quantify("m")

    kernel_from_m = physio.get_spatial_smoothing_kernel(pos_m, sigma_mm=40.0)
    kernel_from_mm = physio.get_spatial_smoothing_kernel(pos_mm, sigma_mm=40.0)

    np.testing.assert_allclose(kernel_from_m, kernel_from_mm, rtol=1e-5)


def test_compute_Hglobal_from_PCA_equals_spatial_smoothing_of_data():
    rng = np.random.default_rng(1)
    n_time, n_vertex = 200, 15
    data = xr.DataArray(
        rng.normal(size=(n_time, n_vertex, 2)),
        dims=("time", "vertex", "chromo"),
        coords={
            "time": np.arange(n_time) / 10.0,
            "vertex": np.arange(n_vertex),
            "chromo": ["HbO", "HbR"],
        },
    ).pint.quantify("uM")
    kernel = physio.get_spatial_smoothing_kernel(
        rng.normal(scale=50.0, size=(n_vertex, 3)), sigma_mm=60.0
    )

    result = physio.compute_Hglobal_from_PCA(data, kernel)

    expected = np.einsum(
        "tvc,wv->twc", data.pint.dequantify().values, kernel.astype(np.float64)
    )
    assert result.dims == data.dims
    assert result.pint.units == data.pint.units
    np.testing.assert_allclose(
        result.pint.dequantify().values, expected, atol=1e-4
    )


def test_compute_Hglobal_from_PCA_rejects_wrong_kernel_shape():
    data = xr.DataArray(
        np.ones((10, 4, 1)),
        dims=("time", "vertex", "chromo"),
        coords={"chromo": ["HbO"]},
    ).pint.quantify("uM")

    wrong_kernel = np.eye(3)

    with pytest.raises(ValueError, match="smoothing_kernel shape"):
        physio.compute_Hglobal_from_PCA(
            data,
            smoothing_kernel=wrong_kernel,
            spatial_dim="vertex",
            spectral_dim="chromo",
        )


@pytest.mark.parametrize(
    "sigma_mm",
    [0.0, -80.0, np.nan, np.inf, -np.inf],
)
def test_get_spatial_smoothing_kernel_rejects_invalid_sigma(sigma_mm):
    vertices = np.zeros((2, 3))

    with pytest.raises(ValueError, match="sigma_mm"):
        physio.get_spatial_smoothing_kernel(vertices, sigma_mm=sigma_mm)
