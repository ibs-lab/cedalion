import pytest
import cedalion.sigproc.physio as physio
import cedalion.data
from cedalion import units
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
