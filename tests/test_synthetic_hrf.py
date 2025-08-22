import pytest
import numpy as np
import pandas as pd
import xarray as xr
import cedalion
import cedalion.datasets
from cedalion import units
import cedalion.sim.synthetic_hrf as syn

@pytest.fixture
def head_model():
    return cedalion.datasets.get_colin27_headmodel()


def test_build_spatial_activation(head_model):
    landmarks = ["C2", "C3"]
    seeds = [
        head_model.brain.mesh.kdtree.query(head_model.landmarks.sel(label=landmark))[1]
        for landmark in landmarks
    ]
    scale_small = 10 * units.mm
    scale_big = 2 * units.cm
    intensity_scale = 1 * units.micromolar

    for seed in seeds:
        blob_small = syn.build_spatial_activation(
            head_model.brain,
            seed,
            spatial_scale=scale_small,
            intensity_scale=intensity_scale,
            hbr_scale=-0.4,
        )
        blob_big = syn.build_spatial_activation(
            head_model.brain,
            seed,
            spatial_scale=scale_big,
            intensity_scale=intensity_scale,
            hbr_scale=-0.4,
        )

        assert isinstance(blob_small, xr.DataArray)
        assert "vertex" in blob_small.dims
        assert "chromo" in blob_small.dims
        assert blob_small.pint.units == units.molar
        blob_small = blob_small.sel(chromo="HbO")
        blob_big = blob_big.sel(chromo="HbO")
        assert np.all(blob_small >= 0 * intensity_scale) and np.all(
            blob_big <= intensity_scale
        )
        blob_small = blob_small.pint.dequantify()
        blob_big = blob_big.pint.dequantify()
        assert len(blob_small) == len(head_model.brain.vertices)
        assert np.all(blob_big >= blob_small)
        sum_blob_small = blob_small.sum()
        sum_blob_big = blob_big.sum()
        assert sum_blob_small < sum_blob_big
        seed_pos = head_model.brain.vertices[seed]
        seed_pos = seed_pos.pint.to("mm").pint.dequantify()
        # get all vertices that are within stdev mm of the seed_vertex
        close_vertices_small = head_model.brain.mesh.kdtree.query_ball_point(
            seed_pos, 10
        )
        close_vertices_big = head_model.brain.mesh.kdtree.query_ball_point(seed_pos, 20)
        # check that sum of close vertices is ~ 0.4 of the total sum
        assert 0.35 < blob_small[close_vertices_small].sum() / sum_blob_small < 0.45
        assert 0.35 < blob_big[close_vertices_big].sum() / sum_blob_big < 0.45


def test_build_stim_df():
    max_time = 600 * units.seconds
    trial_types = ["StimA", "StimB"]

    stim_df = syn.build_stim_df(max_time=max_time, trial_types=trial_types)

    assert isinstance(stim_df, pd.DataFrame)
    assert "onset" in stim_df.columns
    assert "duration" in stim_df.columns
    assert "value" in stim_df.columns
    assert "trial_type" in stim_df.columns
