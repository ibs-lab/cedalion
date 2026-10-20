import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cedalion.sim.synthetic_hrf as syn
from cedalion import units
from cedalion.dot import get_standard_headmodel


@pytest.fixture
def head_model():
    head_ijk =  get_standard_headmodel("colin27")
    head_ras = head_ijk.apply_transform(head_ijk.t_ijk2ras)
    return head_ras


@pytest.fixture
def head_model_voxel():
    head_ijk = get_standard_headmodel("colin27", kind="voxel")
    head_ras = head_ijk.apply_transform(head_ijk.t_ijk2ras)
    return head_ras



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
        sum_blob_small = blob_small.sum().item()
        sum_blob_big = blob_big.sum().item()
        assert sum_blob_small < sum_blob_big
        seed_pos = head_model.brain.vertices[seed]
        seed_pos = seed_pos.pint.to("mm").pint.dequantify()
        # get all vertices that are within stdev mm of the seed_vertex
        close_vertices_small = head_model.brain.mesh.kdtree.query_ball_point(
            seed_pos, 10
        )
        close_vertices_big = head_model.brain.mesh.kdtree.query_ball_point(seed_pos, 20)
        # check that sum of close vertices is ~ 0.4 of the total sum
        assert (
            0.35 < blob_small[close_vertices_small].sum().item() / sum_blob_small < 0.50
        )
        # check that sum of close vertices is ~ 0.6 of the total sum
        assert 0.55 < blob_big[close_vertices_big].sum().item() / sum_blob_big < 0.65


def test_build_spatial_activation_voxels(head_model_voxel):
    landmarks = ["C2", "C3"]
    seeds = [
        head_model_voxel.brain.kdtree.query(
            head_model_voxel.landmarks.sel(label=landmark)
        )[1]
        for landmark in landmarks
    ]
    scale_small = 10 * units.mm
    scale_big = 2 * units.cm
    intensity_scale = 1 * units.micromolar

    for seed in seeds:
        blob_small = syn.build_spatial_activation_voxels(
            head_model_voxel.brain,
            seed,
            spatial_scale=scale_small,
            intensity_scale=intensity_scale,
            hbr_scale=-0.4,
        )
        blob_big = syn.build_spatial_activation_voxels(
            head_model_voxel.brain,
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
        assert len(blob_small) == len(head_model_voxel.brain.vertices)
        assert np.all(blob_big >= blob_small)
        sum_blob_small = blob_small.sum().item()
        sum_blob_big = blob_big.sum().item()
        assert sum_blob_small < sum_blob_big
        # Peak of the blob is at the seed voxel
        assert int(np.argmax(blob_small.data)) == int(seed)
        assert int(np.argmax(blob_big.data)) == int(seed)

        seed_pos = head_model_voxel.brain.vertices[seed]
        seed_pos = seed_pos.pint.to("mm").pint.dequantify()
        # For an isotropic 3D Gaussian, P(|x-mu| < sigma) ~= 0.199 and
        # P(|x-mu| < 2*sigma) ~= 0.739 (chi-square, dof=3).  Allow a wide
        # tolerance for the discrete brain-voxel grid and clipping near the
        # cortex boundary.
        close_small_1s = head_model_voxel.brain.kdtree.query_ball_point(seed_pos, 10)
        close_small_2s = head_model_voxel.brain.kdtree.query_ball_point(seed_pos, 20)
        frac_1s = blob_small[close_small_1s].sum().item() / sum_blob_small
        frac_2s = blob_small[close_small_2s].sum().item() / sum_blob_small
        assert 0.10 < frac_1s < 0.35
        assert 0.55 < frac_2s < 0.90
        assert frac_2s > frac_1s


def test_build_stim_df():
    max_time = 600 * units.seconds
    trial_types = ["StimA", "StimB"]

    stim_df = syn.build_stim_df(max_time=max_time, trial_types=trial_types)

    assert isinstance(stim_df, pd.DataFrame)
    assert "onset" in stim_df.columns
    assert "duration" in stim_df.columns
    assert "value" in stim_df.columns
    assert "trial_type" in stim_df.columns
