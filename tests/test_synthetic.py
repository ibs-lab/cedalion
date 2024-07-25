import pytest
import numpy as np
import pandas as pd
import xarray as xr
import cedalion
import cedalion.sim.synthetic_hrf as syn
import cedalion.imagereco.forward_model as fw
from cedalion import units


@pytest.fixture
def head_model():
    # load head model with 60k brain faces
    hm_folder = "/home/thomas/Dokumente/SHK/Data/headmodel/colin2SHM_60k"
    head_model = fw.TwoSurfaceHeadModel.load(hm_folder)
    head_model.brain.units = cedalion.units.mm
    head_model.scalp.units = cedalion.units.mm
    return head_model


@pytest.fixture
def od():
    num_channels = 10
    num_wavelengths = 2
    num_timepoints = 1001
    channel = np.arange(num_channels)
    wavelength = np.array([760.0, 850.0])
    od = xr.DataArray(
        np.random.random((num_channels, num_wavelengths, num_timepoints)),
        dims=["channel", "wavelength", "time"],
        coords={
            "time": np.linspace(0, num_timepoints // 10, num_timepoints),
            "samples": ("time", np.arange(num_timepoints)),
            "channel": channel,
            "wavelength": wavelength,
        },
    )
    return od


@pytest.fixture
def time_axis_hrf(od):
    return od.time.sel(time=(od.time < 20))


def test_generate_hrf(time_axis_hrf):
    stim_dur = 10 * units.seconds
    params_basis = np.random.random(4)
    scale = np.random.random(2)

    hrf = syn.generate_hrf(time_axis_hrf, stim_dur, params_basis, scale)

    assert isinstance(hrf, xr.DataArray)
    assert "time" in hrf.dims
    assert "chromo" in hrf.dims
    assert hrf.sizes["time"] == len(time_axis_hrf)
    assert hrf.sizes["chromo"] == 2


def test_build_blob(head_model):
    landmarks = ["C2", "C3"]
    scale_small = 10 * units.mm
    scale_big = 2 * units.cm
    m = 10.0

    for landmark in landmarks:
        blob_small = syn.build_blob(head_model, landmark, scale_small, m)
        blob_big = syn.build_blob(head_model, landmark, scale_big, m)

        assert isinstance(blob_small, xr.DataArray)
        assert "vertex" in blob_small.dims
        assert len(blob_small) == head_model.brain.mesh.vertices.shape[0]
        assert np.all(blob_small >= 0) and np.all(blob_big <= 1)
        assert np.all(blob_big >= blob_small)
        sum_blob_small = blob_small.sum()
        sum_blob_big = blob_big.sum()
        assert sum_blob_small < sum_blob_big
        seed_lm = head_model.landmarks.sel(label=landmark).pint.dequantify()
        seed_vertex = head_model.brain.mesh.kdtree.query(seed_lm)[1]
        seed_pos = head_model.brain.vertices[seed_vertex]
        # get all vertices that are within stdev mm of the seed_vertex
        close_vertices_small = head_model.brain.mesh.kdtree.query_ball_point(
            seed_pos, 10
        )
        close_vertices_big = head_model.brain.mesh.kdtree.query_ball_point(
            seed_pos, 20
        )
        # check that sum of close vertices is ~ 0.4 of the total sum
        assert 0.35 < blob_small[close_vertices_small].sum() / sum_blob_small < 0.45
        assert 0.35 < blob_big[close_vertices_big].sum() / sum_blob_big < 0.45


def test_hrfs_from_image_reco(head_model, od, time_axis_hrf):
    blob = xr.DataArray(
        np.random.random(len(head_model.brain.mesh.vertices)), dims=["vertex"]
    )
    hrf_model = syn.generate_hrf(time_axis_hrf)

    n_brain = head_model.brain.nvertices
    n_scalp = head_model.scalp.nvertices
    is_brain = np.zeros((n_brain + n_scalp), dtype=bool)
    is_brain[:n_brain] = True
    Adot = xr.DataArray(
        np.random.random(
            (
                len(od.channel),
                n_brain + n_scalp,
                len(od.wavelength),
            )
        ),
        dims=["channel", "vertex", "wavelength"],
        coords={
            "channel": ("channel", od.channel.values),
            "wavelength": ("wavelength", od.wavelength.values),
            "is_brain": ("vertex", is_brain),
        },
    )

    hrfs = syn.hrfs_from_image_reco(blob, hrf_model, Adot)

    assert isinstance(hrfs, xr.DataArray)
    assert "channel" in hrfs.dims
    assert "wavelength" in hrfs.dims
    assert "time" in hrfs.dims


def test_add_hrf_to_vertices():
    num_vertices = 100
    time_axis = xr.DataArray(
        np.linspace(0, 30, 300), dims=["time"], coords={"time": np.linspace(0, 30, 300)}
    )
    hrf_basis = syn.generate_hrf(time_axis)

    scale = xr.DataArray(np.random.random(num_vertices), dims=["vertex"])
    hrf_real_image = syn.add_hrf_to_vertices(hrf_basis, num_vertices, scale)

    assert isinstance(hrf_real_image, xr.DataArray)
    assert "time" in hrf_real_image.dims
    assert "flat_vertex" in hrf_real_image.dims
    assert hrf_real_image.sizes["flat_vertex"] == num_vertices * 2


def test_build_stim_df():
    num_stims = 10
    stim_dur = 10 * units.seconds
    trial_types = ["StimA", "StimB"]
    min_interval = 5 * units.seconds
    max_interval = 10 * units.seconds
    order = "alternating"

    stim_df = syn.build_stim_df(
        num_stims, stim_dur, trial_types, min_interval, max_interval, order
    )

    assert isinstance(stim_df, pd.DataFrame)
    assert "onset" in stim_df.columns
    assert "duration" in stim_df.columns
    assert "value" in stim_df.columns
    assert "trial_type" in stim_df.columns
    assert len(stim_df) == num_stims * len(trial_types)


def test_add_hrf_to_od(od, time_axis_hrf):
    hrfs = xr.DataArray(
        np.random.random((len(od.channel), len(od.wavelength), len(time_axis_hrf))),
        dims=["channel", "wavelength", "time"],
        coords={
            "time": time_axis_hrf.time,
            "samples": time_axis_hrf.samples,
            "channel": od.channel,
            "wavelength": od.wavelength,
        },
    )
    stim_df = pd.DataFrame(
        {
            "onset": [5, 25, 45],
            "duration": [10, 10, 10],
            "value": [1, 1, 1],
            "trial_type": ["Stim", "Stim", "Stim"],
        }
    )

    od_w_hrf = syn.add_hrf_to_od(od, hrfs, stim_df)

    assert isinstance(od_w_hrf, xr.DataArray)
    assert "channel" in od_w_hrf.dims
    assert "wavelength" in od_w_hrf.dims
    assert "time" in od_w_hrf.dims
    assert od_w_hrf.sizes["channel"] == len(od.channel)
    assert od_w_hrf.sizes["wavelength"] == len(od.wavelength)
    assert od_w_hrf.sizes["time"] == len(od.time)
