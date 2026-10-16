import pytest

import cedalion.dot.steps as steps

@pytest.fixture(scope="module")
def tapping_snirf():
    import pooch

    import cedalion.data

    fnames = cedalion.data.DATASETS.fetch(
        "fingertappingDOT.zip", processor=pooch.Unzip()
    )
    return [f for f in fnames if f.endswith(".snirf")][0]


def _register(tmp_path, snirf, **kwargs):
    output_optodes = tmp_path / "sub-01_optodes.tsv"
    steps.register_probe(snirf, output_optodes, head_model="colin27", **kwargs)
    return output_optodes


def _scalp_distance(head, geo3d):
    """Distance of each optode to the scalp surface."""
    import numpy as np
    import pyvista as pv

    import cedalion.dataclasses as cdc

    optodes = geo3d[geo3d.type.isin([cdc.PointType.SOURCE, cdc.PointType.DETECTOR])]
    optodes = optodes.pint.dequantify().values
    scalp = pv.wrap(head.scalp.mesh)
    _, closest = scalp.find_closest_cell(optodes, return_closest_point=True)
    return np.linalg.norm(optodes - closest, axis=1)


@pytest.mark.parametrize(
    "kwargs, expected_crs",
    [
        ({}, "mni"),
        ({"method": "spring_relax"}, "mni"),
        ({"scale_to": "landmarks"}, "pos"),
        (
            {
                "scale_to": "headsize",
                "circumference": "56 cm",
                "nz_cz_iz": "35 cm",
                "lpa_cz_rpa": "36 cm",
            },
            "ellipsoid",
        ),
    ],
)
def test_register_probe(tmp_path, tapping_snirf, kwargs, expected_crs):
    import json

    output_optodes = _register(tmp_path, tapping_snirf, **kwargs)

    assert output_optodes.exists()
    assert (tmp_path / "sub-01_channels.tsv").exists()
    with open(tmp_path / "sub-01_coordsystem.json") as fin:
        registration = json.load(fin)[steps.REGISTRATION_FIELD]
    assert registration["head_model"] == "colin27"
    assert registration["crs"] == expected_crs

    head, geo3d, meas_list = steps._load_registered_probe(output_optodes, "colin27")

    assert head.crs == geo3d.points.crs == expected_crs
    assert head.t_ijk2ras.dims == (expected_crs, "ijk")
    assert {"source", "detector", "wavelength"} <= set(meas_list.columns)
    # optodes stay on the scalp of the registered head
    assert _scalp_distance(head, geo3d).max() < 1e-3


@pytest.mark.parametrize(
    "landmarks, expected",
    [
        (["Nz", "Iz", "LPA", "RPA", "Cz"], "general"),
        (["Nz", "LPA", "RPA"], "trans_rot_isoscale"),
        (["Nz", "HP_1"], "identity"),
    ],
)
def test_auto_initial_align_mode(landmarks, expected):
    import numpy as np

    import cedalion.dataclasses as cdc

    def points(labels):
        return cdc.build_labeled_points(
            np.zeros((len(labels), 3)),
            crs="digitized",
            units="mm",
            labels=labels,
            types=[cdc.PointType.LANDMARK] * len(labels),
        )

    head_landmarks = points(["Nz", "Iz", "LPA", "RPA", "Cz", "Fpz"])
    assert steps._auto_initial_align_mode(points(landmarks), head_landmarks) == expected


def test_register_probe_with_fiducials_only(monkeypatch, tmp_path, tapping_snirf):
    """A probe with only the three fiducials (typical for BIDS) is registered."""
    import cedalion.dataclasses as cdc

    read_snirf = steps.cedalion.io.read_snirf

    def read_snirf_fiducials_only(fname):
        rec = read_snirf(fname)[0]
        geo3d = rec.geo3d.points.rename({"NASION": "Nz"})
        keep = (geo3d.type != cdc.PointType.LANDMARK) | geo3d.label.isin(
            ["Nz", "LPA", "RPA"]
        )
        rec.geo3d = geo3d[keep]
        return [rec]

    monkeypatch.setattr(steps.cedalion.io, "read_snirf", read_snirf_fiducials_only)

    output_optodes = _register(tmp_path, tapping_snirf)
    head, geo3d, _ = steps._load_registered_probe(output_optodes, "colin27")

    assert _scalp_distance(head, geo3d).max() < 1e-3

    with pytest.raises(ValueError, match="trans_rot_isoscale"):
        _register(tmp_path, tapping_snirf, initial_align_mode="identity",
                  scale_to="landmarks")


def test_register_probe_requires_headsize(tmp_path, tapping_snirf):
    with pytest.raises(ValueError, match="circumference"):
        _register(tmp_path, tapping_snirf, scale_to="headsize")


def test_load_registered_probe_warns_on_other_head_model(
    monkeypatch, tmp_path, tapping_snirf
):
    output_optodes = _register(tmp_path, tapping_snirf)
    colin27 = steps.get_standard_headmodel("colin27")
    monkeypatch.setattr(
        steps, "STANDARD_HEAD_MODELS", steps.STANDARD_HEAD_MODELS + ("other",)
    )
    monkeypatch.setattr(steps, "get_standard_headmodel", lambda name: colin27)

    with pytest.warns(UserWarning, match="registered to head model"):
        steps._load_registered_probe(output_optodes, "other")


def test_load_registered_probe_requires_registration(tmp_path, tapping_snirf):
    import json

    output_optodes = _register(tmp_path, tapping_snirf)
    fname = tmp_path / "sub-01_coordsystem.json"
    cs = json.loads(fname.read_text())
    del cs[steps.REGISTRATION_FIELD]
    fname.write_text(json.dumps(cs))

    with pytest.raises(ValueError, match="register_probe"):
        steps._load_registered_probe(output_optodes, "colin27")


def _patch_forward_model(monkeypatch):
    """Patch the forward model used by steps.sensitivity and record its calls."""
    calls = {}

    class FakeForwardModel:
        def __init__(self, head, geometry, meas_list):
            calls["forward_model"] = (head, geometry, meas_list)

        def compute_fluence_nirfaster(self, fname):
            calls["fluence_nirfaster"] = fname

        def compute_fluence_mcx(self, fname):
            calls["fluence_mcx"] = fname

        def compute_sensitivity(self, fluence_fname, sensitivity_fname):
            calls["sensitivity"] = (fluence_fname, sensitivity_fname)

    monkeypatch.setattr(steps, "ForwardModel", FakeForwardModel)

    return calls


def test_sensitivity(monkeypatch, tmp_path, tapping_snirf):
    calls = _patch_forward_model(monkeypatch)

    input_optodes = _register(tmp_path, tapping_snirf, scale_to="landmarks")
    output_fluence = tmp_path / "fluence.h5"
    output_sensitivity = tmp_path / "sensitivity.h5"

    steps.sensitivity(
        input_optodes=input_optodes,
        output_fluence=output_fluence,
        output_sensitivity=output_sensitivity,
        head_model="colin27",
    )

    # the forward model gets the registered head and probe in the same CRS
    head, geo3d, meas_list = calls["forward_model"]
    assert head.crs == geo3d.points.crs == "pos"
    assert len(meas_list) > 0

    assert calls["fluence_nirfaster"] == output_fluence
    assert "fluence_mcx" not in calls
    assert calls["sensitivity"] == (output_fluence, output_sensitivity)


def test_sensitivity_mcx(monkeypatch, tmp_path, tapping_snirf):
    calls = _patch_forward_model(monkeypatch)

    input_optodes = _register(tmp_path, tapping_snirf)
    output_fluence = tmp_path / "fluence.h5"

    steps.sensitivity(
        input_optodes=input_optodes,
        output_fluence=output_fluence,
        output_sensitivity=tmp_path / "sensitivity.h5",
        method="mcx",
    )

    assert calls["fluence_mcx"] == output_fluence
    assert "fluence_nirfaster" not in calls


def test_sensitivity_rejects_unknown_method(monkeypatch, tmp_path, tapping_snirf):
    _patch_forward_model(monkeypatch)
    input_optodes = _register(tmp_path, tapping_snirf)

    with pytest.raises(ValueError, match="nirfaster"):
        steps.sensitivity(
            input_optodes=input_optodes,
            output_fluence=tmp_path / "fluence.h5",
            output_sensitivity=tmp_path / "sensitivity.h5",
            method="unknown",
        )


def test_load_head_model(monkeypatch, tmp_path):
    monkeypatch.setattr(steps, "get_standard_headmodel", lambda name: ("std", name))
    monkeypatch.setattr(
        steps.TwoSurfaceHeadModel, "from_manifest", lambda fname: ("manifest", fname)
    )

    manifest = tmp_path / "headmodel_manifest.json"

    assert steps._load_head_model("colin27") == ("std", "colin27")
    assert steps._load_head_model(str(manifest)) == ("manifest", manifest)
    assert steps._load_head_model(tmp_path) == ("manifest", manifest)


def _patch_image_reconstruction(monkeypatch):
    """Patch the collaborators of steps.image_reconstruction."""
    import numpy as np
    import xarray as xr

    sensitivity = xr.DataArray(
        np.ones((2, 3, 3)),
        dims=("wavelength", "channel", "vertex"),
        coords={
            "wavelength": [760, 850],
            "channel": ["ch1", "ch2", "ch3"],
            "parcel": ("vertex", ["p1", "p2", "p1"]),
        },
    )

    # ch2 represents a channel physically removed during preprocessing.
    od = xr.DataArray(
        np.ones((2, 2, 4)),
        dims=("wavelength", "channel", "time"),
        coords={
            "wavelength": [760, 850],
            "channel": ["ch1", "ch3"],
            "time": [0.0, 1.0, 2.0, 3.0],
        },
    )

    rec = {"od": od}
    calls = {}

    monkeypatch.setattr(
        steps.cedalion.io,
        "read_snirf",
        lambda fname: [rec],
    )
    monkeypatch.setattr(
        steps,
        "load_Adot",
        lambda fname: sensitivity,
    )

    def fake_parcel_sensitivity(
        Adot,
        chan_droplist=None,
        dOD_thresh=0.001,
        **kwargs,
    ):
        calls["parcel_droplist"] = chan_droplist
        calls["dOD_thresh"] = dOD_thresh
        return None, xr.DataArray(
            [True, False],
            dims=("parcel",),
            coords={"parcel": ["p1", "p2"]},
        )

    monkeypatch.setattr(
        steps.ForwardModel,
        "parcel_sensitivity",
        fake_parcel_sensitivity,
    )

    c_meas = xr.DataArray(
        np.ones((2, 2)),
        dims=("wavelength", "channel"),
        coords={
            "wavelength": [760, 850],
            "channel": ["ch1", "ch3"],
        },
    )

    def fake_measurement_variance(ts, calc_covariance=False):
        calls["variance_channels"] = ts.channel.values.tolist()
        calls["calc_covariance"] = calc_covariance
        return c_meas

    monkeypatch.setattr(
        steps,
        "measurement_variance",
        fake_measurement_variance,
    )

    def fake_estimate_alpha_meas(values, K=0.01):
        calls["alpha_values"] = values.copy()
        calls["alpha_k"] = K
        return 7.5

    monkeypatch.setattr(
        steps,
        "estimate_alpha_meas",
        fake_estimate_alpha_meas,
    )

    class FakeImageRecon:
        def __init__(self, Adot, **kwargs):
            calls["recon_channels"] = Adot.channel.values.tolist()
            calls["recon_vertices"] = Adot.vertex.values.tolist()
            calls["recon_kwargs"] = kwargs

        def reconstruct(self, y, c_meas=None):
            calls["input_channels"] = y.channel.values.tolist()
            calls["c_meas"] = c_meas

            return xr.DataArray(
                np.ones((2, 3, 4)),
                dims=("chromo", "vertex", "time"),
                coords={
                    "chromo": ["HbO", "HbR"],
                    "vertex": [0, 1, 2],
                    "time": [0.0, 1.0, 2.0, 3.0],
                    "is_brain": ("vertex", [True, True, False]),
                    "parcel": ("vertex", ["p1", "p2", "p1"]),
                },
            )

    monkeypatch.setattr(steps, "ImageRecon", FakeImageRecon)

    return calls


def test_image_reconstruction(monkeypatch, tmp_path):
    import xarray as xr

    calls = _patch_image_reconstruction(monkeypatch)

    output_image = tmp_path / "image.nc"

    steps.image_reconstruction(
        input_snirf=tmp_path / "input.snirf",
        input_sensitivity=tmp_path / "sensitivity.h5",
        output_image=output_image,
        timeseries="od",
        alpha_meas_k=0.02,
        alpha_spatial=0.001,
        dOD_thresh=0.002,
    )

    assert calls["parcel_droplist"] == ["ch2"]
    assert calls["dOD_thresh"] == 0.002

    assert calls["variance_channels"] == ["ch1", "ch3"]
    assert calls["calc_covariance"] is False

    assert calls["recon_channels"] == ["ch1", "ch3"]
    assert calls["recon_vertices"] == [0, 1, 2]
    assert calls["input_channels"] == ["ch1", "ch3"]

    assert calls["alpha_k"] == 0.02
    assert calls["recon_kwargs"]["alpha_meas"] == 7.5
    assert calls["recon_kwargs"]["alpha_spatial"] == 0.001
    assert calls["recon_kwargs"]["apply_c_meas"] is True
    assert calls["recon_kwargs"]["recon_mode"] == "mua2conc"

    saved = xr.load_dataarray(output_image)

    # By default all reconstructed vertices, brain and scalp, are kept.
    assert saved.vertex.values.tolist() == [0, 1, 2]


def test_image_reconstruction_restrict_to_sensitive_parcels(monkeypatch, tmp_path):
    import xarray as xr

    _patch_image_reconstruction(monkeypatch)

    output_image = tmp_path / "image.nc"

    steps.image_reconstruction(
        input_snirf=tmp_path / "input.snirf",
        input_sensitivity=tmp_path / "sensitivity.h5",
        output_image=output_image,
        timeseries="od",
        restrict_to_sensitive_parcels=True,
    )

    saved = xr.load_dataarray(output_image)

    # vertex 2 is scalp; vertex 1 belongs to insensitive parcel p2.
    assert saved.vertex.values.tolist() == [0]


def test_image_blockaverage_preserves_vertices_and_recovers_time_units(
    monkeypatch, tmp_path
):
    import numpy as np
    import pandas as pd
    import xarray as xr

    image = xr.DataArray(
        np.ones((2, 2, 10)),
        dims=("chromo", "vertex", "time"),
        coords={
            "chromo": ["HbO", "HbR"],
            "vertex": [101, 205],
            "time": np.arange(10, dtype=float),
            "samples": ("time", np.arange(10)),
        },
        attrs={"units": "micromolar"},
    )

    # Deliberately leave image.time without units to exercise the fallback.
    input_image = tmp_path / "image.nc"
    image.to_netcdf(input_image)

    amp = xr.DataArray(
        np.ones(10),
        dims=("time",),
        coords={"time": np.arange(10, dtype=float)},
    )
    amp.time.attrs["units"] = "second"

    stim = pd.DataFrame(
        {
            "onset": [4.0],
            "duration": [1.0],
            "value": [1.0],
            "trial_type": ["motor"],
        }
    )

    class FakeRec:
        def __init__(self):
            self.stim = stim

        def __getitem__(self, key):
            if key == "amp":
                return amp
            raise KeyError(key)

    monkeypatch.setattr(
        steps.cedalion.io,
        "read_snirf",
        lambda fname: [FakeRec()],
    )

    output_image = tmp_path / "image_blockaverage.nc"

    steps.image_blockaverage(
        input_image=input_image,
        input_snirf=tmp_path / "input.snirf",
        output_image=output_image,
        t_pre="2 s",
        t_post="2 s",
        trial_types=["motor"],
    )

    saved = xr.load_dataarray(output_image)

    assert saved.vertex.values.tolist() == [101, 205]
    assert saved.trial_type.values.tolist() == ["motor"]
    assert saved.attrs["units"] == "micromolar"
    assert saved.reltime.attrs["units"] == "second"
    np.testing.assert_allclose(saved.values, 0.0, atol=1e-12)


def test_image_parcel_average(tmp_path):
    import numpy as np
    import xarray as xr

    image = xr.DataArray(
        np.array(
            [
                [
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
                ]
            ]
        ),
        dims=("trial_type", "chromo", "vertex", "reltime"),
        coords={
            "trial_type": ["motor"],
            "chromo": ["HbO", "HbR"],
            "vertex": [101, 205, 310],
            "parcel": ("vertex", ["p1", "p1", "p2"]),
            "reltime": [-1.0, 0.0],
        },
        attrs={"units": "micromolar"},
    )
    image.reltime.attrs["units"] = "second"

    input_image = tmp_path / "image_blockaverage.nc"
    output_image = tmp_path / "image_parcel_average.nc"
    image.to_netcdf(input_image)

    steps.image_parcel_average(
        input_image=input_image,
        output_image=output_image,
    )

    saved = xr.load_dataarray(output_image)

    assert saved.dims == ("trial_type", "chromo", "parcel", "reltime")
    assert saved.parcel.values.tolist() == ["p1", "p2"]
    assert saved.attrs["units"] == "micromolar"
    assert saved.reltime.attrs["units"] == "second"

    np.testing.assert_allclose(
        saved.sel(chromo="HbO", parcel="p1").values,
        [[2.0, 3.0]],
    )
    np.testing.assert_allclose(
        saved.sel(chromo="HbO", parcel="p2").values,
        [[5.0, 6.0]],
    )


def test_image_group_average_aligns_parcels_and_counts_samples(tmp_path):
    import numpy as np
    import xarray as xr

    def make_image(parcels, values):
        image = xr.DataArray(
            np.asarray(values, dtype=float).reshape(1, 1, len(parcels), 2),
            dims=("trial_type", "chromo", "parcel", "reltime"),
            coords={
                "trial_type": ["motor"],
                "chromo": ["HbO"],
                "parcel": parcels,
                "reltime": [-1.0, 0.0],
            },
            attrs={"units": "micromolar"},
        )
        image.reltime.attrs["units"] = "second"
        return image

    # Run 1 has p1 and p2.
    image1 = make_image(
        ["p1", "p2"],
        [
            1.0, 2.0,
            3.0, 4.0,
        ],
    )

    # Run 2 has p2 and p3. p1 is therefore missing from this run.
    image2 = make_image(
        ["p2", "p3"],
        [
            5.0, 6.0,
            7.0, 8.0,
        ],
    )

    input1 = tmp_path / "run1.nc"
    input2 = tmp_path / "run2.nc"
    output = tmp_path / "group.nc"

    image1.to_netcdf(input1)
    image2.to_netcdf(input2)

    steps.image_group_average(
        input_images=[input1, input2],
        output_image=output,
    )

    saved = xr.load_dataset(output)

    assert saved.parcel.values.tolist() == ["p1", "p2", "p3"]
    assert saved["mean"].attrs["units"] == "micromolar"
    assert saved.reltime.attrs["units"] == "second"

    np.testing.assert_allclose(
        saved["mean"].sel(chromo="HbO", parcel="p1").values,
        [[1.0, 2.0]],
    )
    np.testing.assert_allclose(
        saved["mean"].sel(chromo="HbO", parcel="p2").values,
        [[4.0, 5.0]],
    )
    np.testing.assert_allclose(
        saved["mean"].sel(chromo="HbO", parcel="p3").values,
        [[7.0, 8.0]],
    )

    assert saved["n"].sel(chromo="HbO", parcel="p1").values.tolist() == [[1, 1]]
    assert saved["n"].sel(chromo="HbO", parcel="p2").values.tolist() == [[2, 2]]
    assert saved["n"].sel(chromo="HbO", parcel="p3").values.tolist() == [[1, 1]]

    np.testing.assert_allclose(
        saved["sem"].sel(chromo="HbO", parcel="p2").values,
        [[1.0, 1.0]],
    )
    assert np.isnan(
        saved["sem"].sel(chromo="HbO", parcel="p1").values
    ).all()
    assert np.isnan(
        saved["sem"].sel(chromo="HbO", parcel="p3").values
    ).all()


def test_image_group_average_rejects_mismatched_reltime(tmp_path):
    import numpy as np
    import pytest
    import xarray as xr

    def make_image(reltime):
        image = xr.DataArray(
            np.ones((1, 1, 1, 2)),
            dims=("trial_type", "chromo", "parcel", "reltime"),
            coords={
                "trial_type": ["motor"],
                "chromo": ["HbO"],
                "parcel": ["p1"],
                "reltime": reltime,
            },
            attrs={"units": "micromolar"},
        )
        image.reltime.attrs["units"] = "second"
        return image

    input1 = tmp_path / "run1.nc"
    input2 = tmp_path / "run2.nc"
    output = tmp_path / "group.nc"

    make_image([-1.0, 0.0]).to_netcdf(input1)
    make_image([-1.0, 0.1]).to_netcdf(input2)

    with pytest.raises(ValueError, match="reltime"):
        steps.image_group_average(
            input_images=[input1, input2],
            output_image=output,
        )


def test_image_feature_auc_diff_preserves_epochs(monkeypatch, tmp_path):
    import numpy as np
    import pandas as pd
    import xarray as xr

    time = np.arange(12, dtype=float)

    # One parcel with two vertices. HbO-HbR is 0 before the event and
    # 1 micromolar from one second after the event onward.
    values = np.zeros((2, 2, len(time)), dtype=float)
    values[0, :, 5:] = 2.0  # HbO
    values[1, :, 5:] = 1.0  # HbR

    image = xr.DataArray(
        values,
        dims=("chromo", "vertex", "time"),
        coords={
            "chromo": ["HbO", "HbR"],
            "vertex": [101, 205],
            "parcel": ("vertex", ["p1", "p1"]),
            "time": time,
            "samples": ("time", np.arange(len(time))),
        },
        attrs={"units": "micromolar"},
    )
    image.time.attrs["units"] = "second"

    input_image = tmp_path / "image.nc"
    image.to_netcdf(input_image)

    amp = xr.DataArray(
        np.ones(len(time)),
        dims=("time",),
        coords={"time": time},
    )
    amp.time.attrs["units"] = "second"

    stim = pd.DataFrame(
        {
            "onset": [4.0],
            "duration": [1.0],
            "value": [1.0],
            "trial_type": ["motor"],
        }
    )

    class FakeRec:
        def __init__(self):
            self.stim = stim

        def __getitem__(self, key):
            if key == "amp":
                return amp
            raise KeyError(key)

    monkeypatch.setattr(
        steps.cedalion.io,
        "read_snirf",
        lambda fname: [FakeRec()],
    )

    output_feature = tmp_path / "image_feature.nc"

    steps.image_feature(
        input_image=input_image,
        input_snirf=tmp_path / "input.snirf",
        output_feature=output_feature,
        t_pre="2 s",
        t_post="3 s",
        feature="auc_diff",
        baseline_window=["-2 s", "0 s"],
        activity_window=["1 s", "2 s"],
        trial_types=["motor"],
    )

    saved = xr.load_dataarray(output_feature)

    assert saved.dims == ("epoch", "parcel")
    assert saved.parcel.values.tolist() == ["p1"]
    assert saved.trial_type.values.tolist() == ["motor"]

    np.testing.assert_allclose(saved.values, [[1.0]], atol=1e-12)


def test_image_statistics_ttest_and_fdr(tmp_path):
    import numpy as np
    import xarray as xr

    feature = xr.DataArray(
        np.array(
            [
                [2.0, 0.0],
                [3.0, 1.0],
                [4.0, -1.0],
                [5.0, 0.0],
            ]
        ),
        dims=("epoch", "parcel"),
        coords={
            "parcel": ["p1", "p2"],
            "trial_type": ("epoch", ["motor", "motor", "motor", "motor"]),
        },
        attrs={"units": "micromolar * second"},
    )

    input_feature = tmp_path / "feature.nc"
    output_statistics = tmp_path / "statistics.nc"
    feature.to_netcdf(input_feature)

    steps.image_statistics(
        input_feature=input_feature,
        output_statistics=output_statistics,
        alpha=0.05,
        fdr_method="indep",
    )

    saved = xr.load_dataset(output_statistics)

    assert set(saved.data_vars) == {
        "t",
        "p",
        "p_fdr",
        "rejected",
        "n",
    }
    assert saved.t.dims == ("trial_type", "parcel")
    assert saved.trial_type.values.tolist() == ["motor"]
    assert saved.parcel.values.tolist() == ["p1", "p2"]

    np.testing.assert_array_equal(saved.n.values, [[4, 4]])

    # p1 = [2, 3, 4, 5] has mean 3.5 and sample SEM sqrt(5/3)/2,
    # giving t = 5.422176685...
    np.testing.assert_allclose(
        saved.t.sel(trial_type="motor", parcel="p1"),
        5.422176684690384,
    )

    # p2 is symmetric around zero.
    np.testing.assert_allclose(
        saved.t.sel(trial_type="motor", parcel="p2"),
        0.0,
        atol=1e-12,
    )

    assert bool(saved.rejected.sel(trial_type="motor", parcel="p1"))
    assert not bool(saved.rejected.sel(trial_type="motor", parcel="p2"))
    assert (
        saved.p_fdr.sel(trial_type="motor", parcel="p1")
        <= saved.p_fdr.sel(trial_type="motor", parcel="p2")
    )


def test_parcel_statistics_to_vertex_map_masks_nonsignificant():
    import numpy as np
    import xarray as xr

    statistics = xr.Dataset(
        {
            "t": (
                ("trial_type", "parcel"),
                [[2.5, -4.0]],
            ),
            "rejected": (
                ("trial_type", "parcel"),
                [[True, False]],
            ),
        },
        coords={
            "trial_type": ["motor"],
            "parcel": ["p1", "p2"],
        },
    )

    image = xr.DataArray(
        np.zeros((2, 3, 1)),
        dims=("chromo", "vertex", "time"),
        coords={
            "chromo": ["HbO", "HbR"],
            "vertex": [1, 3, 4],
            "parcel": ("vertex", ["p1", "p1", "p2"]),
            "time": [0.0],
        },
    )

    result = steps._parcel_statistics_to_vertex_map(
        statistics=statistics,
        image=image,
        nvertices=6,
        trial_type="motor",
    )

    expected = np.array(
        [np.nan, 2.5, np.nan, 2.5, np.nan, np.nan]
    )
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_parcel_statistics_to_vertex_map_ignores_scalp_vertices():
    import numpy as np
    import xarray as xr

    statistics = xr.Dataset(
        {
            "t": (("trial_type", "parcel"), [[2.5]]),
            "rejected": (("trial_type", "parcel"), [[True]]),
        },
        coords={"trial_type": ["motor"], "parcel": ["p1"]},
    )

    # vertices 0..2 are on the brain, 3..4 on the scalp
    image = xr.DataArray(
        np.zeros((5, 1)),
        dims=("vertex", "time"),
        coords={
            "vertex": [0, 1, 2, 3, 4],
            "parcel": ("vertex", ["p1", "p1", "p1", "scalp", "scalp"]),
            "is_brain": ("vertex", [True, True, True, False, False]),
            "time": [0.0],
        },
    )

    result = steps._parcel_statistics_to_vertex_map(
        statistics=statistics,
        image=image,
        nvertices=3,
        trial_type="motor",
    )

    np.testing.assert_allclose(result, [2.5, 2.5, 2.5])


def test_image_visualization_saves_significant_tmap(monkeypatch, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import xarray as xr

    statistics = xr.Dataset(
        {
            "t": (("trial_type", "parcel"), [[3.0, -2.0]]),
            "rejected": (("trial_type", "parcel"), [[True, False]]),
        },
        coords={
            "trial_type": ["motor"],
            "parcel": ["p1", "p2"],
        },
        attrs={"alpha": 0.01},
    )

    image = xr.DataArray(
        np.zeros((2, 3, 1)),
        dims=("chromo", "vertex", "time"),
        coords={
            "chromo": ["HbO", "HbR"],
            "vertex": [1, 3, 4],
            "parcel": ("vertex", ["p1", "p1", "p2"]),
            "time": [0.0],
        },
    )

    input_statistics = tmp_path / "statistics.nc"
    input_image = tmp_path / "image.nc"
    output_figure = tmp_path / "tmap.png"

    statistics.to_netcdf(input_statistics)
    image.to_netcdf(input_image)

    brain = SimpleNamespace(
        nvertices=6,
        vertices=xr.DataArray(
            np.zeros((6, 3)),
            dims=("label", "ijk"),
        ),
    )
    monkeypatch.setattr(
        steps,
        "get_standard_headmodel",
        lambda name: SimpleNamespace(brain=brain),
    )

    plotted = []

    def fake_plot_surface(plotter, surface, color=None, **kwargs):
        plotted.append(np.asarray(color).copy())

    monkeypatch.setattr(steps, "plot_surface", fake_plot_surface)

    class FakePlotter:
        def __init__(self, *args, **kwargs):
            self.camera_position = None

        def subplot(self, *args):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def screenshot(self, filename):
            Path(filename).touch()

        def close(self):
            pass

    monkeypatch.setattr(steps.pv, "Plotter", FakePlotter)

    steps.image_visualization(
        input_statistics=input_statistics,
        input_image=input_image,
        output_figure=output_figure,
        head_model="colin27",
        trial_type="motor",
    )

    assert output_figure.exists()
    assert len(plotted) == 5

    expected = np.array(
        [np.nan, 3.0, np.nan, 3.0, np.nan, np.nan]
    )
    for vertex_map in plotted:
        np.testing.assert_allclose(
            vertex_map,
            expected,
            equal_nan=True,
        )


def test_image_reconstruction_exposes_reconstruction_parameters(
    monkeypatch, tmp_path
):
    import numpy as np
    import xarray as xr

    sensitivity = xr.DataArray(
        np.ones((2, 1, 2)),
        dims=("wavelength", "channel", "vertex"),
        coords={
            "wavelength": [760, 850],
            "channel": ["ch1"],
            "vertex": [0, 1],
            "is_brain": ("vertex", [True, True]),
            "parcel": ("vertex", ["p1", "p1"]),
        },
    )

    od = xr.DataArray(
        np.ones((2, 1, 4)),
        dims=("wavelength", "channel", "time"),
        coords={
            "wavelength": [760, 850],
            "channel": ["ch1"],
            "time": [0.0, 1.0, 2.0, 3.0],
        },
    )

    monkeypatch.setattr(
        steps.cedalion.io,
        "read_snirf",
        lambda fname: [{"od": od}],
    )
    monkeypatch.setattr(steps, "load_Adot", lambda fname: sensitivity)
    monkeypatch.setattr(
        steps.ForwardModel,
        "parcel_sensitivity",
        lambda *args, **kwargs: (
            None,
            xr.DataArray(
                [True],
                dims=("parcel",),
                coords={"parcel": ["p1"]},
            ),
        ),
    )

    c_meas = xr.DataArray(
        np.ones((2, 1)),
        dims=("wavelength", "channel"),
        coords={"wavelength": [760, 850], "channel": ["ch1"]},
    )
    monkeypatch.setattr(
        steps,
        "measurement_variance",
        lambda *args, **kwargs: c_meas,
    )

    calls = {}

    class FakeImageRecon:
        def __init__(self, Adot, **kwargs):
            calls["kwargs"] = kwargs

        def reconstruct(self, y, c_meas=None):
            calls["c_meas"] = c_meas
            return xr.DataArray(
                np.ones((2, 2, 4)),
                dims=("wavelength", "vertex", "time"),
                coords={
                    "wavelength": [760, 850],
                    "vertex": [0, 1],
                    "time": [0.0, 1.0, 2.0, 3.0],
                    "is_brain": ("vertex", [True, True]),
                    "parcel": ("vertex", ["p1", "p1"]),
                },
            )

    monkeypatch.setattr(steps, "ImageRecon", FakeImageRecon)

    steps.image_reconstruction(
        input_snirf=tmp_path / "input.snirf",
        input_sensitivity=tmp_path / "sensitivity.h5",
        output_image=tmp_path / "image.nc",
        timeseries="od",
        alpha_meas=0.25,
        alpha_spatial=None,
        lambda_R_conc=1e-6,
        apply_c_meas=False,
        recon_mode="mua",
        brain_only=True,
    )

    assert calls["kwargs"]["alpha_meas"] == 0.25
    assert calls["kwargs"]["alpha_spatial"] is None
    assert calls["kwargs"]["lambda_R_conc"] == 1e-6
    assert calls["kwargs"]["apply_c_meas"] is False
    assert calls["kwargs"]["recon_mode"] == "mua"
    assert calls["kwargs"]["brain_only"] is True


@pytest.mark.parametrize("registered", [False, True])
def test_image_reconstruction_gaussian_spatial_basis_functions(
    monkeypatch, tmp_path, registered
):
    import numpy as np
    import xarray as xr

    sensitivity = xr.DataArray(
        np.ones((2, 1, 2)),
        dims=("wavelength", "channel", "vertex"),
        coords={
            "wavelength": [760, 850],
            "channel": ["ch1"],
            "vertex": [0, 1],
            "is_brain": ("vertex", [True, True]),
            "parcel": ("vertex", ["p1", "p1"]),
        },
    )

    od = xr.DataArray(
        np.ones((2, 1, 4)),
        dims=("wavelength", "channel", "time"),
        coords={
            "wavelength": [760, 850],
            "channel": ["ch1"],
            "time": [0.0, 1.0, 2.0, 3.0],
        },
    )

    monkeypatch.setattr(
        steps.cedalion.io,
        "read_snirf",
        lambda fname: [{"od": od}],
    )
    monkeypatch.setattr(steps, "load_Adot", lambda fname: sensitivity)
    monkeypatch.setattr(
        steps.ForwardModel,
        "parcel_sensitivity",
        lambda *args, **kwargs: (
            None,
            xr.DataArray(
                [True],
                dims=("parcel",),
                coords={"parcel": ["p1"]},
            ),
        ),
    )

    c_meas = xr.DataArray(
        np.ones((2, 1)),
        dims=("wavelength", "channel"),
        coords={"wavelength": [760, 850], "channel": ["ch1"]},
    )
    monkeypatch.setattr(
        steps,
        "measurement_variance",
        lambda *args, **kwargs: c_meas,
    )
    monkeypatch.setattr(
        steps,
        "estimate_alpha_meas",
        lambda *args, **kwargs: 0.25,
    )

    fake_head_ras = object()
    calls = {}

    class FakeHead:
        t_ijk2ras = object()

        def apply_transform(self, transform):
            calls["head_transform"] = transform
            return fake_head_ras

    fake_head = FakeHead()

    def fake_get_standard_headmodel(name):
        calls["head_model"] = name
        return fake_head

    monkeypatch.setattr(
        steps,
        "get_standard_headmodel",
        fake_get_standard_headmodel,
    )

    fake_sbf = object()

    def fake_gaussian_sbf(head_model, Adot, **kwargs):
        calls["sbf_head"] = head_model
        calls["sbf_channels"] = Adot.channel.values.tolist()
        calls["sbf_kwargs"] = kwargs
        return fake_sbf

    monkeypatch.setattr(
        steps,
        "GaussianSpatialBasisFunctions",
        fake_gaussian_sbf,
    )

    class FakeImageRecon:
        def __init__(self, Adot, **kwargs):
            calls["recon_kwargs"] = kwargs

        def reconstruct(self, y, c_meas=None):
            return xr.DataArray(
                np.ones((2, 2, 4)),
                dims=("chromo", "vertex", "time"),
                coords={
                    "chromo": ["HbO", "HbR"],
                    "vertex": [0, 1],
                    "time": [0.0, 1.0, 2.0, 3.0],
                    "is_brain": ("vertex", [True, True]),
                    "parcel": ("vertex", ["p1", "p1"]),
                },
            )

    monkeypatch.setattr(steps, "ImageRecon", FakeImageRecon)

    fake_head_registered = object()
    input_optodes = tmp_path / "sub-01_optodes.tsv" if registered else None

    def fake_load_registered_probe(fname, head_model):
        calls["registered"] = (fname, head_model)
        return fake_head_registered, None, None

    monkeypatch.setattr(steps, "_load_registered_probe", fake_load_registered_probe)

    steps.image_reconstruction(
        input_snirf=tmp_path / "input.snirf",
        input_sensitivity=tmp_path / "sensitivity.h5",
        output_image=tmp_path / "image.nc",
        timeseries="od",
        head_model="colin27",
        input_optodes=input_optodes,
        spatial_basis_functions={
            "type": "gaussian",
            "mask_threshold": -2,
            "threshold_brain": "1 mm",
            "threshold_scalp": "5 mm",
            "sigma_brain": "1 mm",
            "sigma_scalp": "5 mm",
            "verbose": False,
        },
    )

    if registered:
        # the basis functions are built on the registered head model
        assert calls["registered"] == (input_optodes, "colin27")
        assert calls["sbf_head"] is fake_head_registered
    else:
        assert "registered" not in calls
        assert calls["head_model"] == "colin27"
        assert calls["head_transform"] is fake_head.t_ijk2ras
        assert calls["sbf_head"] is fake_head_ras
    assert calls["sbf_channels"] == ["ch1"]

    assert calls["sbf_kwargs"]["mask_threshold"] == -2
    assert calls["sbf_kwargs"]["verbose"] is False
    assert calls["sbf_kwargs"]["threshold_brain"].to("mm").magnitude == 1
    assert calls["sbf_kwargs"]["threshold_scalp"].to("mm").magnitude == 5
    assert calls["sbf_kwargs"]["sigma_brain"].to("mm").magnitude == 1
    assert calls["sbf_kwargs"]["sigma_scalp"].to("mm").magnitude == 5

    assert calls["recon_kwargs"]["spatial_basis_functions"] is fake_sbf


def test_image_group_average_visualization_saves_heatmap(tmp_path):
    import numpy as np
    import xarray as xr

    mean = xr.DataArray(
        np.array(
            [
                [
                    [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
                    [[-1.0, -2.0, -3.0], [-2.0, -3.0, -4.0]],
                ]
            ]
        ),
        dims=("trial_type", "chromo", "parcel", "reltime"),
        coords={
            "trial_type": ["motor"],
            "chromo": ["HbO", "HbR"],
            "parcel": ["p1", "p2"],
            "reltime": [-1.0, 0.0, 1.0],
        },
        attrs={"units": "micromolar"},
    )
    mean.reltime.attrs["units"] = "second"

    group = xr.Dataset(
        {
            "mean": mean,
            "sem": xr.zeros_like(mean),
            "n": xr.ones_like(mean, dtype=int),
        }
    )

    input_image = tmp_path / "group.nc"
    output_figure = tmp_path / "group.png"
    group.to_netcdf(input_image)

    steps.image_group_average_visualization(
        input_image=input_image,
        output_figure=output_figure,
        trial_type="motor",
    )

    assert output_figure.exists()
    assert output_figure.stat().st_size > 0


def test_image_reconstruction_without_parcels_skips_parcel_filtering(
    monkeypatch,
    tmp_path,
):
    import numpy as np
    import xarray as xr

    sensitivity = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("wavelength", "channel", "vertex"),
        coords={
            "wavelength": [760],
            "channel": ["ch1", "ch2"],
            "vertex": [0, 1],
            "is_brain": ("vertex", [True, True]),
        },
        attrs={"units": "mm"},
    )

    od = xr.DataArray(
        np.ones((1, 2, 4)),
        dims=("wavelength", "channel", "time"),
        coords={
            "wavelength": [760],
            "channel": ["ch1", "ch2"],
            "time": np.arange(4, dtype=float),
        },
        attrs={"units": "1"},
    )
    od.time.attrs["units"] = "second"

    class FakeRec:
        def __getitem__(self, key):
            assert key == "od"
            return od

    monkeypatch.setattr(
        steps.cedalion.io,
        "read_snirf",
        lambda fname: [FakeRec()],
    )
    monkeypatch.setattr(
        steps,
        "load_Adot",
        lambda fname: sensitivity,
    )

    def fail_parcel_sensitivity(*args, **kwargs):
        raise AssertionError(
            "parcel_sensitivity must not be called without parcel coordinates"
        )

    monkeypatch.setattr(
        steps.ForwardModel,
        "parcel_sensitivity",
        fail_parcel_sensitivity,
    )

    c_meas = xr.DataArray(
        np.ones((2,)),
        dims=("channel",),
        coords={"channel": ["ch1", "ch2"]},
        attrs={"units": "1"},
    )
    monkeypatch.setattr(
        steps,
        "measurement_variance",
        lambda *args, **kwargs: c_meas,
    )

    class FakeRecon:
        def __init__(self, Adot, **kwargs):
            pass

        def reconstruct(self, y, c_meas=None):
            result = xr.DataArray(
                np.ones((1, 2, 4)),
                dims=("chromo", "vertex", "time"),
                coords={
                    "chromo": ["HbO"],
                    "vertex": [0, 1],
                    "is_brain": ("vertex", [True, True]),
                    "time": np.arange(4, dtype=float),
                },
                attrs={"units": "micromolar"},
            )
            result.time.attrs["units"] = "second"
            return result

    monkeypatch.setattr(steps, "ImageRecon", FakeRecon)

    output = tmp_path / "image.nc"

    steps.image_reconstruction(
        input_snirf=tmp_path / "input.snirf",
        input_sensitivity=tmp_path / "sensitivity.h5",
        output_image=output,
        timeseries="od",
        alpha_meas=1.0,
    )

    saved = xr.load_dataarray(output)

    assert saved.sizes["vertex"] == 2
    assert "parcel" not in saved.coords
