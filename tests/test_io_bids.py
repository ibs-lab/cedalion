import json

import numpy as np
import pandas as pd
import pytest

import cedalion.dataclasses as cdc
import cedalion.io.bids as bids


@pytest.fixture
def probe():
    geo3d = cdc.build_labeled_points(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.1, 0.2, 0.3]],
        crs="digitized",
        units="cm",
        labels=["S1", "D1", "Nz", "Cz"],
        types=[
            cdc.PointType.SOURCE,
            cdc.PointType.DETECTOR,
            cdc.PointType.LANDMARK,
            cdc.PointType.ELECTRODE,
        ],
    )
    meas_list = pd.DataFrame(
        {
            "channel": ["S1D1", "S1D1"],
            "source": ["S1", "S1"],
            "detector": ["D1", "D1"],
            "wavelength": [760.0, 850.0],
        }
    )
    return geo3d, meas_list


def test_bids_sibling():
    sibling = bids.bids_sibling("a/sub-01_ses-1_optodes.tsv", "channels.tsv")
    assert sibling.as_posix() == "a/sub-01_ses-1_channels.tsv"
    with pytest.raises(ValueError):
        bids.bids_sibling("optodes.tsv", "channels.tsv")


def test_coordsystem_json(tmp_path, probe):
    geo3d, _ = probe
    fname = tmp_path / "sub-01_coordsystem.json"
    bids.export_to_bids_coordsystem_json(fname, geo3d)
    cs = json.loads(fname.read_text())

    assert cs["NIRSCoordinateSystem"] == "Other"
    assert cs["NIRSCoordinateUnits"] == "mm"
    assert "digitized" in cs["NIRSCoordinateSystemDescription"]
    assert list(cs["AnatomicalLandmarkCoordinates"]) == ["Nz", "Cz"]
    np.testing.assert_allclose(cs["AnatomicalLandmarkCoordinates"]["Nz"], [70, 80, 90])
    assert cs["AnatomicalLandmarkCoordinateUnits"] == "mm"


def test_channels_tsv(tmp_path, probe):
    _, meas_list = probe
    fname = tmp_path / "sub-01_channels.tsv"
    bids.export_to_bids_channels_tsv(fname, meas_list)

    df = pd.read_csv(fname, sep="\t", keep_default_na=False)
    assert list(df.columns[:6]) == [
        "name", "type", "source", "detector", "wavelength_nominal", "units"
    ]
    assert list(df.name) == ["S1D1 760", "S1D1 850"]
    assert (df.type == "NIRSCWAMPLITUDE").all()
    assert (df.units == "n/a").all()

    loaded = bids.load_from_bids_channels_tsv(fname)
    pd.testing.assert_frame_equal(
        loaded[["channel", "source", "detector", "wavelength"]],
        meas_list,
        check_dtype=False,
    )


def test_probe_roundtrip(tmp_path, probe):
    geo3d, meas_list = probe
    fname = tmp_path / "sub-01_optodes.tsv"
    bids.export_to_bids_probe(fname, geo3d, meas_list)

    assert (tmp_path / "sub-01_coordsystem.json").exists()
    assert (tmp_path / "sub-01_channels.tsv").exists()

    points, ml = bids.load_from_bids_probe(fname)
    assert list(points.label.values) == ["S1", "D1", "Nz", "Cz"]
    # non-optode points are stored as anatomical landmarks
    assert list(points.type.values) == [
        cdc.PointType.SOURCE,
        cdc.PointType.DETECTOR,
        cdc.PointType.LANDMARK,
        cdc.PointType.LANDMARK,
    ]
    np.testing.assert_allclose(
        points.pint.to("cm").pint.dequantify().values,
        geo3d.pint.dequantify().values,
    )
    assert list(ml.channel) == ["S1D1", "S1D1"]

    with pytest.raises(ValueError, match="_optodes.tsv"):
        bids.export_to_bids_probe(tmp_path / "probe.tsv", geo3d, meas_list)
