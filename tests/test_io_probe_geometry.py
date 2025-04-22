import os
import tempfile
import numpy as np
import pytest
import pandas as pd
import xarray as xr
from numpy.testing import assert_array_almost_equal

import cedalion
import cedalion.dataclasses as cdc
from cedalion.io import load_tsv, export_to_tsv



def test_export_to_tsv_measurement_list():
    data = np.array([[1,2], [3,4]])
    meas_list = pd.DataFrame({'sourceIndex': data[:, 0], 'detectorIndex': data[:, 1]})
    out_file = tempfile.NamedTemporaryFile()
    export_to_tsv(out_file.name, meas_list)
    content = pd.read_csv(out_file.name, sep="\t")
    assert isinstance(content, pd.DataFrame)
    assert 'sourceIndex' in content.columns
    assert 'detectorIndex' in content.columns
    assert len(content) == 2


def test_export_to_tsv_other_points():
    # Create mock data
    coords = {"dim_0": [0, 1, 2]}
    data = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]])
    points = cdc.build_labeled_points(data, labels=["Fp1", "Fp2", "Cz"], crs="ijk", units="mm")

    # File path to write to
    out_file = tempfile.NamedTemporaryFile()
    export_to_tsv(out_file.name, points)

    # Read and validate output
    with open(out_file.name, 'r') as f:
        content = f.read()
        lines = content.strip().split('\n')

    assert lines[0] == "labels\tX\tY\tZ\tPointType\tcrs=ijk\tunits=mm"
    assert lines[1].startswith("Fp1\t1.000000\t2.000000\t3.000000\tPointType.UNKNOWN")
    assert lines[3].startswith("Cz\t7.000000\t8.000000\t9.000000\tPointType.UNKNOWN")



def test_load_tsv_with_source_index():
    # Create a TSV file with sourceIndex in header
    tsv_path = tempfile.NamedTemporaryFile()
    content = "sourceIndex\tdetectorIndex\n0\t1\n1\t2"
    with open(str(tsv_path.name), 'w') as f:
        f.write(content)

    df = load_tsv(tsv_path.name)
    assert isinstance(df, pd.DataFrame)
    assert "sourceIndex" in df.columns


def test_load_tsv_with_header_and_metadata():
    # Create a TSV with header and metadata crs/units
    tsv_path = tempfile.NamedTemporaryFile()
    content = "labels\tX\tY\tZ\tcrs=ijk\tunits=mm\nFp1\t1\t2\t3\nFp2\t4\t5\t6"
    with open(tsv_path.name, 'w') as f:
        f.write(content)

    geo = load_tsv(tsv_path.name)
    assert isinstance(geo, xr.DataArray)
    assert list(geo.label.values) == ["Fp1", "Fp2"]
    assert geo.points.crs == "ijk"
    assert geo.pint.units == cedalion.units.mm


def test_load_tsv_without_header():
    # No header, just raw data
    tsv_path = tempfile.NamedTemporaryFile()
    content = "Fp1\t1\t2\t3\nFp2\t4\t5\t6"
    with open(tsv_path.name, 'w') as f:
        f.write(content)

    geo = load_tsv(tsv_path.name, crs="RAS", units="mm")
    assert isinstance(geo, xr.DataArray)
    assert list(geo.label.values) == ["Fp1", "Fp2"]
    assert geo.points.crs == "RAS"
    assert geo.pint.units == cedalion.units.mm


def test_load_tsv_point_type():
    # Create a TSV with point type 'Fp1'
    tsv_path = tempfile.NamedTemporaryFile()
    content = "labels\tX\tY\tZ\tPointType\n" + \
              "U1\t1\t2\t3\tPointType.OBSCURE\n" + \
              "S1\t1\t2\t3\tPointType.SOURCE\n" + \
              "D1\t4\t5\t6\tPointType.DETECTOR\n" + \
              "RPA\t10\t11\t12\tPointType.LANDMARK\n" + \
              "Fp1\t7\t8\t9\tPointType.ELECTRODE"
    with open(tsv_path.name, 'w') as f:
        f.write(content)

    geo = load_tsv(tsv_path.name)
    assert isinstance(geo, xr.DataArray)
    assert list(geo.label.values) == ["U1", "S1", "D1", "RPA", "Fp1"]
    for i in range(5):
        assert geo.type.to_numpy()[i] == cdc.PointType(i)

def test_load_tsv_missing_columns():
    # Missing 'Z' column
    tsv_path = tempfile.NamedTemporaryFile()
    content = "labels\tX\tY\nFp1\t1\t2"
    with open(tsv_path.name, 'w') as f:
        f.write(content)

    with pytest.raises(ValueError, match="Missing Z in tsv file"):
        load_tsv(tsv_path.name)

