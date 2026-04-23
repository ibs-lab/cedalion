import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cedalion
import cedalion.dataclasses as cdc
from cedalion.io import export_to_tsv, load_tsv
from cedalion.testing import temporary_filename


def test_export_to_tsv_measurement_list():
    data = np.array([[1, 2], [3, 4]])
    meas_list = pd.DataFrame({"sourceIndex": data[:, 0], "detectorIndex": data[:, 1]})

    with temporary_filename(suffix=".tsv") as tsv_path:
        export_to_tsv(tsv_path, meas_list)
        content = pd.read_csv(tsv_path, sep="\t")

    assert isinstance(content, pd.DataFrame)
    assert "sourceIndex" in content.columns
    assert "detectorIndex" in content.columns
    assert len(content) == 2


def test_export_to_tsv_other_points():
    # Create mock data

    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    points = cdc.build_labeled_points(
        data, labels=["Fp1", "Fp2", "Cz"], crs="ijk", units="mm"
    )

    # File path to write to
    with temporary_filename(suffix=".tsv") as out_file:
        export_to_tsv(out_file, points)

        # Read and validate output
        with open(out_file, "r") as f:
            content = f.read()
            lines = content.strip().split("\n")

    assert lines[0] == "labels\tX\tY\tZ\tPointType\tcrs=ijk\tunits=mm"
    assert lines[1].startswith("Fp1\t1.000000\t2.000000\t3.000000\tPointType.UNKNOWN")
    assert lines[3].startswith("Cz\t7.000000\t8.000000\t9.000000\tPointType.UNKNOWN")


def test_load_tsv_with_source_index():
    # Create a TSV file with sourceIndex in header

    with temporary_filename(suffix=".tsv") as tsv_path:
        content = "sourceIndex\tdetectorIndex\n0\t1\n1\t2"
        with open(str(tsv_path), "w") as f:
            f.write(content)

        df = load_tsv(tsv_path)

    assert isinstance(df, pd.DataFrame)
    assert "sourceIndex" in df.columns


def test_load_tsv_with_header_and_metadata():
    # Create a TSV with header and metadata crs/units
    with temporary_filename(suffix=".tsv") as tsv_path:
        content = "labels\tX\tY\tZ\tcrs=ijk\tunits=mm\nFp1\t1\t2\t3\nFp2\t4\t5\t6"
        with open(tsv_path, "w") as f:
            f.write(content)

        geo = load_tsv(tsv_path)

    assert isinstance(geo, xr.DataArray)
    assert list(geo.label.values) == ["Fp1", "Fp2"]
    assert geo.points.crs == "ijk"
    assert geo.pint.units == cedalion.units.mm


def test_load_tsv_without_header():
    # No header, just raw data
    with temporary_filename(suffix=".tsv") as tsv_path:
        content = "Fp1\t1\t2\t3\nFp2\t4\t5\t6"
        with open(tsv_path, "w") as f:
            f.write(content)

        geo = load_tsv(tsv_path, crs="RAS", units="mm")

    assert isinstance(geo, xr.DataArray)
    assert list(geo.label.values) == ["Fp1", "Fp2"]
    assert geo.points.crs == "RAS"
    assert geo.pint.units == cedalion.units.mm


def test_load_tsv_point_type():
    # Create a TSV with point type 'Fp1'
    with temporary_filename(suffix=".tsv") as tsv_path:
        content = (
            "labels\tX\tY\tZ\tPointType\n"
            + "U1\t1\t2\t3\tPointType.OBSCURE\n"
            + "S1\t1\t2\t3\tPointType.SOURCE\n"
            + "D1\t4\t5\t6\tPointType.DETECTOR\n"
            + "RPA\t10\t11\t12\tPointType.LANDMARK\n"
            + "Fp1\t7\t8\t9\tPointType.ELECTRODE"
        )

        with open(tsv_path, "w") as f:
            f.write(content)

        geo = load_tsv(tsv_path)

    assert isinstance(geo, xr.DataArray)
    assert list(geo.label.values) == ["U1", "S1", "D1", "RPA", "Fp1"]
    for i in range(5):
        assert geo.type.to_numpy()[i] == cdc.PointType(i)


def test_load_tsv_missing_columns():
    # Missing 'Z' column
    with temporary_filename(suffix=".tsv") as tsv_path:
        content = "labels\tX\tY\nFp1\t1\t2"
        with open(tsv_path, "w") as f:
            f.write(content)

        with pytest.raises(ValueError, match="Missing Z in tsv file"):
            load_tsv(tsv_path)
