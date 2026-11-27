"""Tests for cedalion.io.read_snirf."""

import pytest
import os
import shutil
from pathlib import Path
import h5py
import numpy as np
import cedalion.io
import cedalion.io.snirf
import cedalion.data
from tempfile import TemporaryDirectory

# Edge cases in the handling of snirf files are often discovered in files provided
# by users. Ideally, we assemble a zoo of such edge case files and regularly test
# against them. But we won't get permission to share all of these files. Hence, this
# test looks for an environment variable "SNIRF_ZOO" that points to a local directory.
# The test tries to read all snirf files in it. The test is skipped if the directory is
# not available.

skip_if_snirf_zoo_unavailable = pytest.mark.skipif(
    "SNIRF_ZOO" not in os.environ, reason="snirf zoo not available"
)

testfiles = []

if "SNIRF_ZOO" in os.environ:
    snirf_zoo_dir = Path(os.environ["SNIRF_ZOO"])
    testfiles.extend(sorted(map(str, snirf_zoo_dir.glob("**/*.snirf"))))


@skip_if_snirf_zoo_unavailable
@pytest.mark.parametrize("fname", testfiles)
def test_read_snirf(fname: str):
    cedalion.io.read_snirf(fname)


@skip_if_snirf_zoo_unavailable
@pytest.mark.parametrize("fname", testfiles)
def test_write_snirf(fname):
    recs1 = cedalion.io.read_snirf(fname)

    with TemporaryDirectory() as tmpdirname:
        tmp_fname = Path(tmpdirname) / "test.snirf"
        cedalion.io.snirf.write_snirf(tmp_fname, recs1)

        recs2 = cedalion.io.read_snirf(tmp_fname)

        assert len(recs2) == len(recs1)
        for r1, r2 in zip (recs1, recs2):
            assert (r1.geo3d == r2.geo3d).all().item()




def test_add_number_to_name():
    keys = ["amp"]
    assert cedalion.io.snirf.add_number_to_name("amp", keys) == "amp_02"

    keys = ["amp", "amp_02"]
    assert cedalion.io.snirf.add_number_to_name("amp", keys) == "amp_03"

    keys = ["amp", "od", "od_02", "od_03", "amp_02"]
    assert cedalion.io.snirf.add_number_to_name("amp", keys) == "amp_03"
    assert cedalion.io.snirf.add_number_to_name("od", keys) == "od_04"


def test_read_snirf_crs():
    path = cedalion.data.get_fingertapping_snirf_path()

    rec = cedalion.io.read_snirf(path)[0]
    assert rec.geo3d.points.crs == "pos"

    rec = cedalion.io.read_snirf(path, crs="another_crs")[0]
    assert rec.geo3d.points.crs == "another_crs"

def _replace_dataset(group, name, value):
    del group[name]
    group[name] = value


@pytest.fixture
def mixed_wavelength_snirf(tmp_path):
    """Fingertapping dataset in which channels of source S1 use other wavelengths."""
    fname = tmp_path / "mixed_wavelengths.snirf"
    shutil.copy(cedalion.data.get_fingertapping_snirf_path(), fname)

    with h5py.File(fname, "r+") as f:
        _replace_dataset(
            f["nirs/probe"], "wavelengths", np.array([760.0, 850.0, 690.0, 830.0])
        )
        data = f["nirs/data1"]
        for key in data:
            ml = data[key]
            if key.startswith("measurementList") and ml["sourceIndex"][()] == 1:
                _replace_dataset(ml, "wavelengthIndex", ml["wavelengthIndex"][()] + 2)

    return fname


def test_read_snirf_no_split():
    path = cedalion.data.get_fingertapping_snirf_path()
    rec = cedalion.io.read_snirf(path)[0]

    assert list(rec.timeseries.keys()) == ["amp"]
    assert rec["amp"].wavelength.values.tolist() == [760.0, 850.0]
    assert len(rec._measurement_lists["amp"]) == rec["amp"].sizes["channel"] * 2
    assert not rec["amp"].isnull().any()


def test_read_snirf_split_wavelengths(mixed_wavelength_snirf):
    rec_orig = cedalion.io.read_snirf(cedalion.data.get_fingertapping_snirf_path())[0]
    amp_orig = rec_orig["amp"]

    rec = cedalion.io.read_snirf(mixed_wavelength_snirf)[0]

    assert list(rec.timeseries.keys()) == ["amp_690_830", "amp_760_850"]

    amp_s1 = rec["amp_690_830"]
    amp_other = rec["amp_760_850"]

    assert (amp_s1.source == "S1").all()
    assert not (amp_other.source == "S1").any()
    assert amp_s1.sizes["channel"] + amp_other.sizes["channel"] == (
        amp_orig.sizes["channel"]
    )
    assert amp_s1.wavelength.values.tolist() == [690.0, 830.0]
    assert amp_other.wavelength.values.tolist() == [760.0, 850.0]

    for amp in [amp_s1, amp_other]:
        assert not amp.isnull().any()

    # values are taken from the right columns of the data block
    orig_s1 = amp_orig.sel(channel=amp_s1.channel.values).pint.dequantify()
    assert (amp_s1.pint.dequantify().values == orig_s1.values).all()
    orig_other = amp_orig.sel(channel=amp_other.channel.values).pint.dequantify()
    assert (amp_other.pint.dequantify().values == orig_other.values).all()

    # each time series gets its own subset of the measurement list
    ml_s1 = rec._measurement_lists["amp_690_830"]
    ml_other = rec._measurement_lists["amp_760_850"]
    assert len(ml_s1) == amp_s1.sizes["channel"] * 2
    assert len(ml_other) == amp_other.sizes["channel"] * 2
    assert set(ml_s1.wavelength) == {690.0, 830.0}
    assert set(ml_other.wavelength) == {760.0, 850.0}
    assert sorted(ml_s1.index.tolist() + ml_other.index.tolist()) == list(
        range(amp_orig.sizes["channel"] * 2)
    )


def test_read_snirf_split_chromophores(tmp_path):
    rec = cedalion.io.read_snirf(cedalion.data.get_fingertapping_snirf_path())[0]
    amp = rec["amp"].pint.dequantify()
    conc = amp.rename({"wavelength": "chromo"}).assign_coords(chromo=["HbO", "HbR"])
    conc.attrs["units"] = "micromolar"
    rec.timeseries = {"conc": conc.pint.quantify()}

    fname = tmp_path / "conc.snirf"
    cedalion.io.write_snirf(fname, rec)

    # channels of source S1 provide HbO and HbT instead of HbO and HbR
    with h5py.File(fname, "r+") as f:
        source_labels = [s.decode() for s in f["nirs/probe/sourceLabels"][:]]
        s1_index = source_labels.index("S1") + 1
        data = f["nirs/data1"]
        for key in data:
            ml = data[key]
            if (
                key.startswith("measurementList")
                and ml["sourceIndex"][()] == s1_index
                and ml["dataTypeLabel"][()].decode() == "HbR"
            ):
                _replace_dataset(ml, "dataTypeLabel", "HbT")

    rec2 = cedalion.io.read_snirf(fname)[0]
    assert list(rec2.timeseries.keys()) == ["conc_HbO_HbR", "conc_HbO_HbT"]
    assert (rec2["conc_HbO_HbT"].source == "S1").all()
    assert not (rec2["conc_HbO_HbR"].source == "S1").any()
    assert rec2["conc_HbO_HbT"].chromo.values.tolist() == ["HbO", "HbT"]


def test_write_snirf_split_wavelengths(mixed_wavelength_snirf, tmp_path):
    rec = cedalion.io.read_snirf(mixed_wavelength_snirf)[0]

    fname = tmp_path / "roundtrip.snirf"
    cedalion.io.write_snirf(fname, rec)
    rec2 = cedalion.io.read_snirf(fname)[0]

    assert rec2.wavelengths == [690.0, 760.0, 830.0, 850.0]

    # match time series by their wavelengths
    def by_wavelengths(r):
        return {tuple(ts.wavelength.values): ts for ts in r.timeseries.values()}

    ts1 = by_wavelengths(rec)
    ts2 = by_wavelengths(rec2)
    assert ts1.keys() == ts2.keys()
    for wl in ts1:
        assert (ts1[wl] == ts2[wl]).all()


def test_read_snirf_time_units():
    path = cedalion.data.get_fingertapping_snirf_path()

    rec = cedalion.io.read_snirf(path)[0]
    assert rec["amp"].time.units == cedalion.units.s

    rec = cedalion.io.read_snirf(path, time_units="ms")[0]
    assert rec["amp"].time.units == cedalion.units.ms
