"""Cedalion datasets and utility functions."""

import os.path
import pickle
from gzip import GzipFile
from pathlib import Path
from cedalion.imagereco.forward_model import TwoSurfaceHeadModel

import pooch
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.io
from cedalion.io.forward_model import load_fluence

DATASETS = pooch.create(
    path=pooch.os_cache("cedalion"),
    base_url="https://doc.ibs.tu-berlin.de/cedalion/datasets",
    env="CEDALION_DATA_DIR",
    registry={
        "mne_nirsport2_raw.snirf": "sha256:12e5fabe64ecc7ef4b83f6bcd77abb41f5480d5f17a2b1aae0e2ad0406670944",  # noqa: E501
        "colin27_segmentation.zip": "sha256:a51700d53a00cc15e82696c40761c902b56543aa7ff41ae9c4b1281de13273b5",  # noqa: E501
        "colin27_segmentation_downsampled_3x3x3.zip": "sha256:ab98b6bae3ef76be6110dc544917f4f2f7ef7233ac697d9cf8bb4a395e81b6cd",  # noqa: E501
        "fingertapping.zip": "sha256:f2253cca6eef8221d536da54b74d8556b28be93df9143ea53652fdc3bc011875",  # noqa: E501
        "fingertappingDOT.zip": "sha256:03e620479bd48aea8457050b7ce42e0c73ef5448296272448b54cee0e883853e",  # noqa: E501
        "multisubject-fingertapping.zip": "sha256:9949c46ed676e52c385b4c09e3a732f6e742bf745253f4b4208ba678f9a0709b",  # noqa: E501
        "photogrammetry_example_scan.zip": "sha256:f4e4beb32a8217ba9f821edd8b5917a79ee88805a75a84a2aea9fac7b38ccbab",  # noqa: E501
        "colin2SHM.zip": "sha256:7568452d38d80bab91eb4b99c4dd85f3302243ecf9d5cf55afe629502e9d9960",  # noqa: E501
        "ICBM152_2020.zip": "sha256:0721fc4aa3886b8d4af3eb1fbdf74c366e7effdf7503e57bdacfd14edaa429fb",  # noqa: E501
        "fluence_fingertapping_colin27.h5": "sha256:5db30eaaf0dbd614ecefff3734822864b8357841e6c93be78344574889e1d06d",  # noqa:E501
        "fluence_fingertapping_icbm152.h5": "sha256:5b807253e2d0ca0dcc15ac18029cd73404cc9ee589937f2394ae0a2e57dcd98f",  # noqa:E501
        "fluence_fingertappingDOT_colin27.h5": "sha256:f321190e9ab537e0f020cbcca40d9ef909f67ce9c33791be14033daf162acaf7",  # noqa:E501
        "fluence_fingertappingDOT_icbm152.h5": "sha256:4e75e80d906f6c62802d9b39382f34e7546ca1cc7a737e30755666d767e1c697",  # noqa:E501
        "nn22_resting_state.zip": "sha256:0394347af172d906fe33403e84303435af26d82fdcf1d36dad5c7b05beb82d88",  # noqa:E501
    },
)


def get_snirf_test_data():
    fname = DATASETS.fetch("mne_nirsport2_raw.snirf")
    return cedalion.io.read_snirf(fname)


def get_colin27_segmentation(downsampled=False):
    """Retrieves the Colin27 segmentation dataset, based on :cite:t:`Holmes1998`."""
    if downsampled:
        fnames = DATASETS.fetch(
            "colin27_segmentation_downsampled_3x3x3.zip", processor=pooch.Unzip()
        )
        basedir = os.path.commonpath(fnames)

    else:
        fnames = DATASETS.fetch("colin27_segmentation.zip", processor=pooch.Unzip())
        basedir = os.path.join(os.path.commonpath(fnames), "colin27_segmentation")

    mask_files = {
        "csf": "mask_csf.nii",
        "gm": "mask_gray.nii",
        "scalp": "mask_skin.nii",
        "skull": "mask_bone.nii",
        "wm": "mask_white.nii",
    }
    landmarks_ras_file = "landmarks.mrk.json"

    return basedir, mask_files, landmarks_ras_file


def get_colin27_headmodel():
    """Retrieves the Colin27 headmodel, based on :cite:t:`Holmes1998`."""
    fnames = DATASETS.fetch("colin2SHM.zip", processor=pooch.Unzip())
    directory = Path(fnames[0]).parent
    head_model = TwoSurfaceHeadModel.load(directory)
    head_model.brain.units = cedalion.units.mm
    head_model.scalp.units = cedalion.units.mm
    return head_model


def get_icbm152_segmentation():
    fnames = DATASETS.fetch("ICBM152_2020.zip", processor=pooch.Unzip())

    basedir = os.path.dirname(fnames[0])

    mask_files = {
        "csf": "mask_csf.nii",
        "gm": "mask_gray.nii",
        "scalp": "mask_skin.nii",
        "skull": "mask_bone.nii",
        "wm": "mask_white.nii",
    }
    landmarks_ras_file = "landmarks.mrk.json"

    return basedir, mask_files, landmarks_ras_file


def get_fingertapping() -> cdc.Recording:
    """Retrieves a finger tapping recording in BIDS format.

    Data is from :cite:t:`Luke2021`
    """
    fnames = DATASETS.fetch("fingertapping.zip", processor=pooch.Unzip())

    fname = [i for i in fnames if i.endswith(".snirf")][0]

    rec = cedalion.io.read_snirf(fname)[0]

    geo3d = rec.geo3d.points.rename({"NASION": "Nz"})
    geo3d = geo3d.rename({"pos": "digitized"})
    rec.geo3d = geo3d

    amp = rec.get_timeseries("amp")
    amp = amp.pint.dequantify().pint.quantify("V")
    rec.set_timeseries("amp", amp, overwrite=True)

    return rec


def get_fingertappingDOT() -> cdc.Recording:
    """Retrieves a finger tapping DOT example dataset from the IBS Lab."""

    fnames = DATASETS.fetch("fingertappingDOT.zip", processor=pooch.Unzip())

    fname = [i for i in fnames if i.endswith(".snirf")][0]

    rec = cedalion.io.read_snirf(fname)[0]

    geo3d = rec.geo3d.points.rename({"NASION": "Nz"})
    geo3d = geo3d.rename({"pos": "digitized"})
    rec.geo3d = geo3d

    amp = rec.get_timeseries("amp")
    amp = amp.pint.dequantify().pint.quantify("V")
    rec.set_timeseries("amp", amp, overwrite=True)

    return rec


def get_fingertapping_snirf_path() -> Path:
    fnames = DATASETS.fetch("fingertapping.zip", processor=pooch.Unzip())
    fname = [Path(i) for i in fnames if i.endswith(".snirf")][0]
    return fname


def get_multisubject_fingertapping_snirf_paths():
    fnames = DATASETS.fetch("multisubject-fingertapping.zip", processor=pooch.Unzip())
    fnames = sorted([i for i in fnames if i.endswith(".snirf")])
    return fnames


def get_multisubject_fingertapping_path() -> Path:
    fnames = DATASETS.fetch("multisubject-fingertapping.zip", processor=pooch.Unzip())
    return [Path(i).parent for i in fnames if i.endswith("README.md")][0]


def get_photogrammetry_example_scan():
    fnames = DATASETS.fetch("photogrammetry_example_scan.zip", processor=pooch.Unzip())
    fname_scan = [i for i in fnames if i.endswith(".obj")][0]
    fname_snirf = [i for i in fnames if i.endswith(".snirf")][0]
    fname_montage = [i for i in fnames if i.endswith(".png")][0]
    return fname_scan, fname_snirf, fname_montage


def get_imagereco_example_fluence() -> tuple[xr.DataArray, xr.DataArray]:
    fname = DATASETS.fetch("image_reconstruction_fluence.pickle.gz")

    with GzipFile(fname) as fin:
        fluence_all, fluence_at_optodes = pickle.load(fin)

    return fluence_all, fluence_at_optodes


def get_precomputed_fluence(
    dataset: str, head_model: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """Precomputed forward model results for examples and documentation.

    Args:
        dataset: "fingertapping" or "fingertappingDOT"
        head_model: "colin27" or "icbm152"

    Returns:
        fluence_all, fluence_at_optodes
    """

    if dataset not in ["fingertapping", "fingertappingDOT"]:
        raise ValueError(f"unknown dataset {dataset}")
    if head_model not in ["colin27", "icbm152"]:
        raise ValueError(f"unknown head_model {head_model}")

    fname = DATASETS.fetch(f"fluence_{dataset}_{head_model}.h5")

    return load_fluence(fname)


def get_nn22_resting_state() -> cdc.Recording:
    fnames = DATASETS.fetch("nn22_resting_state.zip", processor=pooch.Unzip())
    fname = [Path(i) for i in fnames if i.endswith(".snirf")][0]
    rec = cedalion.io.read_snirf(fname)[0]

    return rec
