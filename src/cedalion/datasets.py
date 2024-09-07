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

DATASETS = pooch.create(
    path=pooch.os_cache("cedalion"),
    base_url="https://doc.ibs.tu-berlin.de/cedalion/datasets",
    env="CEDALION_DATA_DIR",
    registry={
        "mne_nirsport2_raw.snirf": "sha256:12e5fabe64ecc7ef4b83f6bcd77abb41f5480d5f17a2b1aae0e2ad0406670944",  # noqa: E501
        "colin27_segmentation.zip": "sha256:75adce4289a28640c15ab5efe5f05fbbd6d7d56ac233ac3458974350b3882d18",  # noqa: E501
        "colin27_segmentation_downsampled_3x3x3.zip": "sha256:ab98b6bae3ef76be6110dc544917f4f2f7ef7233ac697d9cf8bb4a395e81b6cd",  # noqa: E501
        "fingertapping.zip": "sha256:f2253cca6eef8221d536da54b74d8556b28be93df9143ea53652fdc3bc011875",  # noqa: E501
        "fingertappingDOT.zip": "sha256:03e620479bd48aea8457050b7ce42e0c73ef5448296272448b54cee0e883853e",  # noqa: E501
        "multisubject-fingertapping.zip": "sha256:9949c46ed676e52c385b4c09e3a732f6e742bf745253f4b4208ba678f9a0709b",  # noqa: E501
        "photogrammetry_example_scan.zip": "sha256:f4e4beb32a8217ba9f821edd8b5917a79ee88805a75a84a2aea9fac7b38ccbab",  # noqa: E501
        "image_reconstruction_fluence.pickle.gz": "sha256:b647c07484a3cc2435b5def7abb342ba7a19aef66f749ed6b3cf3c26deec406f",  # noqa: E501
        "image_reconstruction_fluence_DOT.pickle.gz": "sha256:44e8e316460a6579ac42c597c953ff050961171303372c06aaad20562aa0fea4",  # noqa: E501
        "colin2SHM.zip": "sha256:7568452d38d80bab91eb4b99c4dd85f3302243ecf9d5cf55afe629502e9d9960",  # noqa: E501
    },
)


def get_snirf_test_data():
    fname = DATASETS.fetch("mne_nirsport2_raw.snirf")
    return cedalion.io.read_snirf(fname)


def get_colin27_segmentation(downsampled=False):
    if downsampled:
        fnames = DATASETS.fetch(
            "colin27_segmentation_downsampled_3x3x3.zip", processor=pooch.Unzip()
        )
    else:
        fnames = DATASETS.fetch("colin27_segmentation.zip", processor=pooch.Unzip())

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


def get_colin27_headmodel():
    fnames = DATASETS.fetch("colin2SHM.zip", processor=pooch.Unzip())
    directory = Path(fnames[0]).parent
    head_model = TwoSurfaceHeadModel.load(directory)
    head_model.brain.units = cedalion.units.mm
    head_model.scalp.units = cedalion.units.mm
    return head_model


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
    """Retrieves a finger tapping DOT example dataset from the IBS Lab.

    """
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

def get_imagereco_example_fluence_DOT() -> tuple[xr.DataArray, xr.DataArray]:
    fname = DATASETS.fetch("image_reconstruction_fluence_DOT.pickle.gz")

    with GzipFile(fname) as fin:
        fluence_all, fluence_at_optodes = pickle.load(fin)

    return fluence_all, fluence_at_optodes
