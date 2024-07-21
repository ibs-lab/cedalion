"""Cedalin datasets and utility functions."""

import pooch
import os.path
import cedalion.io
from gzip import GzipFile
import pickle
from pathlib import Path

DATASETS = pooch.create(
    path=pooch.os_cache("cedalion"),
    base_url="https://doc.ml.tu-berlin.de/cedalion/datasets",
    env="CEDALION_DATA_DIR",
    registry={
        "mne_nirsport2_raw.snirf": "sha256:12e5fabe64ecc7ef4b83f6bcd77abb41f5480d5f17a2b1aae0e2ad0406670944",  # noqa: E501
        "colin27_segmentation.zip": "sha256:75adce4289a28640c15ab5efe5f05fbbd6d7d56ac233ac3458974350b3882d18",  # noqa: E501
        "colin27_segmentation_downsampled_3x3x3.zip": "sha256:ab98b6bae3ef76be6110dc544917f4f2f7ef7233ac697d9cf8bb4a395e81b6cd",  # noqa: E501
        "fingertapping.zip": "sha256:f2253cca6eef8221d536da54b74d8556b28be93df9143ea53652fdc3bc011875",  # noqa: E501
        "multisubject-fingertapping.zip": "sha256:9949c46ed676e52c385b4c09e3a732f6e742bf745253f4b4208ba678f9a0709b",  # noqa: E501
        "photogrammetry_example_scan.zip": "sha256:2828b74526cb501a726753881d59fdd362cf5a6c46cbacacbb9d9649d8ce3d64",  # noqa: E501
        "image_reconstruction_fluence.pickle.gz": "sha256:b647c07484a3cc2435b5def7abb342ba7a19aef66f749ed6b3cf3c26deec406f",  # noqa: E501
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


def get_fingertapping():
    fnames = DATASETS.fetch("fingertapping.zip", processor=pooch.Unzip())

    fname = [i for i in fnames if i.endswith(".snirf")][0]

    elements = cedalion.io.read_snirf(fname)
    geo3d = elements[0].geo3d.points.rename({"NASION": "Nz"})
    geo3d = geo3d.rename({"pos": "digitized"})
    elements[0].geo3d = geo3d

    amp = elements[0].data[0]
    amp = amp.pint.dequantify().pint.quantify("V")
    elements[0].data[0] = amp

    return elements


def get_fingertapping_snirf_path():
    fnames = DATASETS.fetch("fingertapping.zip", processor=pooch.Unzip())
    fname = [i for i in fnames if i.endswith(".snirf")][0]
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
    fname = [i for i in fnames if i.endswith(".obj")][0]
    return fname


def get_imagereco_example_fluence():
    fname = DATASETS.fetch("image_reconstruction_fluence.pickle.gz")

    with GzipFile(fname) as fin:
        fluence_all, fluence_at_optodes = pickle.load(fin)

    return fluence_all, fluence_at_optodes
