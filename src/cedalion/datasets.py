"""Cedalion datasets and utility functions."""

import os.path
import pickle
from gzip import GzipFile
from pathlib import Path
import pandas as pd
from cedalion.imagereco.forward_model import TwoSurfaceHeadModel

import pooch
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.io
from cedalion.io.forward_model import load_Adot

DATASETS = pooch.create(
    path=pooch.os_cache("cedalion"),
    base_url="https://doc.ibs.tu-berlin.de/cedalion/datasets/{version}/",
    env="CEDALION_DATA_DIR",
    version="v25.1.0",
    registry={
        "mne_nirsport2_raw.snirf": "sha256:12e5fabe64ecc7ef4b83f6bcd77abb41f5480d5f17a2b1aae0e2ad0406670944",  # noqa: E501
        "colin27_segmentation.zip": "sha256:ad02dbd9582033d3c6f712761c2a79228e4bf30c8539f23f3280b8c4c7f9ace1",  # noqa: E501
        "colin27_segmentation_downsampled_3x3x3.zip": "sha256:ab98b6bae3ef76be6110dc544917f4f2f7ef7233ac697d9cf8bb4a395e81b6cd",  # noqa: E501
        "fingertapping.zip": "sha256:f2253cca6eef8221d536da54b74d8556b28be93df9143ea53652fdc3bc011875",  # noqa: E501
        "fingertappingDOT.zip": "sha256:03e620479bd48aea8457050b7ce42e0c73ef5448296272448b54cee0e883853e",  # noqa: E501
        "multisubject-fingertapping.zip": "sha256:9949c46ed676e52c385b4c09e3a732f6e742bf745253f4b4208ba678f9a0709b",  # noqa: E501
        "photogrammetry_example_scan.zip": "sha256:f4e4beb32a8217ba9f821edd8b5917a79ee88805a75a84a2aea9fac7b38ccbab",  # noqa: E501
        "colin2SHM.zip": "sha256:7568452d38d80bab91eb4b99c4dd85f3302243ecf9d5cf55afe629502e9d9960",  # noqa: E501
        "ICBM152_2020.zip": "sha256:8dda23aa1f4592d50ba8528bb4ef7124c6593872bddeb9cbd510e7b1891568f3",  # noqa: E501

        "fluence_fingertapping_colin27.h5": "sha256:48befc2297378230ec69411f25dc850956958915e6c3060c3eb18354f186ef29",  # noqa:E501
        "fluence_fingertapping_icbm152.h5": "sha256:39e4a09ab84461b421f28705f52a9e201473e17ac44798e973ab68ad2838069e",  # noqa:E501
        "fluence_fingertappingDOT_colin27.h5": "sha256:2f851a4105c16bc9030b43c04912df28ff889e9aefe9df5fa4bf17918c13ea7b",  # noqa:E501
        "fluence_fingertappingDOT_icbm152.h5": "sha256:9da531414e5acf4b13b9cfd5fab544a42acd96d0471fa047ed56e4900c4c0dcc",  # noqa:E501

        "sensitivity_fingertapping_colin27.nc": "sha256:52c7282c4bff4b9aa9d302535f24c89b4a66bc14144c7705a8b1f26010b0f613",  # noqa:E501
        "sensitivity_fingertapping_icbm152.nc": "sha256:f7769407774b52b887a302d373ca316697e9c1fe2e9b53ea081a5e257f181109",  # noqa:E501
        "sensitivity_fingertappingDOT_colin27.nc": "sha256:6b69f17834d8d790f831e0837aa836e87ff091171a68c5fe2a2b703ea3a566f2",  # noqa:E501
        "sensitivity_fingertappingDOT_icbm152.nc": "sha256:14dae2a398e7031b357e46e30e4488cc4784a724ae618a461ab0ae6a174c5991",  # noqa:E501
        "sensitivity_ninja_cap_56x144_colin27.nc": "sha256:51bca422640345f97a83fbced05934872e1a98cc798ae2613e9bafbe6b010772",  # noqa:E501^
        "sensitivity_ninja_cap_56x144_icbm152.nc": "sha256:d98f6f197600e95b010f1cb9692f480e81055e8b4e633138add770aeb132c0d1",  # noqa:E501^
        "sensitivity_ninja_uhd_cap_164x496_colin27.nc": "sha256:7fd82726b78baf47a6a103bfc049b2d04b5f061d25d737e612ff0646ff1304ea",  # noqa:E501^
        "sensitivity_ninja_uhd_cap_164x496_icbm152.nc": "sha256:29b20249e451f44fc6c8ef7a85d328f085feca21dbe2a34b903bbf04942547ac",  # noqa:E501^
        "sensitivity_nn22_resting_colin27.nc": "sha256:812a67b648a88b3fa3614cdc011dca034b9ea827591cedf4adf549cff67595ca",  # noqa:E501^
        "sensitivity_nn22_resting_icbm152.nc": "sha256:4e8e8e16f167241835f4a8cad408dc7a0144401fe7fb3228248d2261d650c797",  # noqa:E501^

        "nn22_resting_state.zip": "sha256:0394347af172d906fe33403e84303435af26d82fdcf1d36dad5c7b05beb82d88",  # noqa:E501
        "colin27_parcellation.zip": "sha256:70cb51cc587b7a7389050b854beede76327ed8b105fa12971584a7d1bb7fa080",  # noqa:E501
        "icbm152_parcellation.zip": "sha256:b69ffdb3ff2fe3d85a6d5c139e59147d05ca97127589c1e4c2a8d031850f0148",  # noqa:E501
    },
)

DATA_DIR = Path(__file__).parent / "data"

def get_ninja_cap_probe():
    """Load the fullhead Ninja NIRS cap probe."""
    probe_dir = DATA_DIR / 'ninja_cap_probe'
    raw_fn = 'fullhead_56x144_System2'
    geo3d = cedalion.io.load_tsv(probe_dir / f"{raw_fn}_optodes.tsv")
    landmarks = cedalion.io.load_tsv(probe_dir / f"{raw_fn}_landmarks.tsv")
    meas_list = pd.read_csv(probe_dir / f"{raw_fn}_measlist.tsv", sep="\t")
    return geo3d, landmarks, meas_list


def get_ninja_uhd_cap_probe():
    """Load the fullhead Ninja NIRS ultra HD cap probe."""
    probe_dir = DATA_DIR / 'ninja_uhd_cap_probe'
    raw_fn = 'fullhead_164x496'
    geo3d = cedalion.io.load_tsv(probe_dir / f"{raw_fn}_optodes.tsv")
    landmarks = cedalion.io.load_tsv(probe_dir / f"{raw_fn}_landmarks.tsv")
    meas_list = pd.read_csv(probe_dir / f"{raw_fn}_measlist.tsv", sep="\t")
    return geo3d, landmarks, meas_list


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
        basedir = os.path.commonpath(fnames)

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


def get_colin27_parcel_file():
    """Retrieves the Colin27 headmodel, based on :cite:t:`Holmes1998`."""
    fnames = DATASETS.fetch("colin27_parcellation.zip", processor=pooch.Unzip())
    parcel_file = fnames[0]

    return parcel_file


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


def get_icbm152_parcel_file():
    """Retrieves the Colin27 headmodel, based on :cite:t:`Holmes1998`."""
    fnames = DATASETS.fetch("icbm152_parcellation.zip", processor=pooch.Unzip())
    parcel_file = fnames[0]

    return parcel_file


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


def get_precomputed_fluence(dataset: str, head_model: str) -> Path:
    """Precomputed forward model results for examples and documentation.

    Args:
        dataset: "fingertapping" or "fingertappingDOT"
        head_model: "colin27" or "icbm152"

    Returns:
        A Path object pointing to the fluence file.
    """

    fname = f"fluence_{dataset}_{head_model}.h5"

    if fname not in DATASETS.registry:
        raise ValueError(
            f"We don't provide precomputed fluence for dataset "
            f"'{dataset}' and head_model '{head_model}'"
        )

    fname = DATASETS.fetch(fname)

    return Path(fname)


def get_precomputed_sensitivity(dataset: str, head_model: str) -> xr.DataArray:
    """Precomputed sensitivities for examples and documentation.

    Args:
        dataset: "fingertapping", "fingertappingDOT", "nn22_resting",
            "ninja_cap_56x144", "ninja_uhd_cap_164x496"
        head_model: "colin27" or "icbm152"

    Returns:
        The precomputed sensitivity (Adot) matrix
    """

    fname = f"sensitivity_{dataset}_{head_model}.nc"

    if fname not in DATASETS.registry:
        raise ValueError(
            f"We don't provide precomputed sensitivity for dataset "
            f"'{dataset}' and head_model '{head_model}'"
        )

    fname = DATASETS.fetch(fname)

    Adot = load_Adot(fname)

    return Adot


def get_nn22_resting_state() -> cdc.Recording:
    fnames = DATASETS.fetch("nn22_resting_state.zip", processor=pooch.Unzip())
    fname = [Path(i) for i in fnames if i.endswith(".snirf")][0]
    rec = cedalion.io.read_snirf(fname)[0]

    return rec
