"""Cedalion datasets and utility functions."""

import os.path
from pathlib import Path
from importlib.resources import files
from collections import namedtuple
from dataclasses import dataclass

import pandas as pd
import pooch
import xarray as xr
import json
import scipy
import gzip

import cedalion
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.io
from cedalion.io.forward_model import load_Adot
from cedalion.utils import deprecated

DATASETS = pooch.create(
    path=pooch.os_cache("cedalion"),
    base_url="https://doc.ibs.tu-berlin.de/cedalion/datasets/{version}/",
    env="CEDALION_DATA_DIR",
    version=cedalion.__version__,
    version_dev="dev",
    registry={
        # recordings
        "mne_nirsport2_raw.snirf": "sha256:12e5fabe64ecc7ef4b83f6bcd77abb41f5480d5f17a2b1aae0e2ad0406670944",  # noqa: E501
        "fingertapping.zip": "sha256:f2253cca6eef8221d536da54b74d8556b28be93df9143ea53652fdc3bc011875",  # noqa: E501
        "fingertappingDOT.zip": "sha256:03e620479bd48aea8457050b7ce42e0c73ef5448296272448b54cee0e883853e",  # noqa: E501
        "multisubject-fingertapping.zip": "sha256:9949c46ed676e52c385b4c09e3a732f6e742bf745253f4b4208ba678f9a0709b",  # noqa: E501
        "nn22_resting_state.zip": "sha256:0394347af172d906fe33403e84303435af26d82fdcf1d36dad5c7b05beb82d88",  # noqa:E501
        "snirf2bids_example_dataset.zip": "sha256:f14508e332c7d259c13b9717ac3c490ab2cabfd7b30fdf97b347d5ba59b783d1",  # noqa:E501
        "spafNIRS_example_sub179.zip": "sha256:0a247be5bfa3c7b5bc12d19203e2bd5432df964d72646945891601d0ba944141",  # noqa:E501

        # head models
        #   deprecated:
        "colin27_segmentation.zip": "sha256:783eeaf75a64aff27c2c07d4e6a8e9c6d5df66886f1b5696fd3f10a402f30d29",  # noqa: E501
        "colin27_segmentation_downsampled_3x3x3.zip": "sha256:ab98b6bae3ef76be6110dc544917f4f2f7ef7233ac697d9cf8bb4a395e81b6cd",  # noqa: E501
        "colin2SHM.zip": "sha256:7568452d38d80bab91eb4b99c4dd85f3302243ecf9d5cf55afe629502e9d9960",  # noqa: E501
        "ICBM152_2020.zip": "sha256:43e2250288830ca3b0ef6da73f62afcc9233e2fb783498645a36f4f8972106fe",  # noqa: E501
        "colin27_parcellation.zip": "sha256:70cb51cc587b7a7389050b854beede76327ed8b105fa12971584a7d1bb7fa080",  # noqa:E501
        "icbm152_parcellation.zip": "sha256:b69ffdb3ff2fe3d85a6d5c139e59147d05ca97127589c1e4c2a8d031850f0148",  # noqa:E501

        #   current
        "hm_colin27.zip": "sha256:481aed79d9542d96b62b182cb624176108464933faf8cc7be2877cdc9a914c39",  # noqa:E501
        "hm_icbm152.zip": "sha256:91bb99709b6ceadd41674acc0db6cf26d70dccb57e41797b474aa9ce6aeed3e8",  # noqa:E501
        "fs_reconall_colin27.zip": "sha256:988b74efddb7cc2551dced25d8cbee89e8b32f1d8e1f07d84796e59db6b5f736",  # noqa:E501
        "fs_reconall_icbm152.zip": "sha256:c8d7ae1923724d15074a03155920b4a817b8054f38cf6f5f37b4f37b26dbdfb6",  # noqa:E501

        "fieldtrip_standard1005.elc": "sha256:1ee59197946d62de872db2ac7f2243a596662c231427366f6dc5d84ed237f853",  # noqa:E501

        # precomputed forward model results
        "fluence_fingertapping_colin27.h5": "sha256:07ac3695a3b3f5782e6c8ede04b9ce620b023d8198babb7b1076fe760a726c36",  # noqa:E501
        "fluence_fingertapping_icbm152.h5": "sha256:566fec429da99d9e966cbd225cc4bc80777d0dc08a5b195fd2cd7f154f5afcae",  # noqa:E501
        "fluence_fingertappingDOT_colin27.h5": "sha256:8768db54ec30071bbb6d5fe4b3e986e2d94aef3cd86cbec2e3fa2b37c83d2cdb",  # noqa:E501
        "fluence_fingertappingDOT_icbm152.h5": "sha256:583f4bcffe7f9a7874f108df4a96aaf213e1bd361e05c4f69ca3cd0dbcd33fd0",  # noqa:E501
        "fluence_ninjanirs_colin27.h5": "sha256:89d82c4f5a985f79777fceeffab9ef90365056ccda8ea4e29bc71c4d24fb0e0a",  # noqa: E501

        "sensitivity_fingertapping_colin27.nc": "sha256:abda3b0cb93ce79a8e5e867875d9f181289206a6db445826e8dc3a231a082298",  # noqa:E501
        "sensitivity_fingertapping_icbm152.nc": "sha256:3ead570f53b276f9b876954a95501b73ae22cee10583b6d67b1380e13afec3c5",  # noqa:E501
        "sensitivity_fingertappingDOT_colin27.nc": "sha256:e9cfbc9fb34cb4898467dd8a8fdd84510c91cc413499f4ccd11d72e8a1a3a2c8",  # noqa:E501
        "sensitivity_fingertappingDOT_icbm152.nc": "sha256:f6dcd84e7330849ba84621a41939b51e8b6fa596e31c2da96ccca2bb979212a8",  # noqa:E501
        "sensitivity_ninja_cap_56x144_colin27.nc": "sha256:c892df3400747c3ec010006241c457a67a003b4672bf98caf89a91976cdea6d4",  # noqa:E501
        "sensitivity_ninja_cap_56x144_icbm152.nc": "sha256:fcc06e81d5cb3732cbb0df6df0045a835c899cad79ae952fa0ca20af4212a7c9",  # noqa:E501
        "sensitivity_ninja_uhd_cap_164x496_colin27.nc": "sha256:4c8d456a0a7f829353e4f58d9249e183fc5fdb349cf0e113c7387ec2b14ec224",  # noqa:E501
        "sensitivity_ninja_uhd_cap_164x496_icbm152.nc": "sha256:411ef97ba9d4f6a46d94c78f3febf508397d9e3538d31cbabb4e6cf682c1866b",  # noqa:E501
        "sensitivity_nn22_resting_colin27.nc": "sha256:1954f466a7103e9e3ad379af86d3148b78c6e5f778db78972300715d52db342b",  # noqa:E501
        "sensitivity_nn22_resting_icbm152.nc": "sha256:a9c577470450d7fe9b9c534a813622a195b66e82ab628d2828fe0525b0355f39",  # noqa:E501

        "Adot_ninjanirs_colin27.nc": "sha256:3382e6bfd62b5e1213332cc74c88cc8af04a4fd5cebe7001ebc111cf9e9b2d00",  # noqa:E501

        # other:
        "photogrammetry_example_scan.zip": "sha256:f4e4beb32a8217ba9f821edd8b5917a79ee88805a75a84a2aea9fac7b38ccbab",  # noqa: E501

        # fmt: on
    },
    urls={
        "fieldtrip_standard1005.elc": "https://raw.githubusercontent.com/fieldtrip/fieldtrip/refs/heads/master/template/electrode/standard_1005.elc"
    },
)


def get(fname: str | Path) -> Path:
    """Returns the absolute path to a file under cedalion/data/."""

    return files("cedalion.data").joinpath(fname)


def get_ninja_cap_probe():
    """Load the fullhead Ninja NIRS cap probe."""

    probe_dir = get("ninja_cap_probe")
    raw_fn = 'fullhead_56x144_System2'
    geo3d = cedalion.io.load_tsv(probe_dir / f"{raw_fn}_optodes.tsv")
    landmarks = cedalion.io.load_tsv(probe_dir / f"{raw_fn}_landmarks.tsv")
    meas_list = pd.read_csv(probe_dir / f"{raw_fn}_measlist.tsv", sep="\t")
    return geo3d, landmarks, meas_list


def get_ninja_uhd_cap_probe():
    """Load the fullhead Ninja NIRS ultra HD cap probe."""

    probe_dir = get('ninja_uhd_cap_probe')
    raw_fn = 'fullhead_164x496'
    geo3d = cedalion.io.load_tsv(probe_dir / f"{raw_fn}_optodes.tsv")
    landmarks = cedalion.io.load_tsv(probe_dir / f"{raw_fn}_landmarks.tsv")
    meas_list = pd.read_csv(probe_dir / f"{raw_fn}_measlist.tsv", sep="\t")
    return geo3d, landmarks, meas_list


def get_snirf_test_data():
    fname = DATASETS.fetch("mne_nirsport2_raw.snirf")
    return cedalion.io.read_snirf(fname)

@deprecated(
    "This function and the corresponding data files were replaced by "
    "cedalion.data.get_icbm152_headmodel_files ."
)
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

@deprecated(
    "This function and the corresponding data files were replaced by "
    "cedalion.data.get_colin27_headmodel_files ."
)
def get_colin27_parcel_file():
    """Retrieves the Colin27 headmodel, based on :cite:t:`Holmes1998`."""
    fnames = DATASETS.fetch("colin27_parcellation.zip", processor=pooch.Unzip())
    parcel_file = fnames[0]

    return parcel_file


@deprecated(
    "This function and the corresponding data files were replaced by "
    "cedalion.data.get_icbm152_headmodel_files ."
)
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

@deprecated(
    "This function and the corresponding data files were replaced by "
    "cedalion.data.get_icbm152_headmodel_files ."
)
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


def get_snirf2bids_example_dataset() -> tuple[Path, Path]:
    """A dataset for demonstrating the snirf2bids notebook.

    Returns:
        The path to the downloaded dataset directory and the path to the mapping file
        with changes that the user would do manually.
    """
    fnames = DATASETS.fetch("snirf2bids_example_dataset.zip", processor=pooch.Unzip())

    mapping_file = Path(
        [i for i in fnames if i.endswith("snirf2BIDS_mapping_edited.csv")][0]
    )
    dataset_dir = mapping_file.parent

    return dataset_dir, mapping_file


def get_fieldtrip_colin27_landmarks() -> cdt.LabeledPoints:
    fname = DATASETS.fetch("fieldtrip_standard1005.elc")
    return cedalion.io.read_fieldtrip_elc(fname)


def get_spa_fnirs() -> cdc.Recording:
    fnames = DATASETS.fetch("spafNIRS_example_sub179.zip", processor=pooch.Unzip())
    fname = [Path(i) for i in fnames if i.endswith(".snirf")][0]
    rec = cedalion.io.read_snirf(fname)[0]

    return rec


@dataclass
class HeadModelFiles:
    basedir : Path
    mask_files : dict[str, str]
    landmarks_ras_file : str
    brain_vertex_coordinates : str
    scalp_surface_obj : str
    brain_surface_obj : str
    freesurfer_surface_obj : str
    inflated_surface_obj : str
    parcel_colors : str
    voxel_to_vertex_mapping : str

    def load_parcel_colors(self):
        with (self.basedir / self.parcel_colors).open("r") as fin:
            return json.load(fin)


    def load_voxel_to_vertex_mapping(self):
        with gzip.GzipFile(self.basedir / self.voxel_to_vertex_mapping) as fin:
            return scipy.io.mmread(fin)


    def load_brain_vertex_coordinates(self):
        return pd.read_csv(self.basedir / self.brain_vertex_coordinates)



def get_colin27_headmodel_files() -> HeadModelFiles:
    """Retrieves the Colin27 segmentation dataset, based on :cite:t:`Holmes1998`."""

    fnames = DATASETS.fetch("hm_colin27.zip", processor=pooch.Unzip())

    return HeadModelFiles(
        basedir=Path(os.path.commonpath(fnames)),
        mask_files={
            "csf": "mask_csf.nii",
            "gm": "mask_gray.nii",
            "scalp": "mask_skin.nii",
            "skull": "mask_bone.nii",
            "wm": "mask_white.nii",
        },
        landmarks_ras_file="landmarks.mrk.json",
        brain_vertex_coordinates="brain_vertex_coordinates.csv",
        scalp_surface_obj="mask_scalp.obj",
        brain_surface_obj="mask_brain.obj",
        freesurfer_surface_obj="cortex_pial_high.obj",
        inflated_surface_obj="cortex_pial_inflated.obj",
        parcel_colors="parcel_colors.json",
        voxel_to_vertex_mapping="voxel_to_vertex_brain.mtx.gz",
    )


def get_icbm152_headmodel_files() -> HeadModelFiles:
    """Retrieves the ICBM-152 segmentation dataset."""

    fnames = DATASETS.fetch("hm_icbm152.zip", processor=pooch.Unzip())

    return HeadModelFiles(
        basedir=Path(os.path.commonpath(fnames)),
        mask_files={
            "csf": "mask_csf.nii",
            "gm": "mask_gray.nii",
            "scalp": "mask_skin.nii",
            "skull": "mask_bone.nii",
            "wm": "mask_white.nii",
        },
        landmarks_ras_file="landmarks.mrk.json",
        brain_vertex_coordinates="brain_vertex_coordinates.csv",
        scalp_surface_obj="mask_scalp.obj",
        brain_surface_obj="mask_brain.obj",
        freesurfer_surface_obj="cortex_pial_high.obj",
        inflated_surface_obj="cortex_pial_inflated.obj",
        parcel_colors="parcel_colors.json",
        voxel_to_vertex_mapping="voxel_to_vertex_brain.mtx.gz",
    )

def get_colin27_freesurfer_directory() -> Path:
    fnames = DATASETS.fetch("fs_reconall_colin27.zip", processor=pooch.Unzip())
    dirname = os.path.commonpath(fnames)

    return Path(dirname)

def get_icbm152_freesurfer_directory() -> Path:
    fnames = DATASETS.fetch("fs_reconall_icbm152.zip", processor=pooch.Unzip())
    dirname = os.path.commonpath(fnames)

    return Path(dirname)
