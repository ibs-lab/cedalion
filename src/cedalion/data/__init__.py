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

import cedalion
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.io
from cedalion.io.anatomy import HEADMODEL_MANIFEST, HeadModelFiles  # noqa: F401
from cedalion.io.forward_model import load_Adot
from cedalion.utils import deprecated

from cedalion import cite

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
        "lumo_testdataset.zip" : "sha256:d7c0b74538c98b4e26e73e9606e7e8125f236cdda4c913a10c2d4db93e4357f8", # noqa:E501
        "kernel_testdataset.zip" : "sha256:cfff80c2064b2ddbb884db8aad08004b450cdd7324ee14733c3344b0706a3f18", # noqa:E501
        "artinis_testdataset.zip" : "sha256:bc2b583a1d70e0997701406b8ba02b5a9f9034f30a5ae0882376055181ab453e", # noqa:E501

        # head models
        #   deprecated:
        "colin27_segmentation.zip": "sha256:783eeaf75a64aff27c2c07d4e6a8e9c6d5df66886f1b5696fd3f10a402f30d29",  # noqa: E501
        "colin27_segmentation_downsampled_3x3x3.zip": "sha256:ab98b6bae3ef76be6110dc544917f4f2f7ef7233ac697d9cf8bb4a395e81b6cd",  # noqa: E501
        "colin2SHM.zip": "sha256:7568452d38d80bab91eb4b99c4dd85f3302243ecf9d5cf55afe629502e9d9960",  # noqa: E501
        "ICBM152_2020.zip": "sha256:43e2250288830ca3b0ef6da73f62afcc9233e2fb783498645a36f4f8972106fe",  # noqa: E501
        "colin27_parcellation.zip": "sha256:70cb51cc587b7a7389050b854beede76327ed8b105fa12971584a7d1bb7fa080",  # noqa:E501
        "icbm152_parcellation.zip": "sha256:b69ffdb3ff2fe3d85a6d5c139e59147d05ca97127589c1e4c2a8d031850f0148",  # noqa:E501

        #   current
        "hm_colin27.zip": "sha256:b023cb1e04808daf54c55ea08d962abf69c64c6d6ca6fa2b4b6a27f28858cdc7",  # noqa:E501
        "hm_icbm152.zip": "sha256:b45a7e27383eb92ff6aa4a80e2b78fcdcb0bb2f3a85b0ba1686417f470792c46",  # noqa:E501
        "fs_reconall_colin27.zip": "sha256:1803c42a421ca850b89479ac9ac2de4c53fbe5e65241f70b97f6502ee435ba7b",  # noqa:E501
        "fs_reconall_icbm152.zip": "sha256:81284f9f65e727fd1244e82ef575e22dc3abe45c7603b1aad7fdc384feda5d72",  # noqa:E501

        "fieldtrip_standard1005.elc": "sha256:1ee59197946d62de872db2ac7f2243a596662c231427366f6dc5d84ed237f853",  # noqa:E501

        # precomputed forward model results
        "fluence_fingertapping_colin27.h5": "sha256:659c24675db9c3bd598727c75c05842ab621193cefea05fdb8f08110f6877f35",  # noqa:E501
        "fluence_fingertapping_icbm152.h5": "sha256:a14c7aa142f61019b7e41606b5df29a5a30504609f17dca4ce2e31e7b68889da",  # noqa:E501
        "fluence_fingertappingDOT_colin27.h5": "sha256:ded2eef45c58f8d29cb2323d0f7cd31779bdcae5bdc2ac45cf54d33323386e03",  # noqa:E501
        "fluence_fingertappingDOT_icbm152.h5": "sha256:9e6c5a031a724b23dd3253e4a56150c1b88cd45443c352e5a2655ad83e1c15e0",  # noqa:E501

        "sensitivity_fingertapping_colin27.nc": "sha256:3fb8adf273d22f71ba2500a90f196e890450052696501a4c67e1cd49cb45d21e",  # noqa:E501
        "sensitivity_fingertapping_icbm152.nc": "sha256:b71d891b0beec3518eea866d77993ca5ece6e89392d40cde62c382456b70b623",  # noqa:E501
        "sensitivity_fingertappingDOT_colin27.nc": "sha256:0d773edc82bbbc00ff9dee21f1cd85f610437111fc7afe2fdbd3a147668b0709",  # noqa:E501
        "sensitivity_fingertappingDOT_icbm152.nc": "sha256:55764b4aef00aded0b0ac4e06a1bf702c070d8dfb676bd4c48f6928a15e550e8",  # noqa:E501
        "sensitivity_ninja_cap_56x144_colin27.nc": "sha256:3c5a74775e87792a43e1b10c18343269c1692055eb21fd82360fe3f4fe496ebf",  # noqa:E501
        "sensitivity_ninja_cap_56x144_icbm152.nc": "sha256:d1ce1e70c043ee44899f4637d7cb7a7cb77e8aacb01d16f2b90d2aa01b30fd3b",  # noqa:E501
        "sensitivity_ninja_uhd_cap_164x496_colin27.nc": "sha256:2284589c67775b5f99f4fa483c0786ac8241dadd422cff05c0b2aa66fe9a379e",  # noqa:E501
        "sensitivity_ninja_uhd_cap_164x496_icbm152.nc": "sha256:e2d912255255040f413e3b5d1438572dc91de18f9fcb9896fa3c6b03770e5625",  # noqa:E501
        "sensitivity_nn22_resting_colin27.nc": "sha256:5ce778a1b14b7e6d677c9dcd5368f469abea5c6d173400d9b3fc9403cbedbdc5",  # noqa:E501
        "sensitivity_nn22_resting_icbm152.nc": "sha256:dec7d7918ebcf5f021dc6680229c3d7a2ea193268413fab68eb059b2b6f31215",  # noqa:E501

        "sensitivity_lumo_testdataset_colin27.nc" : "sha256:a2d09fb7fa37a3e44fa99f314149d2194d59760cd11768751e6cd29d9e648256", # noqa:E501
        "sensitivity_lumo_testdataset_icbm152.nc" : "sha256:0666e948d90d208441dd68b049eb116ee020051ca8e339bdf4203da01488fc73", # noqa:E501
        "sensitivity_kernel_testdataset_colin27.nc" : "sha256:450b5a3e11ab8474fd24ec254afc122d4a0c5a3ffb9384cac67cc459ba6bbc61", # noqa:E501
        "sensitivity_kernel_testdataset_icbm152.nc" : "sha256:f3987da1bdf7d69e25915d39f0a04d992517810a2c1d241a5a56178e511d2b95", # noqa:E501
        "sensitivity_artinis_testdataset_colin27.nc" : "sha256:cc38f8e76abf2d00de848ce7afc089c4188780e672e687f50292d5e09fc23b6e", # noqa:E501
        "sensitivity_artinis_testdataset_icbm152.nc" : "sha256:f2a0fef64f736b5ea98324b7fd61a2b5622f204b317aba306f42997f16f41a23", # noqa:E501

        # atlases:
        "atlas_aal3.zip" : "sha256:8e2dcea9c1df1cd71855584a1f6bd11b6c6f3a60d11ce85d37e6f41966474af3", # noqa:E501
        "atlas_brodmann.zip" :"sha256:551492873096a1cd9e00fb4edbb011bdc185ab7dc9859c92c003948dc36f3da6", # noqa:E501

        # photogrammetry scans:
        "photogrammetry_example_scan.zip": "sha256:f4e4beb32a8217ba9f821edd8b5917a79ee88805a75a84a2aea9fac7b38ccbab",  # noqa: E501
        "fingertappingDOT_photogrammetry.zip" : "sha256:2d0372902f49500dda8e8d3ef037e613286eda403cf4e8afd53820916353052f", # noqa: E501

        # fmt: on
    },
    urls={
        "fieldtrip_standard1005.elc": "https://raw.githubusercontent.com/fieldtrip/fieldtrip/refs/heads/master/template/electrode/standard_1005.elc"
    },
)


def get(fname: str | Path) -> Path:
    """Returns the absolute path to a file under cedalion/data/."""

    return files("cedalion.data").joinpath(fname)


@dataclass
class AtlasFiles:
    """Paths for a bundled atlas volume and label lookup."""

    basedir: Path
    nifti: Path
    labels_json: Path


def get_aal3_atlas_files() -> AtlasFiles:
    """Return paths to the bundled AAL3 atlas volume and label lookup."""

    atlas_dir = get("atlases/aal3")
    return AtlasFiles(
        basedir=atlas_dir,
        nifti=atlas_dir / "ROI_MNI_V7_1mm.nii",
        labels_json=atlas_dir / "aal_labels.json",
    )


def get_brodmann_atlas_files() -> AtlasFiles:
    """Return paths to the bundled Brodmann atlas volume and label lookup."""

    atlas_dir = get("atlases/brodmann")
    return AtlasFiles(
        basedir=atlas_dir,
        nifti=atlas_dir / "Brodmann_Mai_Matajnik.nii",
        labels_json=atlas_dir / "brodmann_labels.json",
    )


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
    cite("Holmes1998")
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
    cite("Holmes1998")
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
    """Retrieves the ICBM headmodel parcellation."""
    fnames = DATASETS.fetch("icbm152_parcellation.zip", processor=pooch.Unzip())
    parcel_file = fnames[0]

    return parcel_file


def get_fingertapping() -> cdc.Recording:
    """Retrieves a finger tapping recording in BIDS format.

    Data is from :cite:t:`Luke2021`
    """

    cite("Luke2021")

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


def get_fingertappingDOT(crs : str = "digitized") -> cdc.Recording:
    """Retrieves a finger tapping DOT example dataset from the IBS Lab."""

    fnames = DATASETS.fetch("fingertappingDOT.zip", processor=pooch.Unzip())

    fname = [i for i in fnames if i.endswith(".snirf")][0]

    rec = cedalion.io.read_snirf(fname, crs=crs)[0]

    geo3d = rec.geo3d.points.rename({"NASION": "Nz"})

    rec.geo3d = geo3d

    amp = rec.get_timeseries("amp")
    amp = amp.pint.dequantify().pint.quantify("V")
    rec.set_timeseries("amp", amp, overwrite=True)

    return rec


def get_fingertapping_snirf_path() -> Path:
    cite("Luke2021")

    fnames = DATASETS.fetch("fingertapping.zip", processor=pooch.Unzip())
    fname = [Path(i) for i in fnames if i.endswith(".snirf")][0]
    return fname


def get_multisubject_fingertapping_snirf_paths():
    cite("Luke2021")

    fnames = DATASETS.fetch("multisubject-fingertapping.zip", processor=pooch.Unzip())
    fnames = sorted([i for i in fnames if i.endswith(".snirf")])
    return fnames


def get_multisubject_fingertapping_path() -> Path:
    cite("Luke2021")

    fnames = DATASETS.fetch("multisubject-fingertapping.zip", processor=pooch.Unzip())
    return [Path(i).parent for i in fnames if i.endswith("README.md")][0]


def get_photogrammetry_example_scan():
    fnames = DATASETS.fetch("photogrammetry_example_scan.zip", processor=pooch.Unzip())
    fname_scan = [i for i in fnames if i.endswith(".obj")][0]
    fname_snirf = [i for i in fnames if i.endswith(".snirf")][0]
    fname_montage = [i for i in fnames if i.endswith(".png")][0]
    return fname_scan, fname_snirf, fname_montage


def get_fingertappingDOT_photogrammetry_scan():
    fnames = DATASETS.fetch(
        "fingertappingDOT_photogrammetry.zip", processor=pooch.Unzip()
    )
    fname_scan = [i for i in fnames if i.endswith(".obj")][0]
    fname_montage = [i for i in fnames if i.endswith(".png")][0]
    return fname_scan, fname_montage


def get_lumo_testdataset():
    fnames = DATASETS.fetch("lumo_testdataset.zip", processor=pooch.Unzip())
    fname_snirf = [i for i in fnames if i.endswith(".snirf")][0]
    return cedalion.io.read_snirf(fname_snirf)[0]


def get_artinis_testdataset():
    fnames = DATASETS.fetch("artinis_testdataset.zip", processor=pooch.Unzip())
    fname_snirf = [i for i in fnames if i.endswith(".snirf")][0]
    return cedalion.io.read_snirf(fname_snirf)[0]


def get_kernel_testdataset():
    fnames = DATASETS.fetch("kernel_testdataset.zip", processor=pooch.Unzip())
    fname_snirf = [i for i in fnames if i.endswith(".snirf")][0]
    return cedalion.io.read_snirf(fname_snirf)[0]


EXAMPLE_PROBES = (
    "fingertapping",
    "fingertappingDOT",
    "nn22_resting",
    "ninja_cap_56x144",
    "ninja_uhd_cap_164x496",
    "lumo_testdataset",
    "kernel_testdataset",
    "artinis_testdataset",
)


class ExampleProbe(namedtuple("ExampleProbe", ["geo3d", "meas_list", "amp"])):
    """Probe geometry, measurement list and amplitudes of an example dataset.

    Attributes:
        geo3d: Optode and landmark positions.
        meas_list: Measurement list.
        amp: Amplitude time series. For probes without recordings (ninja caps), a
            one-sample time series with the channels of the measurement list.
    """

    __slots__ = ()


def get_example_probe(
    name: str, max_channel_distance: cdt.QLength | None = None
) -> ExampleProbe:
    """Probe of one of cedalion's example datasets, prepared for registration.

    These are the probes for which :func:`get_precomputed_fluence` and
    :func:`get_precomputed_sensitivity` provide forward models. Compared to the raw
    datasets, landmark labels are normalized and missing landmarks are added, so
    that every probe can be registered to the standard head models.

    Args:
        name: One of :data:`EXAMPLE_PROBES`.
        max_channel_distance: Only for ``"lumo_testdataset"``: drop channels with
            larger source-detector distances. Defaults to 3.5 cm.

    Returns:
        The probe geometry, measurement list and amplitude time series.
    """
    from cedalion.geometry.landmarks import normalize_landmarks_labels
    from cedalion.nirs import channel_distances

    def from_recording(rec, geo3d=None):
        geo3d = rec.geo3d if geo3d is None else geo3d
        return ExampleProbe(geo3d, rec._measurement_lists["amp"], rec["amp"])

    def from_probe_files(geo3d, landmarks, meas_list):
        geo3d = xr.concat((geo3d, landmarks), dim="label").pint.quantify("mm")
        amp = cdc.empty_timeseries_from_measurement_list(meas_list)
        return ExampleProbe(geo3d, meas_list, amp)

    if name == "fingertapping":
        return from_recording(get_fingertapping())

    if name == "fingertappingDOT":
        return from_recording(get_fingertappingDOT())

    if name == "nn22_resting":
        return from_recording(get_nn22_resting_state())

    if name == "ninja_cap_56x144":
        return from_probe_files(*get_ninja_cap_probe())

    if name == "ninja_uhd_cap_164x496":
        return from_probe_files(*get_ninja_uhd_cap_probe())

    if name == "lumo_testdataset":
        if max_channel_distance is None:
            max_channel_distance = 3.5 * cedalion.units.cm

        rec = get_lumo_testdataset()
        geo3d = normalize_landmarks_labels(rec.geo3d)

        amp = rec["amp"]
        amp = amp.sel(channel=channel_distances(amp, geo3d) < max_channel_distance)

        meas_list = rec._measurement_lists["amp"]
        channels = meas_list.source.astype(str) + meas_list.detector.astype(str)
        meas_list = meas_list[channels.isin(amp.channel.values)]

        return ExampleProbe(geo3d, meas_list, amp)

    if name == "artinis_testdataset":
        rec = get_artinis_testdataset()
        geo3d = normalize_landmarks_labels(rec.geo3d)
        geo3d = geo3d.pint.dequantify().pint.quantify("cm").pint.to("mm")

        # The dataset lacks landmarks but is in MNI space. Adopt the landmarks of
        # the colin27 head model (RAS, mm).
        colin27_landmarks = {
            "Nz": [-0.679195, 83.259347, -39.408527],
            "Iz": [0.0045, -118.565, -23.078],
            "LPA": [-86.0761, -19.9897, -47.986],
            "RPA": [85.7939, -20.0093, -48.031],
            "Cz": [0.938423, -9.965052, 97.700135],
        }
        landmarks = cdc.build_labeled_points(
            list(colin27_landmarks.values()),
            crs=geo3d.points.crs,
            units="mm",
            labels=list(colin27_landmarks.keys()),
            types=[cdc.PointType.LANDMARK] * len(colin27_landmarks),
        )
        return from_recording(rec, xr.concat((geo3d, landmarks), dim="label"))

    if name == "kernel_testdataset":
        rec = get_kernel_testdataset()
        return from_recording(rec, normalize_landmarks_labels(rec.geo3d))

    raise ValueError(f"Unknown example probe {name!r}. Available: {EXAMPLE_PROBES}")


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

    # precomputed fluences were calculated with MCX
    cite("Fang2009")
    cite("Yu2018")
    cite("Yan2020")

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

    # precomputed senitivities were calculated with MCX
    cite("Fang2009")
    cite("Yu2018")
    cite("Yan2020")

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


def _find_manifest(fnames: list[str]) -> Path:
    """Return the head model manifest among the files of an unzipped dataset."""
    manifests = [Path(f) for f in fnames if Path(f).name == HEADMODEL_MANIFEST]
    if len(manifests) != 1:
        raise ValueError(f"expected exactly one {HEADMODEL_MANIFEST} in dataset.")
    return manifests[0]



def get_colin27_headmodel_files() -> HeadModelFiles:
    """Retrieve the Colin27 head model files.

    The Colin27 atlas (:cite:t:`Holmes1998`) was further processed with FreeSurfer
    (:cite:t:`Fischl2012`) to produce cortical surface meshes, parcellations
    (:cite:t:`Schaefer2018`), and voxel-to-vertex mappings used in cedalion's DOT
    pipeline.
    """

    cite("Holmes1998")
    cite("Fischl2012")
    cite("Schaefer2018")

    fnames = DATASETS.fetch("hm_colin27.zip", processor=pooch.Unzip())

    return HeadModelFiles.from_manifest(_find_manifest(fnames))


def get_icbm152_headmodel_files() -> HeadModelFiles:
    """Retrieves the ICBM-152 segmentation dataset.

    The ICBM1-52 atlas (:cite:t:`Fonov2011`) was further processed with FreeSurfer
    (:cite:t:`Fischl2012`) to produce cortical surface meshes, parcellations
    (cite:t:`Schaefer2018`), and voxel-to-vertex mappings used in cedalion's DOT
    pipeline.
    """

    cite("Fonov2011")
    cite("Fischl2012")
    cite("Schaefer2018")

    fnames = DATASETS.fetch("hm_icbm152.zip", processor=pooch.Unzip())

    return HeadModelFiles.from_manifest(_find_manifest(fnames))


def get_colin27_freesurfer_directory() -> Path:
    """Return the path to the FreeSurfer recon-all output for the Colin27 atlas.

    The Colin27 MRI (:cite:t:`Holmes1998`) was processed with FreeSurfer's
    ``recon-all`` pipeline (:cite:t:`Fischl2012`) to generate cortical surface
    reconstructions used in cedalion's DOT head modelling.
    """
    cite("Holmes1998")
    cite("Fischl2012")
    fnames = DATASETS.fetch("fs_reconall_colin27.zip", processor=pooch.Unzip())
    dirname = os.path.commonpath(fnames)

    return Path(dirname)

def get_icbm152_freesurfer_directory() -> Path:
    """Return the path to the FreeSurfer recon-all output for the ICBM-152 atlas.

    The ICBM-152 MRI was processed with FreeSurfer's ``recon-all`` pipeline
    (:cite:t:`Fischl2012`) to generate cortical surface reconstructions used in
    cedalion's DOT head modelling.
    """
    cite("Fonov2011")
    cite("Fischl2012")
    fnames = DATASETS.fetch("fs_reconall_icbm152.zip", processor=pooch.Unzip())
    dirname = os.path.commonpath(fnames)

    return Path(dirname)


def get_atlas_files(atlas : str) -> tuple[Path, Path]:
    """Return atlas volume and label files.

    Args:
        atlas: Name of the atlas. One of ``"aal3"`` or ``"brodmann"``.

    Returns:
        A tuple ``(fname_volume, fname_labels)`` with paths to the NIfTI volume
        and the JSON label file.
    """
    AVAILABLE_ATLASES = ["aal3", "brodmann"]

    if atlas == "aal3":
        fnames = DATASETS.fetch("atlas_aal3.zip", processor=pooch.Unzip())
        fname_volume = [Path(i) for i in fnames if i.endswith(".nii")][0]
        fname_labels = [Path(i) for i in fnames if i.endswith(".json")][0]
        cite("Rolls2020")
    elif atlas == "brodmann":
        fnames = DATASETS.fetch("atlas_brodmann.zip", processor=pooch.Unzip())
        fname_volume = [Path(i) for i in fnames if i.endswith(".nii")][0]
        fname_labels = [Path(i) for i in fnames if i.endswith(".json")][0]
        cite("Mai2017")
    else:
        raise ValueError(f"atlas must be one of {AVAILABLE_ATLASES}")

    return fname_volume, fname_labels
