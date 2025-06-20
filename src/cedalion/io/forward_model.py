"""Module for saving and loading forward model computation results."""

from pathlib import Path

import h5py
import numpy as np
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion.utils import deprecated_api

def save_Adot(fn: str, Adot: xr.DataArray):
    """Save Adot to a netCDF file.

    Args:
        fn (str): File name to save the data to.
        Adot (xr.DataArray): Data to save.

    Returns:
        None
    """

    Adot.to_netcdf(fn)
    return

def load_Adot(fn: str):
    """Load Adot from a netCDF file.

    Args:
        fn (str): File name to load the data from.

    Returns:
        xr.DataArray: Data loaded from the file.
    """

    Adot = xr.load_dataarray(fn)

    return Adot


def save_fluence(fn : str, fluence_all, fluence_at_optodes):
    """Save forward model computation results.

    This method uses a lossy compression algorithm to reduce file size.
    """

    deprecated_api(
        "The functions load_fluence and save_fluence have been replaced by "
        "the FluenceFile classs."
    )

    with h5py.File(fn, "w") as f:
        # with scaleoffset=14 this should be precise to the 14th decimal digit.
        f.create_dataset(
            "fluence_all",
            data=fluence_all,
            scaleoffset=14,
            shuffle=True,
            compression="lzf",
        )

        f["fluence_all"].attrs["dims"] = fluence_all.dims
        f["fluence_all"].attrs["label"] = [str(i) for i in fluence_all.label.values]
        f["fluence_all"].attrs["wavelength"] = fluence_all.wavelength
        f["fluence_all"].attrs["type"] = [i.value for i in fluence_all.type.values]

        f.create_dataset(
            "fluence_at_optodes",
            data=fluence_at_optodes,
            shuffle=True,
            compression="lzf",
        )

        f["fluence_at_optodes"].attrs["dims"] = fluence_at_optodes.dims
        f["fluence_at_optodes"].attrs["optode1"] = [
            str(i) for i in fluence_at_optodes.optode1.values
        ]
        f["fluence_at_optodes"].attrs["optode2"] = [
            str(i) for i in fluence_at_optodes.optode2.values
        ]
        f["fluence_at_optodes"].attrs["wavelength"] = fluence_all.wavelength

        f.flush()


def load_fluence(fn : str):
    """Load forward model computation results.

    Args:
        fn (str): File name to load the data from.

    Returns:
        Tuple[xr.DataArray, xr.DataArray]: Fluence data loaded from the file.
    """

    deprecated_api(
        "The functions load_fluence and save_fluence have been replaced by "
        "the FluenceFile classs."
    )

    with h5py.File(fn, "r") as f:

        ds = f["fluence_all"]
        fluence_all = xr.DataArray(
            ds,
            dims = ds.attrs["dims"],
            coords = {
                "label" : ("label", ds.attrs["label"]),
                "type" : ("label", [cdc.PointType(i) for i in ds.attrs["type"]]),
                "wavelength" : ds.attrs["wavelength"]
            }
        )
        fluence_all.attrs.clear()

        ds = f["fluence_at_optodes"]

        fluence_at_optodes = xr.DataArray(
            ds,
            dims = ds.attrs["dims"],
            coords = {
                "optode1" : ds.attrs["optode1"],
                "optode2" : ds.attrs["optode2"],
                "wavelength" : ds.attrs["wavelength"]
            }
        )
        fluence_at_optodes.attrs.clear()

    return fluence_all, fluence_at_optodes


class FluenceFile:
    def __init__(self, fname : str | Path, mode="r"):
        self.file = h5py.File(fname, mode)

        f = self.file

        if mode == "r":
            if not (("fluence_all" in f) and ("fluence_at_optodes" in f)):
                raise ValueError("this hdf5 file does not contain fluence data.")

            ds = f["fluence_all"]
            self.optode_labels = ds.attrs["label"].tolist()
            self.wavelengths = ds.attrs["wavelength"].tolist()
        else:
            self.optode_labels = None
            self.wavelengths = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.flush()
        self.file.close()

    def create_fluence_dataset(
        self,
        optode_pos: cdt.LabeledPointCloud,
        wavelengths: np.ndarray,
        fluence_shape : tuple[int, int, int],
        units : str
    ):
        f = self.file

        dims = ["label", "wavelength", "i", "j", "k"]

        self.optode_labels = optode_pos.label.values.copy()
        self.wavelengths = wavelengths.copy()

        optode_types = optode_pos.type.values.copy()
        n_optodes = len(self.optode_labels)
        n_wavelength = len(self.wavelengths)

        f.create_dataset(
            "fluence_all",
            shape=(n_optodes, n_wavelength) + fluence_shape,
            chunks=(1, 1) + fluence_shape,
            #scaleoffset=14,
            shuffle=True,
            fletcher32=False,
            compression="lzf",
        )

        f["fluence_all"].attrs["dims"] = dims
        f["fluence_all"].attrs["label"] = [str(i) for i in self.optode_labels]
        f["fluence_all"].attrs["wavelength"] = self.wavelengths
        f["fluence_all"].attrs["type"] = [i.value for i in optode_types]
        f["fluence_all"].attrs["units"] = units

    def get_fluence(self, label : str, wavelength : float) -> np.ndarray:
        i_label = self.optode_labels.index(label)
        i_wl = self.wavelengths.index(wavelength)
        return self.file["fluence_all"][i_label, i_wl, :,:,:]

    def set_fluence_by_label(self, label: str, wavelength: float, fluence: np.ndarray):
        i_label = self.optode_labels.index(label)
        i_wl = self.wavelengths.index(wavelength)
        self.set_fluence_by_index(i_label, i_wl, fluence)

    def set_fluence_by_index(self, i_label: int, i_wl: int, fluence: np.ndarray):
        self.file["fluence_all"][i_label, i_wl, :, :, :] = fluence

    def get_fluence_at_optodes(self):
        ds = self.file["fluence_at_optodes"]
        fluence_at_optodes = xr.DataArray(
            ds,
            dims = ds.attrs["dims"],
            coords = {
                "optode1" : ds.attrs["optode1"],
                "optode2" : ds.attrs["optode2"],
                "wavelength" : ds.attrs["wavelength"]
            }
        )
        fluence_at_optodes.attrs.clear()
        return fluence_at_optodes

    def set_fluence_at_optodes(self, fluence_at_optodes : xr.DataArray):
        f = self.file

        f.create_dataset(
            "fluence_at_optodes",
            data=fluence_at_optodes,
            shuffle=True,
            compression="lzf",
        )

        f["fluence_at_optodes"].attrs["dims"] = fluence_at_optodes.dims
        f["fluence_at_optodes"].attrs["optode1"] = [
            str(i) for i in fluence_at_optodes.optode1.values
        ]
        f["fluence_at_optodes"].attrs["optode2"] = [
            str(i) for i in fluence_at_optodes.optode2.values
        ]
        f["fluence_at_optodes"].attrs["wavelength"] = fluence_at_optodes.wavelength


    def get_fluence_all(self):
        ds = self.file["fluence_all"]

        fluence_all = xr.DataArray(
            ds,
            dims = ds.attrs["dims"],
            coords = {
                "label" : ("label", ds.attrs["label"]),
                "type" : ("label", [cdc.PointType(i) for i in ds.attrs["type"]]),
                "wavelength" : ds.attrs["wavelength"]
            }
        )
        fluence_all.attrs.clear()

        return fluence_all
