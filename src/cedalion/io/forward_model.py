import h5py
import xarray as xr

import cedalion.dataclasses as cdc


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

    Adot = xr.open_dataset(fn)
    Adot = xr.DataArray(
            Adot.to_array()[0],
            dims=["channel", "vertex", "wavelength"],
            coords={"channel": ("channel", Adot.channel.values),
                    "wavelength": ("wavelength", Adot.wavelength.values),
                    "is_brain": ("vertex", Adot.is_brain.values)
                    }
            )
    return Adot



def save_fluence(fn : str, fluence_all, fluence_at_optodes):
    """Save forward model computation results.

    This method uses a lossy compressions algorithm to reduce file size.
    """

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
