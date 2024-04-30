import xarray as xr
import cedalion

def save_Adot(fn: str, Adot: xr.DataArray):
    """Save Adot to a netCDF file.

    Parameters:
    ----------
    fn: str
        File name to save the data to.
    Adot: xr.DataArray
        Data to save.

    Returns:
    -------
        None
    """

    Adot.to_netcdf(fn)
    return

def load_Adot(fn: str):
    """Load Adot from a netCDF file.

    Args:
        fn: str
            File name to load the data from.

    Returns:
        Adot: xr.DataArray
            Data loaded from the file.
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
