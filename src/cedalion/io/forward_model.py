import xarray as xr
import cedalion

def save_Adot(fn: str, Adot: xr.DataArray):
    Adot.to_netcdf(fn)
    return

def load_Adot(fn: str):
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
