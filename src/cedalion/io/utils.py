import h5py
import xarray as xr
import numpy as np

def _has_string_dtype(array):
    if array.dtype.kind in {"U", "S"}:
        return True
    elif array.dtype.kind == "O":
        return any(isinstance(x, (str, bytes, np.str_, np.bytes_)) for x in array.flat)
    else:
        return False



def xarray_to_hdfgroup(f: h5py.File, array: xr.DataArray, name: str):
    group = f.create_group(name)
    ds = group.create_dataset("values", data=array.values)
    ds.attrs["dims"] = array.dims

    coords_group = group.create_group("coords")

    for cname, carray in array.coords.items():
        assert carray.ndim == 1
        dim_name = carray.dims[0]
        if _has_string_dtype(carray.values):
            ds = coords_group.create_dataset(
                cname, data=carray.values.astype("S"), dtype=h5py.string_dtype()
            )
        else:
            ds = coords_group.create_dataset(cname, data=carray.values)
        ds.attrs["dim"] = dim_name


def xarray_from_hdfgroup(f: h5py.File, path: str):
    ds = f[f"{path}/values"]

    coords = {}

    for cname, cdataset in f[f"{path}/coords"].items():
        if cdataset.dtype == h5py.string_dtype():
            values = cdataset.asstr()[:]
        else:
            values = cdataset[:]
        dimname = cdataset.attrs["dim"]
        coords[cname] = (dimname, values)

    return xr.DataArray(ds[:], dims=ds.attrs["dims"], coords=coords)
