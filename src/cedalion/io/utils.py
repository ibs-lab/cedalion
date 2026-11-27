"""HDF5/netCDF/xarray serialisation helpers for cedalion I/O."""

import h5py
import xarray as xr
import numpy as np


def _has_string_dtype(array):
    """Return True if *array* contains string-typed elements."""
    if array.dtype.kind in {"U", "S"}:
        return True
    elif array.dtype.kind == "O":
        return any(isinstance(x, (str, bytes, np.str_, np.bytes_)) for x in array.flat)
    else:
        return False


def xarray_to_hdfgroup(f: h5py.File, array: xr.DataArray, name: str):
    """Serialise an xr.DataArray into an HDF5 group.

    Writes the array values, dimension names, and 1-D coordinates into a
    group at ``f[name]``.  String coordinates are stored with an HDF5
    variable-length string dtype; numeric coordinates are stored as-is.

    Args:
        f: Open, writable HDF5 file object.
        array: DataArray to serialise.  All coordinates must be 1-D.
        name: HDF5 group path to create (e.g. ``"fluence/scalp"``).
    """
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
    """Deserialise a DataArray from an HDF5 group written by :func:`xarray_to_hdfgroup`.

    Args:
        f: Open HDF5 file object.
        path: HDF5 group path (e.g. ``"fluence/scalp"``).

    Returns:
        xr.DataArray reconstructed from the stored values, dims, and coordinates.
    """
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


def save_dataarray_quantified(array: xr.DataArray, path: str) -> None:
    """Write a DataArray to netCDF without losing its pint units.

    netCDF cannot store a pint quantity: passing a quantified DataArray to
    ``to_netcdf`` silently strips the units, emitting only a
    ``UnitStrippedWarning``. Dequantifying first moves the units into
    ``attrs["units"]``, where netCDF keeps them and from where
    :func:`load_dataarray_quantified` restores them.

    Args:
        array: DataArray to write. May be quantified or not; an unquantified
            array is written unchanged.
        path: Destination netCDF file.
    """
    array.pint.dequantify().to_netcdf(path)


def load_dataarray_quantified(
    path: str, fallback_units: str | None = None
) -> xr.DataArray:
    """Read a DataArray from netCDF and restore its pint units.

    Counterpart of :func:`save_dataarray_quantified`, which parks the units in
    ``attrs["units"]`` where netCDF can keep them.

    Dimension names — which encode the coordinate reference system for
    cedalion's affine transforms and point clouds — are preserved by netCDF and
    are taken from the file, never guessed.

    Args:
        path: netCDF file to read.
        fallback_units: Units to assume if the file carries no ``units``
            attribute, i.e. if it was written before units were stored. If
            ``None``, such an array is returned unquantified.

    Returns:
        The DataArray, quantified if unit information was available.
    """
    array = xr.load_dataarray(path)

    if array.attrs.get("units", None) is not None:
        return array.pint.quantify()

    if fallback_units is not None:
        return array.pint.quantify(fallback_units)

    return array
