"""Utility functions for xarray objects."""

import warnings

import numpy as np
import pint
import xarray as xr
import os
import tempfile
import shutil


def pinv(array: xr.DataArray) -> xr.DataArray:
    """Calculate the pseudoinverse of a 2D xr.DataArray.

    FIXME: handles unitless and quantified DataArrays but not
           DataArrays with units in their attrs.

    Args:
        array (xr.DataArray): Input array

    Returns:
        array_inv (xr.DataArray): Pseudoinverse of the input array
    """
    if not array.ndim == 2:
        raise ValueError("array must have only 2 dimensions")

    # /!\ need to transpose dimensions when applying np.linalg.pinv
    dims = list(array.dims)
    transposed_dims = dims[::-1]

    units = array.pint.units
    inv_units = None

    # determine inverted units and dequantify
    if units is not None:
        q = 1 / units
        inv_units = q.units
        array = array.pint.dequantify()

    # apply numpy's pinv
    array_inv = xr.apply_ufunc(
        np.linalg.pinv,
        array,
        input_core_dims=[dims],
        output_core_dims=[transposed_dims],
    )

    # quantify if necessary
    if inv_units is not None:
        array_inv = array_inv.pint.quantify(inv_units)

    return array_inv


def norm(array: xr.DataArray, dim: str) -> xr.DataArray:
    """Calculate the vector norm along a given dimension.

    Extends the behavior of numpy.linalg.norm to xarray DataArrays.

    Args:
        array (xr.DataArray): Input array
        dim (str): Dimension along which to calculate the norm

    Returns:
        normed (xr.DataArray): Array with the norm along the specified dimension
    """
    if dim not in array.dims:
        raise ValueError(f"array does not have dimension '{dim}'")

    if (units := array.pint.units) is not None:
        array = array.pint.dequantify()

    normed = xr.apply_ufunc(
        np.linalg.norm, array, input_core_dims=[[dim]], kwargs={"axis": -1}
    )

    if units is not None:
        normed = normed.pint.quantify(units)

    return normed


def mask(array: xr.DataArray, initval: bool) -> xr.DataArray:
    """Create a boolean mask array with the same shape as the input array."""

    return xr.full_like(array, initval, dtype=bool)


def apply_mask(
    data_array: xr.DataArray, mask: xr.DataArray, operator: str, dim_collapse: str
) -> xr.DataArray:
    """Apply a boolean mask to a DataArray according to the defined "operator".

    Args:
        data_array: NDTimeSeries, input time series data xarray
        mask: input boolean mask array with a subset of dimensions matching data_array
        operator: operators to apply to the mask and data_array
            "nan": inserts NaNs in the data_array where mask is False
            "drop": drops value in the data_array where mask is False
        dim_collapse: Mask dimension to collapse to, merging boolean masks along all
            other dimensions. Can be skipped with "none".
            Example: collapsing to "channel" dimension will drop or nan a channel if it
            is "False" along any other dimensions

    Returns:
        masked_data_array: Input data_array with applied mask
        masked_elements: List of elements in data_array that were masked (e.g.
            dropped or set to NaN)
    """
    flag_collapse = False

    # check if all dimensions in mask are dimensions of data_array
    if not all(dim in data_array.dims for dim in mask.dims):
        raise ValueError("mask dimensions must be a subset of data_array dimensions")
        # check if dim_collapse is a dimension of mask
    if dim_collapse.lower() != "none":
        if dim_collapse not in mask.dims:
            raise ValueError("dim_collapse must be a dimension of mask")
        else:
            # collapse to dimension given by "dim_collapse"
            flag_collapse = True
            dims2collapse = [dim for dim in mask.dims if dim != dim_collapse]
            mask = mask.all(dim=dims2collapse)
            # print(f"mask collapsed to {dim_collapse} dimension")

    # apply the mask to data_array according to instructions from "operator" argument
    if operator.lower() == "nan":
        # inserts NaNs in the data_array where mask is False
        masked_data_array = data_array.where(mask, other=np.nan)
    elif operator.lower() == "drop":
        # drops value in the data_array where mask is False.
        # Note: values are only dropped if mask has "False" across the entire relevant
        # dimension
        masked_data_array = data_array.where(mask, drop=True)

    # return the masked elements if dimensions were collapsed
    if flag_collapse:
        masked_elements = mask.where(~mask, drop=True)[dim_collapse].values
    else:
        # FIXME clean this up: return the masked elements as a list of indices
        masked_elements = "N/A"

    return masked_data_array, masked_elements


def convolve(data_array: xr.DataArray, kernel: np.ndarray, dim: str) -> xr.DataArray:
    """Convolve a DataArray along a given dimension "dim" with a "kernel"."""

    if dim not in data_array.dims:
        raise ValueError(f"array does not have dimension '{dim}'")

    if (units := data_array.pint.units) is not None:
        data_array = data_array.pint.dequantify()

    convolved = xr.apply_ufunc(
        lambda x: np.convolve(x, kernel, mode="same"),
        data_array,
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        vectorize=True,
    )

    if units is not None:
        convolved = convolved.pint.quantify(units)

    return convolved


def other_dim(data_array: xr.DataArray, *dims: str) -> str:
    """Get the dimension name not listed in *dims.

    Checks that there is only one more dimension than given in dims  and returns
    its name.

    Args:
        data_array: an xr.DataArray
        *dims: names of dimensions

    Returns:
        The name of the dimension of data_array.
    """

    dims = set(dims)
    array_dims = set(data_array.dims)

    ndim_expected = len(dims) + 1
    if data_array.ndim != ndim_expected:
        raise ValueError(f"expected data_array to have ndim={ndim_expected}.")

    if not dims.issubset(data_array.dims):
        raise ValueError("not all provided dimensions found in data_array")

    return (array_dims - dims).pop()


def coords_from_other(
    source: xr.DataArray, dims: list[str] = None, **aux_coords
) -> dict[str, tuple[str, xr.DataArray]]:
    """Create a dictionary of coordinates from source for matching dims in target.

    Args:
        source: the DataArray to copy the coordinates from.
        dims: a list of dimensions names. If specified, copy only coords for those dims.
        aux_coords: additional key-value pairs to add to the resulting coords dict.

    Returns:
        A dictionary that can be passed to DataArray.assign_coords.
    """

    for coord_name, coord_dataarray in source.coords.items():
        if coord_dataarray.dims == tuple(): # scalar values without dimension
            if dims: # skip because they cannot belong to any selected dim
                continue
            aux_coords[coord_name] = coord_dataarray.values
        else:
            assert len(coord_dataarray.dims) == 1
            coord_dim = coord_dataarray.dims[0]

            # skip unwanted coordinates
            if dims and (coord_dim not in dims):
                continue

            aux_coords[coord_name] = (coord_dim, coord_dataarray.values)

    return aux_coords


def unit_stripping_is_error(is_error : bool = True):
    if is_error:
        warnings.simplefilter("error", pint.errors.UnitStrippedWarning)
    else:
        for i,f in enumerate(warnings.filters):
            if f[0] =="error" and f[2] == pint.errors.UnitStrippedWarning:
                del warnings.filters[i]
                break


def chunked_eff_xr_matmult(
    A: xr.DataArray,
    B: xr.DataArray,
    contract_dim: str,
    sample_dim: str,
    chunksize: int = 5000,
    tmpdir: str | None = None
) -> xr.DataArray:
    """Performs a large matrix multiplication of A and B, chunking A along `sample_dim`; to avoid memory issues, streams each chunk to disk, and then rebuilds a full DataArray.

    Args:
        A: DataArray to multiply (dims include `contract_dim` and `sample_dim` among others)
        B: DataArray defining the mat-mul (dims include `contract_dim` and others)
        contract_dim: name of the axis to contract (e.g. "flat_channel")
        sample_dim: name of the axis along which to chunk (default "time")
        chunksize: max size of each chunk along `sample_dim`
        tmpdir: optional path to temp directory (auto‐created and removed if None)

    Returns:
        A new DataArray of containing the result of the matrix multiplication over `contract_dim`,
         with coords, dims, and attrs preserved. Should yield the same result as `xr.dot(A, B, dims=[contract_dim])`
         but at increased speed and with a much lower memory footprint.

    Initial Contirbutors:
        - Alexander von Lühmann | vonluehmann@tu-berlin.de | 2025
    """
    # Total samples & number of chunks
    N = A.sizes[sample_dim]
    n_chunks = int(np.ceil(N / chunksize))

    # Build a “shell” result for metadata by doing the dot on the first sample
    A0 = A.isel({sample_dim: slice(0, 1)}) 
    Xres = xr.dot(B, A0, dims=[contract_dim])

    # Prepare for raw numpy multiply
    dims_B_not = [d for d in B.dims if d != contract_dim]
    dims_A_not = [d for d in A.dims if d != contract_dim]
    B_mat = B.transpose(*dims_B_not, contract_dim).values
    A2 = A.transpose(contract_dim, *dims_A_not)

    # Create Temp directory
    cleanup = False
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
        cleanup = True
    else:
        os.makedirs(tmpdir, exist_ok=True)

    print(f"Large Matrix Multiplication: Processing {n_chunks} chunks...")

    # Stream‐compute each chunk
    file_paths = []
    for i in range(n_chunks):
        start = i * chunksize
        stop = min((i + 1) * chunksize, N)
        A_chunk = A2.isel({sample_dim: slice(start, stop)})
        C_chunk = B_mat.dot(A_chunk.values)  # raw (out_dim, chunk_len, ...)
        fn = os.path.join(tmpdir, f"chunk_{i:04d}.npy")
        np.save(fn, C_chunk)
        file_paths.append(fn)
        del A_chunk, C_chunk
        print(f"Chunk {i+1}/{n_chunks} done.")

    # Read back & concatenate along the sample axis
    arrs = [np.load(fp) for fp in sorted(file_paths)]
    axis = Xres.get_axis_num(sample_dim)
    full_arr = np.concatenate(arrs, axis=axis)

    if cleanup:
        shutil.rmtree(tmpdir)

    # create set of coordinates
    coords = {
    name: coord
    for name, coord in Xres.coords.items()
    if sample_dim not in coord.dims
    }
    sample_coords = {
        name: coord
        for name, coord in A.coords.items()
        if sample_dim in coord.dims
    }
    coords.update(sample_coords)

    # rebuild the DataArray using the Xres metadata
    result = xr.DataArray(
        data   = full_arr,
        dims   = Xres.dims,
        coords = coords,
        attrs  = Xres.attrs
    )
    # add time coords
    result.assign_coords()
    return result
