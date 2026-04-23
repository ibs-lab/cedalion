"""Utility functions for xarray objects."""

import warnings

import numpy as np
import pint
import xarray as xr
import scipy.sparse


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


def spatial_dim(data_array: xr.DataArray) -> str:
    """Return the spatial dimension name used by the DataArray."""
    for dim in ("channel", "parcel", "vertex"):
        if dim in data_array.dims:
            return dim
    raise ValueError("could not determine spatial dimension")


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
    """Make UnitStrippedWarning an error.

    This function is useful for debugging cases where UnitStrippedWarning are issued
    in third-party libraries. By raising an exception one can use the debugger to
    find the occurence in cedalion or user code that triggers the warning.
    """
    if is_error:
        warnings.simplefilter("error", pint.errors.UnitStrippedWarning)
    else:
        for i,f in enumerate(warnings.filters):
            if f[0] =="error" and f[2] == pint.errors.UnitStrippedWarning:
                del warnings.filters[i]
                break


def unit_stripping_is_quiet(is_quiet : bool = True):
    """Silence UnitStrippedWarnings.

    It is not recommended to use this functions. Instead consider using
    `unit_stripping_is_error` to find the code responsible for issuing the warning
    and fix it there.
    """
    if is_quiet:
        warnings.filterwarnings("ignore", category=pint.errors.UnitStrippedWarning)
    else:
        warnings.filters[:] = [
            f for f in warnings.filters
            if not (f[0] == "ignore" and f[2] is pint.errors.UnitStrippedWarning)
        ]


def drop_duplicate_dimensions(array : xr.DataArray) -> xr.DataArray:
    """Remove dimensions in which all array values are identical.

    During stacking and unstacking of dimensions, coordinate arrays can occur that
    are attributed to multiple dimensions although their values change only along
    a single dimensions. This function reduces the array to that single dimension.
    """

    drop_dims = []

    for dim in array.dims:
        ref = array.isel({dim : 0})
        if (array == ref).all(): # array values do not change along this dimension
            drop_dims.append(dim)

    # drop dimensions
    reduced_array = array.isel({dim: 0 for dim in drop_dims})

    # drop (now scalar) coordinates that belong to removed dimensions
    reduced_array = reduced_array.drop_vars(drop_dims)

    return reduced_array


def unstack(
    array: xr.DataArray, unstack_dim: str, stacked_dims: tuple[str]
) -> xr.DataArray:
    """Unstack a stacked DataArray.

    This function unstacks a DataArray in which dimensions 'stacked_dims' have
    been stacked into the dimension 'unstack_dim'. The function further processes
    unstacked coordinate arrays, so that they are attributed only to their respective
    dimension.

    Args:
        array: the stacked DataArray
        unstack_dim: the dimension to unstack
        stacked_dims: The dimensions that were stacked together in the order
            given to DataArray.stack.

    Returns:
        The unstacked array.
    """
    if unstack_dim not in array.dims:
        raise ValueError(f"cannot unstack missing dimension '{unstack_dim}'.")

    #coords = ("chromo", "vertex")
    for coord in stacked_dims:
        if coord not in array.coords:
            raise ValueError(f"array misses coordinate '{coord}'.")


    if unstack_dim not in array.indexes:
        array = array.set_xindex(stacked_dims)

    unstacked = array.unstack(unstack_dim)

    # other coorindates of the unstack_dim dimension that are not part
    # of the MultiIndex. These coordinates will be assigned to all stacked_dims
    # even if they initially belonged only to one of them.
    other_dims = [
        c
        for c, da in array.coords.items()
        if unstack_dim in da.dims and c not in [unstack_dim, *stacked_dims]
    ]

    # while they are assigned to all stacked_dims, they may vary only along
    # a single dimension. remove the other duplicate dimensions.
    for unstack_dim in other_dims:
        coords_array = drop_duplicate_dimensions(unstacked.coords[unstack_dim])
        unstacked = unstacked.assign_coords({unstack_dim: coords_array})

    return unstacked


def contract(a1: xr.DataArray, a2: xr.DataArray, dim: str | list[str]) -> xr.DataArray:
    """Apply xr.dot after asserting compatible shapes.

    xr.dot will silently multiply arrays along dimensions which differ in shape if
    these arrays have an overlap in coordinates. This function requires an exact
    match in coordinates before calling xr.dot.

    Args:
        a1: first operand
        a2: second operand
        dim: dimension(s) to contract over

    Returns:
        the result of xr.dot.
    """

    with xr.set_options(arithmetic_join="exact"):
        return xr.dot(a1, a2, dim=dim)

def transpose_like(
    a: xr.DataArray, target: xr.DataArray, dim_map: dict[str, str] | None = None
) -> xr.DataArray:
    """Transpose the dims in a to match the ordering in target."""

    target_dims = list(target.dims)
    if not dim_map:
        dim_map = {}

    new_order = []
    for d in a.dims:
        if d in target_dims:
            new_order.append(target_dims.index(d))
        elif d in dim_map and dim_map[d] in target_dims:
            new_order.append(target_dims.index(dim_map[d]))
        else:
            raise ValueError(f"could not find dim '{d}' in target.")
    new_order = np.argsort(new_order)
    return a.transpose(*[a.dims[i] for i in new_order])


def dot_dataarray_csr(
    a: xr.DataArray, b: scipy.sparse, bdims: list[str, str]
) -> xr.DataArray:
    """Multiply a DataArray and sparse array along one dimension."""

    # figure out the common dimension along which to multiply
    common_dim = set(a.dims) & set(bdims)
    if len(common_dim) != 1:
        raise ValueError("a and b must share a single common dimension!")
    common_dim = common_dim.pop()

    if a.sizes[common_dim] != b.shape[bdims.index(common_dim)]:
        raise ValueError(
            f"shape of common dimension '{common_dim}' does not match. "
            f"a: {a.sizes[common_dim]} "
            f"b: {b.shape[bdims.index(common_dim)]}."
        )

    # move common dimension in DataArray to the end and flatten others
    aT = a.transpose(..., common_dim)
    aT2D = aT.data.reshape(-1, aT.shape[-1])

    if common_dim == bdims[0]:
        tmp = (aT2D @ b).reshape(*aT.shape[:-1], b.shape[1])
        new_dim = bdims[1]
    else:
        tmp = (aT2D @ b.T).reshape(*aT.shape[:-1], b.shape[0])
        new_dim = bdims[0]

    dims = list(aT.dims)[:-1] + [new_dim]

    result = xr.DataArray(
        tmp,
        dims=dims,
        coords=coords_from_other(aT, dims=dims),
        attrs=aT.attrs
    )

    result = transpose_like(result, a, dim_map={new_dim : common_dim})

    return result


def check_units(array : xr.DataArray, dimension : str):
    """Check the dimensionality of quantified and dequantified DataArrays."""

    if array.pint.units is None:
        if (units_str := array.attrs.get("units", None)) is None:
            # fail or return False?
            #raise ValueError("Array is not quantified and has no units in .attrs!")
            return False
        else:
            units = pint.Unit(units_str)
    else:
        units = array.pint.units

    return (1*units).check(dimension)
