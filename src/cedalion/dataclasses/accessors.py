"""Accessors for Cedalion data types."""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pandas as pd
import statsmodels
import xarray as xr
from numpy.typing import ArrayLike
from typing import Callable

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import Quantity, units
from cedalion.errors import CRSMismatchError
from cedalion.sigproc.epochs import to_epochs
from cedalion.sigproc.frequency import freq_filter


@xr.register_dataarray_accessor("cd")
class CedalionAccessor:
    """Accessor for time series data stored in xarray DataArrays."""

    def __init__(self, xarray_obj):
        """Initialize the CedalionAccessor.

        Args:
            xarray_obj (xr.DataArray): The DataArray to which this accessor is attached.
        """
        self._validate(xarray_obj)
        self._obj = xarray_obj

    @staticmethod
    def _validate(obj):
        """Make sure the DataArray has the required dimensions and coordinates."""

        if not (("time" in obj.dims) and ("time" in obj.coords)):
            raise AttributeError("Missing time dimension.")

    @property
    def sampling_rate(self):
        """Return the sampling rate of the time series.

        The sampling rate is calculated as the reciprocal of the mean time difference
            between consecutive samples.
        """
        return 1 / np.diff(self._obj.time).mean()

    def to_epochs(
        self,
        df_stim: pd.DataFrame,
        trial_types: list[str],
        before: cdt.QTime,
        after: cdt.QTime,
    ):
        """Extract epochs from the time series based on stimulus events.

        Args:
            df_stim: DataFrame containing stimulus events.
            trial_types: List of trial types to include in the epochs.
            before: Time before stimulus event to include in epoch.
            after: Time after stimulus event to include in epoch.

        Returns:
            xarray.DataArray: Array containing the extracted epochs.
        """

        return to_epochs(self._obj, df_stim, trial_types, before, after)

    def freq_filter(self, fmin, fmax, butter_order=4):
        """Applys a Butterworth filter.

        Args:
            fmin (float): The lower cutoff frequency.
            fmax (float): The upper cutoff frequency.
            butter_order (int): The order of the Butterworth filter.

        Returns:
            result (xarray.DataArray): The filtered time series.
        """
        array = self._obj

        # FIXME accept unit-less parameters and interpret them as Hz
        if not isinstance(fmin, Quantity):
            fmin = fmin * units.Hz

        if not isinstance(fmax, Quantity):
            fmax = fmax * units.Hz

        return freq_filter(array, fmin, fmax, butter_order)


@xr.register_dataarray_accessor("points")
class PointsAccessor:
    def __init__(self, xarray_obj):
        """TBD."""
        self._validate(xarray_obj)
        self._obj = xarray_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude

        if not (("label" in obj.dims) and ("label" in obj.coords)):
            raise AttributeError(
                "This dataarray does not look like a labled point cloud"
            )

    def to_homogeneous(self):
        tmp = self._obj.pint.dequantify()
        augmented = np.hstack((tmp.values, np.ones((len(tmp), 1))))
        result = xr.DataArray(
            augmented, dims=tmp.dims, coords=tmp.coords, attrs=tmp.attrs
        )
        result = result.pint.quantify()
        return result

    def rename(self, translations: Dict[str, str]):
        new_labels = [translations.get(i, i) for i in self._obj.label.values]
        return self._obj.assign_coords({"label": new_labels})

    def common_labels(self, other: xr.DataArray) -> List[str]:
        """Return labels contained in both LabledPointClouds."""
        assert ("label" in other.dims) and ("label" in other.coords)

        return list(set(self._obj.label.values).intersection(other.label.values))

    def apply_transform(self, transform: Union[cdt.AffineTransform, np.ndarray]):
        if isinstance(transform, xr.DataArray):
            # FIXME validate schema
            return self._apply_xr_transform(transform)
        elif isinstance(transform, np.ndarray):
            return self._apply_numpy_transform(transform)
        else:
            raise ValueError(
                "transform must be either a cdt.AffineTransform or a " "4x4 numpy array"
            )

    def _apply_xr_transform(self, transform: cdt.AffineTransform):
        obj = self._obj

        points_crs = self.crs
        from_crs = transform.dims[1]
        to_crs = transform.dims[0]
        transform_units = transform.pint.units

        assert transform_units is not None
        assert transform.shape == (4, 4)  # FIXME assume 3D
        if from_crs not in obj.dims:
            raise CRSMismatchError.wrong_transform(points_crs, transform.dims)

        transform = transform.pint.dequantify()

        transformed = self._apply_numpy_transform(transform.values, to_crs)

        if transformed.pint.units is not None:
            new_units = transformed.pint.units * transform_units
            transformed = transformed.pint.dequantify().pint.quantify(new_units)
        else:
            raise NotImplementedError()

        return transformed

    def _apply_numpy_transform(self, transform: np.ndarray, to_crs=None):
        obj = self._obj
        assert transform.shape == (4, 4)  # FIXME assume 3D

        if obj.pint.units is not None:
            units = obj.pint.units
            obj = obj.pint.dequantify()
            was_quantified = True
        elif unit_str := obj.attrs.get("units", None) is not None:
            # units = cedalion.units.Unit(unit_str)
            was_quantified = False
        else:
            units = None
            was_quantified = False

        if to_crs is None:
            to_crs = obj.dims[1]

        rzs = transform[:-1, :-1]  # rotations, zooms, shears
        trans = transform[:-1, -1]  # translatations
        transformed = obj.values @ rzs.T + trans

        transformed = xr.DataArray(
            transformed, dims=[obj.dims[0], to_crs], coords=obj.coords
        )

        if was_quantified:
            transformed = transformed.pint.quantify(units)
        else:
            if unit_str is not None:
                transformed.attrs["units"] = unit_str

        return transformed

    @property
    def crs(self):
        assert len(self._obj.dims) == 2
        return [d for d in self._obj.dims if d != "label"][0]

    def set_crs(self, value: str):
        current = self.crs
        return self._obj.rename({current: value})

    def add(
        self,
        label: Union[str, List[str]],
        coordinates: ArrayLike,
        type: Union[cdc.PointType, List[cdc.PointType]],
        group: Union[str, List[str]] = None,
    ) -> cdt.LabeledPointCloud:
        # Handle the single point case
        if isinstance(label, str):
            assert isinstance(
                type, cdc.PointType
            ), "Type must be a PointType for a single label"
            coordinates = np.asarray(coordinates)
            assert (
                coordinates.ndim == 1
            ), "Coordinates for a single point must be 1-dimensional"

            if label in self._obj.label:
                raise KeyError(f"there is already a point with label '{label}'")

            coords_dict = {"label": ("label", [label]), "type": ("label", [type])}
            if group is not None:
                assert isinstance(
                    group, str
                ), "Group must be a string for a single label"
                coords_dict["group"] = ("label", [group])

            tmp = xr.DataArray(
                coordinates.reshape(1, -1),
                dims=self._obj.dims,
                coords=coords_dict,
            )

        # Handle the case where multiple points are added
        else:
            assert len(label) == len(type), "Labels and types must have the same length"
            if group is not None:
                assert len(label) == len(
                    group
                ), "Labels and groups must have the same length"

            for lbl in label:
                if lbl in self._obj.label:
                    raise KeyError(f"there is already a point with label '{lbl}'")

            coords_dict = {"label": ("label", label), "type": ("label", type)}
            if group is not None:
                coords_dict["group"] = ("label", group)

            tmp = xr.DataArray(
                coordinates,
                dims=self._obj.dims,
                coords=coords_dict,
            )

        # Quantify the temporary DataArray with units from the original object
        tmp = tmp.pint.quantify(self._obj.pint.units)

        # Merge the new points into the existing DataArray
        merged = xr.concat((self._obj, tmp), dim="label")

        return merged

    def remove(self, label):
        raise NotImplementedError()


@pd.api.extensions.register_dataframe_accessor("cd")
class StimAccessor:
    """Accessor for stimulus DataFrames."""

    def __init__(self, pandas_obj):
        """Initialize the StimAccessor.

        Args:
            pandas_obj (pd.DataFrame): The pandas DataFrame to which this accessor is
                attached.
        """
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        """Make sure the DataFrame has the required columns for stimulus data."""
        for column_name in ["onset", "duration", "value", "trial_type"]:
            if column_name not in obj.columns:
                raise AttributeError(
                    f"Stimulus DataFame must have column {column_name}."
                )

    def rename_events(self, rename_dict):
        """Renames trial types in the DataFrame based on the provided dictionary.

        Args:
            rename_dict (dict): A dictionary with the old trial type as key and the new
                trial type as value.
        """
        stim = self._obj
        for old_trial_type, new_trial_type in rename_dict.items():
            stim.loc[stim.trial_type == old_trial_type, "trial_type"] = new_trial_type

    def conditions(self):
        return self._obj.trial_type.unique()

    # FIXME obsolete?
    def to_xarray(self, time: xr.DataArray):
        stim = self._obj
        conds = self.conditions()
        stim_arr = xr.DataArray(
            np.zeros((time.shape[0], len(conds))),
            dims=["time", "condition"],
            coords={"time": time, "condition": conds},
        )
        for index, row in stim.iterrows():
            stim_arr.loc[row.onset, row.trial_type] = 1
        return stim_arr



class SMCallableWrapper:
    """Wraps a method of statsmodel's result object."""

    def __init__(self, accessor, attr_name):
        self._accessor = accessor
        self._attr_name = attr_name

    def __call__(self, *args, **kwargs):
        array = self._accessor._array

        first_obj = array[0,0].item()
        function = getattr(first_obj, self._attr_name)

        first_result = function(*args, **kwargs)

        result = self._accessor._build_array(self._attr_name, first_result)

        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                f = getattr(array[i, j].item(), self._attr_name)
                x = f(*args, **kwargs)
                if np.isscalar(first_result):
                    result[i, j] = x
                elif isinstance(first_result, pd.Series):
                    result[i, j, :] = x
                elif isinstance(first_result, pd.DataFrame):
                    result[i, j, :, :] = x.values
                elif isinstance(
                    first_result, statsmodels.stats.contrast.ContrastResults
                ):
                    result[i,j] = x
                else:
                    raise NotImplementedError()

        return result


@xr.register_dataarray_accessor("sm")
class StatsModelsAccessor:
    """Accessor for DataArrays containing statsmodel results."""

    def __init__(self, array : xr.DataArray):
        """Initialize the accessor.

        Args:
            array: The DataArray to which this accessor is attached.
        """
        self._validate(array)
        self._array = array

    @staticmethod
    def _validate(obj : xr.DataArray) :
        """Check that data array contains stats model results."""

        assert obj.ndim == 2

        for i in obj.values.flatten():
            if not isinstance(
                i,
                (
                    statsmodels.regression.linear_model.RegressionResultsWrapper,
                    statsmodels.stats.contrast.ContrastResults,
                ),
            ):
                raise ValueError("data array may contain only RegressionResultsWrapper")

    def _build_array(self, attr_name, attr_result):
        """Helper function to construct the result array."""

        try:
            regressors = self._array[0,0].item().params.index
        except AttributeError: # ContrastResults carry no information about regressors
            regressors = None

        def is_scalar_ndarray(x):
            return isinstance(x, np.ndarray) and (x.ndim == 0 or x.shape == (1,1))

        def is_regressor(x):
            if regressors is not None:
                return (len(x) == len(regressors)) and all(x == regressors)
            else:
                return False

        if np.isscalar(attr_result):
            return xr.zeros_like(self._array, dtype=np.dtype(type(attr_result)))
        elif is_scalar_ndarray(attr_result):
            return xr.zeros_like(self._array, dtype=np.dtype(type(attr_result[()])))
        elif isinstance(attr_result, pd.Series):
            if is_regressor(attr_result.index):
                dim_name = "regressor"
            else:
                dim_name = attr_name

            return (
                xr.zeros_like(self._array, dtype=attr_result.values.dtype)
                .expand_dims({dim_name: attr_result.index}, axis=-1)
                .copy()
            )
        elif isinstance(attr_result, np.ndarray) and attr_result.ndim == 1:
            dim_name = attr_name
            return (
                xr.zeros_like(self._array, dtype=attr_result.dtype)
                .expand_dims({dim_name : len(attr_result)}, axis=-1)
                .copy()
            )
        elif isinstance(attr_result, pd.DataFrame):
            if is_regressor(attr_result.index):
                row_name = "regressor"
            else:
                row_name = f"{attr_name}"
            if is_regressor(attr_result.columns):
                col_name = "regressor"
            else:
                col_name = f"{attr_name}"

            if row_name == col_name:
                row_name = f"{row_name}_r"
                col_name = f"{col_name}_c"

            return (
                xr.zeros_like(self._array, dtype=attr_result.values.dtype)
                .expand_dims(
                    {row_name: attr_result.index, col_name: attr_result.columns},
                    axis=[-2, -1],
                )
                .copy()
            )
        elif isinstance(attr_result, statsmodels.stats.contrast.ContrastResults):
            return xr.zeros_like(self._array)
        else:
            raise NotImplementedError(f"unsupported type '{type(attr_result)}'")


    def __getattr__(self, name):
        try:
            first_obj = self._array[0,0].item()
            attr = getattr(first_obj, name)
        except AttributeError:
            raise AttributeError(
                f"objects of type {str(type(first_obj))} don't have "
                f"an attribute with name '{name}'."
            )

        if callable(attr):
            result = SMCallableWrapper(self, name)
        else:
            result = self._build_array(name, attr)

            for i in range(self._array.shape[0]):
                for j in range(self._array.shape[1]):
                    x = getattr(self._array[i, j].item(), name)

                    if np.isscalar(attr):
                        result[i, j] = x
                    elif isinstance(attr, pd.Series):
                        result[i, j, :] = x
                    elif isinstance(attr, pd.DataFrame):
                        result[i, j, :, :] = x.values
                    else:
                        raise NotImplementedError()

        return result

    def map(self, func: Callable, name: str = "map"):
        """Applies a callable to each object in the array and returns a xr.DataArray.

        Args:
            func: a Callable that expects one argument, the object
            name: used to name dimensions in the result array

        Returns:
            an xr.DataArray whose first to dimensions match our array and that holds
            in each cell the result of the function call.
        """
        first_result = func(self._array[0,0].item())
        result = self._build_array(name, first_result)

        for i in range(self._array.shape[0]):
            for j in range(self._array.shape[1]):
                x = func(self._array[i, j].item())
                result[i, j] = x

        return result


    def regressor_variances(self):
        try:
            regressors = self._array[0, 0].item().params.index
        except AttributeError:
            raise NotImplementedError() #  FIXME better direct test for ContrastResult
        return self.map(
            lambda i: pd.Series(np.diagonal(i.cov_params()), index=regressors)
        )
