from typing import Dict

import numpy as np
import pandas as pd
import scipy.signal
import xarray as xr
import cedalion.typing as cdt
import cedalion.dataclasses as cdc
from typing import Union, List
from numpy.typing import ArrayLike


@xr.register_dataarray_accessor("cd")
class CedalionAccessor:
    def __init__(self, xarray_obj):
        """TBD."""
        self._validate(xarray_obj)
        self._obj = xarray_obj

    @staticmethod
    def _validate(obj):
        # verify there is a column latitude and a column longitude

        if not (("time" in obj.dims) and ("time" in obj.coords)):
            raise AttributeError("Missing time dimension.")

    @property
    def sampling_rate(self):
        return 1 / np.diff(self._obj.time).mean()

    def to_epochs(self, df_stim, trial_types, before, after):
        # FIXME before units
        # FIXME error handling of boundaries
        tmp = df_stim[df_stim.trial_type.isin(trial_types)]
        start = self._obj.time.searchsorted(tmp.onset - before)
        # end = ts.time.searchsorted(tmp.onset+tmp.duration)
        end = self._obj.time.searchsorted(tmp.onset + after)

        # assert len(np.unique(end - start)) == 1  # FIXME

        # find the longest number of samples to cover the epoch
        # because of numerical precision the number of samples per epoch may differ
        # by one. Larger discrepancies would have other unhandled causes.
        # Throw an error for these.
        durations = end - start
        assert np.max(durations) - np.min(durations) <= 1
        duration = np.max(durations)

        # FIXME limit reltime precision (to ns?) to avoid
        # conflicts when concatenating epochs
        reltime = np.round(self._obj.time[start[0] : end[0]] - tmp.onset.iloc[0], 9)
        epochs = xr.concat(
            [
                self._obj[:, :, start[i] : start[i] + duration].drop_vars(
                    ["time", "samples"]
                )
                for i in range(len(start))
            ],
            dim="epoch",
        )
        epochs = epochs.rename({"time": "reltime"})
        epochs = epochs.assign_coords(
            {"reltime": reltime.values, "trial_type": ("epoch", tmp.trial_type.values)}
        )

        return epochs

    def freq_filter(self, fmin, fmax, butter_order=4):
        """Apply a Butterworth frequency filter."""
        array = self._obj

        fny = array.cd.sampling_rate / 2
        b, a = scipy.signal.butter(butter_order, (fmin / fny, fmax / fny), "bandpass")

        if (units := array.pint.units) is not None:
            array = array.pint.dequantify()

        result = xr.apply_ufunc(scipy.signal.filtfilt, b, a, array)

        if units is not None:
            result = result.pint.quantify(units)

        return result


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

        from_crs = transform.dims[1]
        to_crs = transform.dims[0]
        transform_units = transform.pint.units

        assert transform_units is not None
        assert transform.shape == (4, 4)  # FIXME assume 3D
        assert from_crs in obj.dims

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
    ) -> cdt.LabeledPointCloud:
        if isinstance(label, str):  # add single point
            assert isinstance(type, cdc.PointType)
            coordinates = np.asarray(coordinates)
            assert coordinates.ndim == 1

            if label in self._obj.label:
                raise KeyError(f"there is already a point with label '{label}'")

            tmp = xr.DataArray(
                coordinates.reshape(1, -1),
                dims=self._obj.dims,
                coords={"label": ("label", [label]), "type": ("label", [type])},
            )
        else:
            assert len(label) == len(type)
            assert len(label) == len(coordinates)

            for lbl in label:
                if lbl in self._obj.label:
                    raise KeyError(f"there is already a point with label '{lbl}'")

            tmp = xr.DataArray(
                coordinates,
                dims=self._obj.dims,
                coords={"label": ("label", label), "type": ("label", type)},
            )

        tmp = tmp.pint.quantify(self._obj.pint.units)
        merged = xr.concat((self._obj, tmp), dim="label")

        return merged

    def remove(self, label):
        raise NotImplementedError()


@pd.api.extensions.register_dataframe_accessor("cd")
class StimAccessor:
    def __init__(self, pandas_obj):
        """TBD."""
        self._validate(pandas_obj)
        self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        for column_name in ["onset", "duration", "value", "trial_type"]:
            if column_name not in obj.columns:
                raise AttributeError(
                    f"Stimulus DataFame must have column {column_name}."
                )

    def rename_events(self, rename_dict):
        stim = self._obj
        for old_trial_type, new_trial_type in rename_dict.items():
            stim.loc[stim.trial_type == old_trial_type, "trial_type"] = new_trial_type
