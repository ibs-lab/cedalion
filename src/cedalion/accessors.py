import xarray as xr
import numpy as np
import pandas as pd
import scipy.signal


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
