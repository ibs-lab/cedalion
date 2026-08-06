"""Feature extraction from epoched fNIRS data for use with scikit-learn pipelines."""

import numpy as np
import xarray as xr
from typing import Literal


EpochFeatureType = Literal["slope", "mean", "max", "min", "auc"]


def epoch_features(
    epochs: xr.DataArray,
    feature_types: list[EpochFeatureType],
    reltime_slices: dict[EpochFeatureType, slice] | None = None,
):
    """Extract scalar features from epoched data for use in ML classifiers.

    For each requested feature type, a scalar value is computed over the
    ``"reltime"`` axis (optionally restricted to a sub-window).  All non-epoch
    dimensions (channel, chromo, …) are then stacked into a flat ``"feature"``
    dimension so the result is suitable as a 2-D feature matrix for
    scikit-learn estimators (rows = epochs, columns = features).

    Args:
        epochs: DataArray with at least an ``"epoch"`` dimension and a
            ``"reltime"`` dimension.
        feature_types: One or more of ``"slope"``, ``"mean"``, ``"max"``,
            ``"min"``, ``"auc"``.  A string is also accepted as a shorthand for
            a single-element list.
        reltime_slices: Optional mapping from feature type to a
            :class:`slice` of relative-time values used to restrict the
            window for that feature.  Unspecified feature types use the full
            ``reltime`` range.

    Returns:
        xr.DataArray with dimensions ``(epoch, feature)`` where ``feature``
        is a multi-index stacking all non-epoch, non-reltime dimensions and
        the ``feature_type`` label.

    Raises:
        ValueError: If an unrecognised feature type is requested.
    """
    if isinstance(feature_types, str):
        feature_types = [feature_types]

    if reltime_slices is None:
        reltime_slices = {}

    epochs = epochs.pint.dequantify()

    output_features = []

    for feature_type in feature_types:
        if feature_type in reltime_slices:
            # restrict the rel. time range over which the feature is computed
            sliced = epochs.sel(reltime=reltime_slices[feature_type])
        else:
            # no slice provided, use the whole time range
            sliced = epochs

        match feature_type:
            case "slope":
                x = sliced.reltime.values

                # Compute deviations from the mean
                x_dev = x - x.mean()
                denom = np.sum(x_dev**2)
                # Compute mean_y over 'reltime'
                y_dev = sliced - sliced.mean("reltime")
                # Compute numerator: sum over 'reltime' dimension
                numerator = (x_dev * y_dev).sum("reltime")
                # Compute slope
                f = numerator / denom

            case "mean":
                f = sliced.mean("reltime")

            case "max":
                f = sliced.max("reltime")

            case "min":
                f = sliced.min("reltime")

            case "auc":
                f = (sliced * np.diff(sliced.reltime.values).mean()).sum("reltime")

            case _:
                raise ValueError(f"Unknown feature_type: {feature_type}")


        # add feature dimension
        f = f.expand_dims(dim={"feature_type": [feature_type]})
        output_features.append(f)

    output_features = xr.concat(output_features, dim="feature_type")

    # stack all remaining dimensions except for the epoch dimension
    # (channel/vertex,parcel, chromo, wavelength, ...)

    dims_to_stack = [dim for dim in output_features.dims if dim != "epoch"]

    output_features = output_features.stack({"feature" : dims_to_stack} )

    return output_features

