import numpy as np
import xarray as xr

from cedalion.mlutils.features import epoch_features


def _epochs(dims):
    reltime = np.arange(-2, 10, 0.5)
    slopes = np.array([0.5, -1.0, 2.0])  # per epoch
    data = np.zeros((3, 2, 2, len(reltime)))  # epoch, channel, chromo, reltime
    for i_epoch, slope in enumerate(slopes):
        data[i_epoch] = slope * reltime[None, None, :] + 1.0
    epochs = xr.DataArray(
        data,
        dims=["epoch", "channel", "chromo", "reltime"],
        coords={
            "epoch": np.arange(3),
            "channel": ["S1D1", "S1D2"],
            "chromo": ["HbO", "HbR"],
            "reltime": reltime,
        },
    )
    return epochs.transpose(*dims), slopes


def test_epoch_features_slope_independent_of_dimension_order():
    """The slope feature is correct for any position of the reltime dimension.

    Regression test: the reltime axis was a bare numpy array that broadcast along
    the last dimension, so inputs with reltime not being the last dimension
    crashed or produced wrong slopes.
    """
    for dims in [
        ("epoch", "channel", "chromo", "reltime"),
        ("epoch", "channel", "reltime", "chromo"),
        ("reltime", "epoch", "channel", "chromo"),
    ]:
        epochs, slopes = _epochs(dims)
        features = epoch_features(epochs, ["slope"])
        assert features.dims == ("epoch", "feature")
        np.testing.assert_allclose(
            features.values, np.repeat(slopes[:, None], features.sizes["feature"], 1)
        )
