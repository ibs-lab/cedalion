"""Dimension order of the sources returned by transform for trial-wise input."""

import numpy as np
import pytest
import xarray as xr

from cedalion.sigdecomp.multimodal.cca import CCA
from cedalion.sigdecomp.multimodal.mspoc import mSPoC
from cedalion.sigdecomp.multimodal.tcca import tCCA

N_TRIALS, N_TIME, FS = 12, 50, 10.0


def make_trials(n_features: int, seed: int, upsample: int = 1) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    n_time, fs = N_TIME * upsample, FS * upsample
    return xr.DataArray(
        rng.standard_normal((N_TRIALS, n_time, n_features)),
        dims=("epoch", "time", "channel"),
        coords={
            "time": np.arange(n_time) / fs,
            "channel": [f"c{seed}{i}" for i in range(n_features)],
            "trial_type": ("epoch", ["a", "b"] * (N_TRIALS // 2)),
        },
    )


@pytest.fixture
def trials():
    return make_trials(4, seed=0), make_trials(3, seed=1)


def test_cca_trial_sources_are_trial_sample_component(trials):
    X, Y = trials
    cca = CCA(N_components=2, random_state=0)
    cca.fit(X, Y, trial_name="epoch")

    Sx, Sy = cca.transform(X, Y, trial_name="epoch")

    assert Sx.dims == ("epoch", "time", "CCA_X")
    assert Sy.dims == ("epoch", "time", "CCA_Y")
    assert Sx.shape == Sy.shape == (N_TRIALS, N_TIME, 2)
    assert Sx.trial_type.dims == ("epoch",)


def test_cca_trial_sources_match_continuous_sources(trials):
    """Each trial transformed on its own gives the same (time, component) array."""
    X, Y = trials
    cca = CCA(N_components=2, random_state=0)
    cca.fit(X, Y, trial_name="epoch")

    Sx, Sy = cca.transform(X, Y, trial_name="epoch")
    Sx_0, Sy_0 = cca.transform(X.isel(epoch=0), Y.isel(epoch=0))

    assert Sx_0.dims == ("time", "CCA_X")
    np.testing.assert_allclose(Sx[0].values, Sx_0.values)
    np.testing.assert_allclose(Sy[0].values, Sy_0.values)


def test_cca_trial_sources_do_not_depend_on_input_dim_order(trials):
    X, Y = trials
    cca = CCA(N_components=2, random_state=0)
    cca.fit(X, Y, trial_name="epoch")

    Sx, _ = cca.transform(X, Y, trial_name="epoch")
    Sx_t, _ = cca.transform(
        X.transpose("channel", "time", "epoch"),
        Y.transpose("time", "channel", "epoch"),
        trial_name="epoch",
    )

    assert Sx_t.dims == ("epoch", "time", "CCA_X")
    np.testing.assert_allclose(Sx_t.values, Sx.values)


@pytest.mark.parametrize("shift_source", [False, True])
def test_tcca_trial_sources_are_trial_sample_component(trials, shift_source):
    X, Y = trials
    tcca = tCCA(
        N_components=2,
        time_lags=[0.0, 0.5, 1.0],
        random_state=0,
        shift_source=shift_source,
    )
    tcca.fit(X, Y, trial_name="epoch")

    if shift_source:
        with pytest.warns(FutureWarning, match="shift_source is deprecated"):
            Sx, Sy = tcca.transform(X, Y, trial_name="epoch")
    else:
        Sx, Sy = tcca.transform(X, Y, trial_name="epoch")

    assert Sx.dims == ("epoch", "time", "tCCA_X")
    assert Sy.dims == ("epoch", "time", "tCCA_Y")
    assert Sx.sizes["epoch"] == N_TRIALS and Sx.sizes["tCCA_X"] == 2


def test_mspoc_trial_sources_are_trial_sample_component():
    """mSPoC already returned the canonical order; this guards the shared convention."""
    X, Y = make_trials(4, seed=2, upsample=10), make_trials(3, seed=3)
    mspoc = mSPoC(N_components=2, time_lags=[0.0, 0.5, 1.0], random_state=0,
                  l2_reg="lw")
    mspoc.fit(X, Y, trial_name="epoch")

    Sx, Sy, Px, sx = mspoc.transform(
        X, Y, trial_name="epoch", return_unfiltered=True
    )

    # Px and sx follow the same order, so a shared feature extraction can use any
    assert Sx.dims == Px.dims == sx.dims == ("epoch", "time", "mSPoC_X")
    assert Sy.dims == ("epoch", "time", "mSPoC_Y")
    assert Sx.sizes["epoch"] == N_TRIALS and Sx.sizes["mSPoC_X"] == 2
