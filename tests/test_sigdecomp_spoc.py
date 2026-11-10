import numpy as np
import xarray as xr

import cedalion.dataclasses as cdc
from cedalion.sigdecomp.unimodal.spoc import SPoC


def _spoc_test_data(seed=0):
    rng = np.random.default_rng(seed)
    nx, n_epochs, epoch_len = 6, 200, 50
    nt = n_epochs * epoch_len
    fs = 50.0
    t = np.arange(nt) / fs

    z = rng.standard_normal(n_epochs)
    z = (z - z.mean()) / z.std()
    s_target = rng.standard_normal(nt) * np.repeat(np.exp(0.5 * z), epoch_len)
    sources = np.vstack([s_target, rng.standard_normal((nx - 1, nt))])
    x = rng.standard_normal((nx, nx)) @ sources

    x = cdc.build_timeseries(
        x, ["channel", "time"], t, [f"ch{i}" for i in range(nx)], "V", "s"
    )
    z = xr.DataArray(
        z, dims=["time"], coords={"time": (np.arange(n_epochs) + 0.5) * epoch_len / fs}
    )
    return x, z, n_epochs


def test_spoc_number_of_components():
    """SPoC(n_comp=k) returns exactly k components, the k largest of the full solve.

    Regression test: the eigenvalue index range was inclusive on both ends, so k+1
    components were returned.
    """
    x, z, _ = _spoc_test_data()

    full = SPoC()
    full.fit(x, z)

    for n_comp in [1, 2, 3]:
        model = SPoC(n_comp=n_comp)
        scores = model.fit(x, z)
        assert len(scores) == n_comp
        assert model.W.sizes["component"] == n_comp
        np.testing.assert_allclose(scores, full.scores[:n_comp])


def test_spoc_transform_with_bandpower():
    """transform(get_bandpower=True) returns standardized band power per epoch.

    Regression test: np.linspace was called with 0-d DataArrays and raised.
    """
    x, z, n_epochs = _spoc_test_data()
    model = SPoC(n_comp=1)
    model.fit(x, z)

    s, s_power = model.transform(x, get_bandpower=True, Ne=n_epochs)

    assert s_power.sizes["time"] == n_epochs
    # the first component's power co-modulates with the target
    corr = np.corrcoef(s_power.isel(component=0).values, z.values)[0, 1]
    assert abs(corr) > 0.5
