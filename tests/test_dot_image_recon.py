import cedalion.dot as dot
import cedalion.typing as cdt
import numpy as np
import xarray as xr
import cedalion.xrutils as xrutils
import pytest
from dataclasses import dataclass




@dataclass
class ReconData:
    Adot: xr.DataArray
    x_true : cdt.NDTimeSeries
    y : cdt.NDTimeSeries
    c_meas : cdt.NDTimeSeries

@pytest.fixture
def recon_data():
    rng = np.random.default_rng(seed=42)

    nchannel = 50
    nvertex = 20
    nwavelength = 2

    channel = [f"C{c:02d}" for c in range(1, nchannel + 1)]
    wavelength = [760.0, 850.0]

    Adot = xr.DataArray(
        rng.uniform(0.1, 20.0, size=(nchannel, nvertex, nwavelength)),
        dims=["channel", "vertex", "wavelength"],
        coords={
            "is_brain": ("vertex", [True] * 15 + [False] * 5),
            "channel": channel,
            "source": ("channel", [f"S{c:02d}" for c in range(1, nchannel + 1)]),
            "detector": ("channel", [f"D{c:02d}" for c in range(1, nchannel + 1)]),
            "wavelength": wavelength,
        },
    )

    x_true = xr.DataArray(
        np.zeros(Adot.sizes["vertex"]),
        dims=["vertex"],
        coords={"is_brain": ("vertex", Adot.is_brain.values)},
    )

    x_true[5:15] = 1.0

    y = xrutils.contract(Adot, x_true, "vertex")

    c_meas = xr.DataArray(
        np.ones((nchannel, nwavelength)),
        dims=["channel", "wavelength"],
        coords={
            "channel": channel,
            "wavelength": wavelength,
        },
    )

    return ReconData(Adot, x_true, y, c_meas)


# FIXME mock implementation for spatial basis functions
@pytest.mark.parametrize("lambda_R_conc", [None, 1.])
@pytest.mark.parametrize("apply_c_meas", [False, True])
@pytest.mark.parametrize("alpha_spatial", [None, 1.])
@pytest.mark.parametrize("brain_only", [False, True])
@pytest.mark.parametrize("recon_mode", ["mua", "conc", "mua2conc"])
def test_image_recon(
    recon_data: ReconData,
    recon_mode,
    brain_only,
    alpha_spatial,
    apply_c_meas,
    lambda_R_conc,
):

    xrutils.unit_stripping_is_error(True)

    print(
        f"{recon_mode=} {brain_only=} {alpha_spatial=} {apply_c_meas=} {lambda_R_conc=}"
    )

    recon = dot.ImageRecon(
        recon_data.Adot,
        recon_mode=recon_mode,
        brain_only=brain_only,
        spatial_basis_functions=None,
        alpha_meas=1e-6,
        alpha_spatial=alpha_spatial,
        apply_c_meas=apply_c_meas,
        lambda_R_conc= lambda_R_conc,
    )

    if apply_c_meas:
        c_meas = recon_data.c_meas
    else:
        c_meas = None

    x_reco = recon.reconstruct(recon_data.y, c_meas)

    if brain_only:
        nv_brain = np.sum(recon_data.x_true.is_brain.values)
        assert x_reco.shape == (2, nv_brain)
    else:
        assert x_reco.shape == (2, len(recon_data.x_true))


    assert not np.iscomplex(x_reco.data).any()


def _reconstruct_dequantified(recon: dot.ImageRecon, y, c_meas=None):
    img = recon.reconstruct(y, c_meas).pint.dequantify()
    return img.transpose("vertex", ...)


@pytest.mark.parametrize("alpha_spatial", [None, 1.0])
@pytest.mark.parametrize("recon_mode", ["mua", "conc", "mua2conc"])
def test_image_recon_channel_order_independence(
    recon_data: ReconData, recon_mode, alpha_spatial
):
    """Reconstructions must not depend on the order of channels and wavelengths in y.

    Regression test: in recon_mode 'conc' the stacked (wavelength, channel) axis of
    y was matched to the pseudo-inverse by position instead of by label.
    """
    rng = np.random.default_rng(seed=1)

    recon = dot.ImageRecon(
        recon_data.Adot,
        recon_mode=recon_mode,
        alpha_meas=1e-6,
        alpha_spatial=alpha_spatial,
    )

    y = recon_data.y
    img_ref = _reconstruct_dequantified(recon, y)

    y_shuffled = y.isel(channel=rng.permutation(y.sizes["channel"]))
    img_shuffled = _reconstruct_dequantified(recon, y_shuffled)
    xr.testing.assert_allclose(img_ref, img_shuffled)

    y_wl_reversed = y.isel(wavelength=[1, 0])
    img_wl_reversed = _reconstruct_dequantified(recon, y_wl_reversed)
    xr.testing.assert_allclose(img_ref, img_wl_reversed)


@pytest.mark.parametrize("apply_c_meas", [False, True])
@pytest.mark.parametrize("alpha_spatial", [None, 1.0])
@pytest.mark.parametrize("recon_mode", ["mua", "conc", "mua2conc"])
def test_image_recon_pruned_channels(
    recon_data: ReconData, recon_mode, alpha_spatial, apply_c_meas
):
    """Reconstructing from a channel subset must use a pseudo-inverse of that subset.

    Regression test: previously the columns of the full pseudo-inverse were selected,
    which is equivalent to treating pruned channels as zero-valued measurements.
    """
    rng = np.random.default_rng(seed=2)

    keep = sorted(rng.choice(recon_data.Adot.channel.values, size=30, replace=False))
    y_pruned = recon_data.y.sel(channel=keep)

    c_meas = xr.DataArray(
        rng.uniform(0.5, 2.0, size=recon_data.c_meas.shape),
        dims=recon_data.c_meas.dims,
        coords=recon_data.c_meas.coords,
    )
    c_meas_pruned = c_meas.sel(channel=keep) if apply_c_meas else None

    kwargs = dict(
        recon_mode=recon_mode,
        alpha_meas=1e-6,
        alpha_spatial=alpha_spatial,
        apply_c_meas=apply_c_meas,
    )

    recon_full = dot.ImageRecon(recon_data.Adot, **kwargs)
    recon_pruned = dot.ImageRecon(recon_data.Adot.sel(channel=keep), **kwargs)

    img_full_W = _reconstruct_dequantified(recon_full, y_pruned, c_meas_pruned)
    img_pruned_W = _reconstruct_dequantified(recon_pruned, y_pruned, c_meas_pruned)

    xr.testing.assert_allclose(img_full_W, img_pruned_W)

    # after a reconstruction with pruned channels, the full data set must still be
    # reconstructed with the full pseudo-inverse
    img_full = _reconstruct_dequantified(
        recon_full, recon_data.y, c_meas if apply_c_meas else None
    )
    recon_full2 = dot.ImageRecon(recon_data.Adot, **kwargs)
    img_full2 = _reconstruct_dequantified(
        recon_full2, recon_data.y, c_meas if apply_c_meas else None
    )
    xr.testing.assert_allclose(img_full, img_full2)


@pytest.mark.parametrize("recon_mode", ["mua", "conc"])
def test_image_recon_c_meas_order_independence(recon_data: ReconData, recon_mode):
    """c_meas is matched to the sensitivity matrix by channel and wavelength labels."""
    rng = np.random.default_rng(seed=3)

    c_meas = xr.DataArray(
        rng.uniform(0.5, 2.0, size=recon_data.c_meas.shape),
        dims=recon_data.c_meas.dims,
        coords=recon_data.c_meas.coords,
    )
    c_meas_shuffled = c_meas.isel(
        channel=rng.permutation(c_meas.sizes["channel"]), wavelength=[1, 0]
    ).transpose("wavelength", "channel")

    kwargs = dict(recon_mode=recon_mode, alpha_meas=1e-3, apply_c_meas=True)

    img_ref = _reconstruct_dequantified(
        dot.ImageRecon(recon_data.Adot, **kwargs), recon_data.y, c_meas
    )
    img_shuffled = _reconstruct_dequantified(
        dot.ImageRecon(recon_data.Adot, **kwargs), recon_data.y, c_meas_shuffled
    )
    xr.testing.assert_allclose(img_ref, img_shuffled)

    noise_ref = dot.ImageRecon(recon_data.Adot, **kwargs).get_image_noise(c_meas)
    noise_shuffled = dot.ImageRecon(recon_data.Adot, **kwargs).get_image_noise(
        c_meas_shuffled
    )
    xr.testing.assert_allclose(
        noise_ref.transpose("vertex", ...), noise_shuffled.transpose("vertex", ...)
    )


@pytest.mark.parametrize("recon_mode", ["mua", "conc"])
def test_image_recon_noise_matches_covariance_propagation(
    recon_data: ReconData, recon_mode
):
    """get_image_noise must equal diag(W C W^T) for diagonal C."""
    rng = np.random.default_rng(seed=4)

    c_meas = xr.DataArray(
        rng.uniform(0.5, 2.0, size=recon_data.c_meas.shape),
        dims=recon_data.c_meas.dims,
        coords=recon_data.c_meas.coords,
    )

    recon = dot.ImageRecon(
        recon_data.Adot, recon_mode=recon_mode, alpha_meas=1e-3, apply_c_meas=True
    )
    noise = recon.get_image_noise(c_meas)

    if recon_mode == "mua":
        for wl in recon_data.Adot.wavelength.values:
            W = recon._W.sel(wavelength=wl).transpose("vertex", "channel")
            C = np.diag(c_meas.sel(wavelength=wl).sel(channel=W.channel.values).values)
            expected = np.diag(W.values @ C @ W.values.T)
            np.testing.assert_allclose(noise.sel(wavelength=wl).values, expected)
    else:
        W = recon._W.transpose("flat_vertex", "flat_channel")
        c = [
            float(c_meas.sel(wavelength=wl, channel=ch))
            for wl, ch in zip(W.wavelength.values, W.channel.values)
        ]
        expected = np.diag(W.values @ np.diag(c) @ W.values.T)
        expected = expected.reshape(2, -1)  # (chromo, vertex)
        np.testing.assert_allclose(
            noise.transpose("chromo", "vertex").values, expected
        )


def test_image_recon_unknown_channel_raises(recon_data: ReconData):
    recon = dot.ImageRecon(recon_data.Adot, recon_mode="mua", alpha_meas=1e-6)

    y = recon_data.y.assign_coords(
        channel=["X" + c for c in recon_data.y.channel.values]
    )
    with pytest.raises(ValueError):
        recon.reconstruct(y)
