import cedalion.data
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
