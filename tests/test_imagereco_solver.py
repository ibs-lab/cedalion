import cedalion.data
import cedalion.imagereco.forward_model as fw
import cedalion.imagereco.solver as solver
import cedalion.datasets
import numpy as np
import xarray as xr
import pytest


@pytest.fixture
def Adot_stacked():
    rng = np.random.default_rng(seed=42)

    nchannel = 50
    nvertex = 100
    nwavelength = 2
    Adot = xr.DataArray(
        rng.uniform(0, 20.0, size=(nchannel, nvertex, nwavelength)),
        dims=["channel", "vertex", "wavelength"],
        coords={
            "is_brain": ("vertex", [True] * 60 + [False] * 40),
            "channel": [f"C{c:02d}" for c in range(1, nchannel + 1)],
            "wavelength": [760.0, 850.0],
        },
    )

    Adot_stacked = fw.ForwardModel.compute_stacked_sensitivity(Adot)

    assert Adot_stacked.dims == ("flat_channel", "flat_vertex")
    return Adot_stacked


def test_pseudo_inverse_stacked_reg_thikonov(Adot_stacked, alpha=0.01):
    B = solver.pseudo_inverse_stacked(Adot_stacked, alpha=0.01)
    # FIXME check computation result


def test_pseudo_inverse_stacked_reg_thikonov_spatial(Adot_stacked):
    B = solver.pseudo_inverse_stacked(Adot_stacked, alpha=0.01, alpha_spatial=1)
    # FIXME check computation result


def test_pseudo_inverse_stacked_reg_thikonov_spatial_measurent(
    Adot_stacked, alpha=0.01, alpha_spatial=1
):
    nchannel = Adot_stacked.sizes["flat_channel"]
    Cmeas_vector = np.ones(nchannel)
    Cmeas_matrix = np.ones((nchannel, nchannel))

    B = solver.pseudo_inverse_stacked(
        Adot_stacked, alpha=0.01, alpha_spatial=1, Cmeas=Cmeas_vector
    )
    # FIXME check computation result

    B = solver.pseudo_inverse_stacked(
        Adot_stacked, alpha=0.01, alpha_spatial=1, Cmeas=Cmeas_matrix
    )
    # FIXME check computation result
