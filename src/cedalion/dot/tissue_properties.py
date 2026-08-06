"""Tissue properties for light transport simulation."""

from enum import Enum, auto
from typing import Dict
from warnings import warn

import numpy as np
import xarray as xr


class TissueType(Enum):
    """Canonical tissue-type labels used to look up optical properties."""

    SKIN = auto()
    SKULL = auto()
    DM = auto()
    CSF = auto()
    GM = auto()
    WM = auto()
    OTHER = auto()


TISSUE_LABELS: Dict[str, TissueType] = {
    "skin": TissueType.SKIN,
    "scalp": TissueType.SKIN,
    "skull": TissueType.SKULL,
    "bone": TissueType.SKULL,
    "dura": TissueType.DM,
    "dura mater": TissueType.DM,
    "dm": TissueType.DM,
    "csf": TissueType.CSF,
    "cerebral spinal fluid": TissueType.CSF,
    "gm": TissueType.GM,
    "gray matter": TissueType.GM,
    "brain": TissueType.GM,
    "wm": TissueType.WM,
    "white matter": TissueType.WM,
}

# FIXME units, reference

# fmt: off
TISSUE_PROPS_SCATTERING = {
    TissueType.SKIN  : 0.6600,
    TissueType.SKULL : 0.8600,
    TissueType.DM    : 0.6600,
    TissueType.CSF   : 0.0100,
    TissueType.GM    : 1.1000,
    TissueType.WM    : 1.1000,
    TissueType.OTHER : 0.8600,
}

TISSUE_PROPS_ABSORPTION = {
    TissueType.SKIN  : 0.0191,
    TissueType.SKULL : 0.0136,
    TissueType.DM    : 0.0191,
    TissueType.CSF   : 0.0026,
    TissueType.GM    : 0.0186,
    TissueType.WM    : 0.0186,
    TissueType.OTHER : 0.0191,
}

TISSUE_PROPS_ANISOTROPY = {
    TissueType.SKIN  : 0.0010,
    TissueType.SKULL : 0.0010,
    TissueType.DM    : 0.0010,
    TissueType.CSF   : 0.0010,
    TissueType.GM    : 0.0010,
    TissueType.WM    : 0.0010,
    TissueType.OTHER : 0.0010,
}

TISSUE_PROPS_REFRACTION = {
    TissueType.SKIN  : 1.0000,
    TissueType.SKULL : 1.0000,
    TissueType.DM    : 1.0000,
    TissueType.CSF   : 1.0000,
    TissueType.GM    : 1.0000,
    TissueType.WM    : 1.0000,
    TissueType.OTHER : 1.0000,
}
# fmt: on

# FIXME allow for wavelength dependencies


def get_tissue_properties(
    segmentation_masks: xr.DataArray, wavelengths: list
) -> np.ndarray:
    """Assemble a tissue-property array for Monte Carlo light transport simulation.

    For each tissue type present in ``segmentation_masks`` the absorption,
    scattering, anisotropy, and refraction coefficients are looked up from the
    module-level dictionaries and stored in the output array.  Index 0 is
    reserved for the background (vacuum).

    Args:
        segmentation_masks: xr.DataArray with dimension ``"segmentation_type"``
            whose integer values encode tissue identity.
        wavelengths: List of wavelengths for which properties are required.
            Currently the properties are wavelength-independent (FIXME).

    Returns:
        NumPy array of shape ``(n_tissues + 1, 4, n_wavelengths)`` where axis 1
        encodes ``[absorption, scattering, anisotropy, refraction]``.

    Raises:
        ValueError: If a segmentation type string is not in :data:`TISSUE_LABELS`.
    """
    ntissues = segmentation_masks.sizes["segmentation_type"] + 1
    nwavelengths = len(wavelengths)
    tissue_props = np.zeros((ntissues, 4, nwavelengths))
    tissue_props[0, :, :] = np.asarray([0.0, 0.0, 1.0, 1.0])[:, None]  # background

    for i_wl in range(nwavelengths):

        for st in segmentation_masks.segmentation_type.values:
            m = segmentation_masks.sel(segmentation_type=st).values
            int_labels = np.unique(m[m > 0])
            if len(int_labels) == 0:
                warn("Segmentation type %s is empty." % st)
                continue
            int_label = int_labels.item()

            if (tissue_type := TISSUE_LABELS.get(st, None)) is None:
                raise ValueError(f"unknown tissue type '{st}'")

            tissue_props[int_label, 0, i_wl] = TISSUE_PROPS_ABSORPTION[tissue_type]
            tissue_props[int_label, 1, i_wl] = TISSUE_PROPS_SCATTERING[tissue_type]
            tissue_props[int_label, 2, i_wl] = TISSUE_PROPS_ANISOTROPY[tissue_type]
            tissue_props[int_label, 3, i_wl] = TISSUE_PROPS_REFRACTION[tissue_type]

    return tissue_props
