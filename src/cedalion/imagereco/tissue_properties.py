from enum import Enum, auto
from typing import Dict

import numpy as np
import xarray as xr


class TissueType(Enum):
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


def get_tissue_properties(segmentation_masks: xr.DataArray) -> np.ndarray:
    """Return tissue properties for the given segmentation mask."""
    ntissues = segmentation_masks.sizes["segmentation_type"] + 1
    tissue_props = np.zeros((ntissues, 4))
    tissue_props[0, :] = [0.0, 0.0, 1.0, 1.0]  # background

    for st in segmentation_masks.segmentation_type.values:
        m = segmentation_masks.sel(segmentation_type=st).values
        int_label = np.unique(m[m > 0]).item()

        if (tissue_type := TISSUE_LABELS.get(st, None)) is None:
            raise ValueError(f"unknown tissue type '{st}'")

        tissue_props[int_label, 0] = TISSUE_PROPS_ABSORPTION[tissue_type]
        tissue_props[int_label, 1] = TISSUE_PROPS_SCATTERING[tissue_type]
        tissue_props[int_label, 2] = TISSUE_PROPS_ANISOTROPY[tissue_type]
        tissue_props[int_label, 3] = TISSUE_PROPS_REFRACTION[tissue_type]

    return tissue_props
