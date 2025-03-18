import pytest
import cedalion.datasets
import numpy as np
from cedalion.imagereco.sensitivity import parcel_sensitivity_mask


def test_parcel_sensitivity_mask():
    Adot = cedalion.datasets.get_ninjanirs_colin27_precomputed_sensitivity()

    Adot = Adot[:,Adot.is_brain,:]

    nchan, nvert, nwl = Adot.shape

    # create channel mask: select all
    channel_mask = np.ones(nchan * nwl, dtype=bool)

    # create random parcel labels
    rng = np.random.default_rng(seed=42)
    parcels = rng.choice(["A", "B", "C"], size=nvert)

    sensitivity_threshold = {"A": 0.01, "B": 0.01, "C": 0.01}

    mask = parcel_sensitivity_mask(Adot, parcels, channel_mask, sensitivity_threshold)
