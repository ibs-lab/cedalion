import numpy as np
import xarray as xr


def parcel_sensitivity_mask(
    Adot_brain: xr.DataArray,
    parcels: list[str],
    chan_mask: list[bool],
    sens_thresh: dict[str, float],
) -> dict[str, bool]:
    """Compute a mask of parcels based on absolute sensitivity thresholds.

    Args:
        Adot_brain: sensitivity matrix (channels, brain vertices, wavelengths)
        parcels: parcel label for each brain vertex
        chan_mask: boolean mask for channels (True if channel is useable,
                   False if channel has been pruned)
        sens_thresh: sensitivity threshold for each parcel

    Returns:
        dict: parcel mask {parcel: bool} (True if sensitivity is above threshold,
              False otherwise)
    """

    # check if number of channels match everywhere
    assert (
        len(chan_mask) == Adot_brain.shape[0] * Adot_brain.shape[2]
    )  # num_chans*wavelengths

    # check if number of brain voxels/vertices match
    assert len(parcels) == Adot_brain.shape[1]

    # flatten the wavelength dimension
    Adot_brain = np.vstack((Adot_brain[:, :, 0], Adot_brain[:, :, 1]))

    # apply the channel mask
    Adot_brain = Adot_brain[chan_mask, :]

    # get the parcel names
    parcels = np.asarray(parcels)
    parcel_names = np.unique(parcels)

    # for each parcel compute if the summed sensitivity is above the threshold
    parcel_mask = {
        roi: np.sum(Adot_brain[:, parcels == roi]) > sens_thresh[roi]
        for roi in parcel_names
    }

    return parcel_mask
