import pytest
import cedalion_parcellation.datasets
import cedalion_parcellation.imagereco.forward_model as fw
from cedalion.sigproc.quality import parcel_sensitivity

def test_parcel_sensitivity():

    # load example dataset
    rec = cedalion_parcellation.datasets.get_fingertappingDOT()

    # load pathes to segmentation data for the icbm-152 atlas
    SEG_DATADIR, mask_files, landmarks_file = cedalion_parcellation.datasets.get_icbm152_segmentation()
    PARCEL_DIR = cedalion_parcellation.datasets.get_icbm152_parcel_file()

    # create forward model class for icbm152 atlas
    head = fw.TwoSurfaceHeadModel.from_surfaces(
        segmentation_dir=SEG_DATADIR,
        mask_files = mask_files,
        brain_surface_file= SEG_DATADIR+"mask_brain.obj",
        scalp_surface_file= SEG_DATADIR+"mask_scalp.obj",
        landmarks_ras_file=landmarks_file,
        parcel_file=PARCEL_DIR,
        brain_face_count=None,
        scalp_face_count=None
    )

    # snap probe to head and create forward model
    geo3D_snapped = head.align_and_snap_to_scalp(rec.geo3d)
    fwm = fw.ForwardModel(head, geo3D_snapped, rec._measurement_lists["amp"])

    # load precomputed fluence for dataset and headmodel
    fluence_all, fluence_at_optodes = cedalion_parcellation.datasets.get_precomputed_fluence("fingertappingDOT", "icbm152")

    # calculate Adot sensitivity matrix
    Adot = fwm.compute_sensitivity(fluence_all, fluence_at_optodes)

    # test parcel sensitivity function
    parcel_dOD, parcel_mask = parcel_sensitivity(Adot, None, 0.01, 1, 10, -3)

    sensitive_parcels = parcel_mask.where(parcel_mask, drop=True)["parcel"].values.tolist()
    assert len(sensitive_parcels) > 10, "something is wrong with the parcel sensitivity calculation"
