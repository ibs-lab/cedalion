import os
import tempfile

import numpy as np
from scipy.sparse import find

import cedalion.datasets
import cedalion.imagereco.forward_model as fw


def allclose(A, B, atol=1e-8):
    """Check if two sparse matrices are equal within a tolerance."""
    # If you want to check matrix shapes as well
    if np.array_equal(A.shape, B.shape) == 0:
        return False
    r1, c1, v1 = find(A)
    r2, c2, v2 = find(B)
    index_match = np.array_equal(r1, r2) & np.array_equal(c1, c2)
    if index_match == 0:
        return False
    return np.allclose(v1, v2, atol=atol)


def test_TwoSurfaceHeadModel():
    ### tests only save and load methods so far
    # prepare test head
    (
        SEG_DATADIR,
        mask_files,
        landmarks_file,
    ) = cedalion.datasets.get_colin27_segmentation(downsampled=True)
    head = fw.TwoSurfaceHeadModel.from_segmentation(
        segmentation_dir=SEG_DATADIR,
        mask_files=mask_files,
        landmarks_ras_file=landmarks_file,
        # disable mesh smoothing and decimation to speed up runtime
        smoothing=0,
        brain_face_count=None,
        scalp_face_count=None
    )
    # save to folder

    with tempfile.TemporaryDirectory() as dirpath:
        tmp_folder = os.path.join(dirpath, "test_head")
        head.save(tmp_folder)
        # load from folder
        head2 = fw.TwoSurfaceHeadModel.load(tmp_folder)
        # compare
        assert (head.landmarks == head2.landmarks).all()
        assert (head.segmentation_masks == head2.segmentation_masks).all()
        assert (head.brain.mesh.vertices == head2.brain.mesh.vertices).all()
        assert (head.brain.mesh.faces == head2.brain.mesh.faces).all()
        assert (head.t_ijk2ras.values == head2.t_ijk2ras.values).all()
        assert (head.t_ras2ijk.values == head2.t_ras2ijk.values).all()
        assert allclose(head.voxel_to_vertex_brain, head2.voxel_to_vertex_brain)
        assert allclose(head.voxel_to_vertex_scalp, head2.voxel_to_vertex_scalp)
