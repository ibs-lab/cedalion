import os
import tempfile

import numpy as np
import pytest
from scipy.sparse import find
import sys

import cedalion.datasets
import cedalion.imagereco.forward_model as fw

try:
    src_path = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../plugins/nirfaster-uFF",
        )
    )
    if src_path not in sys.path:
        sys.path.append(src_path)

    import nirfasteruff

    NIRFASTERUFF_NOT_AVAILABLE = False
except ImportError:
    NIRFASTERUFF_NOT_AVAILABLE = True


skip_if_nirfaster_unavailable = pytest.mark.skipif(
    NIRFASTERUFF_NOT_AVAILABLE, reason="nirfaster-uff not available"
)


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
        scalp_face_count=None,
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


@skip_if_nirfaster_unavailable
def test_run_nirfaster():
    """A minimal setup to run nirfaster."""

    volume = np.zeros((20, 20, 20), dtype=np.uint8)
    volume[1:-1, 1:-1, 1:-1] = 1

    src_pos = np.array([[1, 5, 10]])
    det_pos = np.array([[1, 15, 10]])

    solver = nirfasteruff.utils.get_solver()
    solver_opt = nirfasteruff.utils.SolverOptions()

    # meshing parameters; should be adjusted depending on the user's need
    meshingparam = nirfasteruff.utils.MeshingParams(
        facet_distance=1.0,
        facet_size=1.0,
        general_cell_size=2.0,
        lloyd_smooth=0,
    )

    # create a nirfaster mesh
    mesh = nirfasteruff.base.stndmesh()

    props = np.zeros((2, 4))
    # absorption, scattering, anisotropy, refraction
    props[0, :] = [0.0, 0.0, 1.0, 1.0]  # background
    props[1, :] = [0.02, 1.1, 0.001, 1.0]

    # make the optical property matrix; unit in mm-1
    tissueprop = np.zeros((1, 4))
    for i in range(tissueprop.shape[0]):
        tissueprop[i, 0] = i + 1
        tissueprop[i, 1] = props[i + 1, 0]
        tissueprop[i, 2] = props[i + 1, 1] * (1 - props[i + 1, 2])
        tissueprop[i, 3] = props[i + 1, 3]

    # all optodes x all optodes
    sources = nirfasteruff.base.optode(coord=src_pos)
    detectors = nirfasteruff.base.optode(coord=det_pos)
    n_optodes = src_pos.shape[0]
    link = np.zeros((n_optodes * n_optodes, 3), dtype=np.int32)
    ch = 0
    for i_s in range(len(src_pos)):
        for i_d in range(len(det_pos)):
            link[ch, 0] = i_s + 1
            link[ch, 1] = i_d + 1
            link[ch, 2] = 1
            ch += 1

    # construct the mesh
    mesh.from_volume(
        volume,
        param=meshingparam,
        prop=tissueprop,
        src=sources,
        det=detectors,
        link=link,
    )
    # calculate the interpolation functions to and from voxel space
    igrid = np.arange(volume.shape[0])
    jgrid = np.arange(volume.shape[1])
    kgrid = np.arange(volume.shape[2])
    mesh.gen_intmat(igrid, jgrid, kgrid)
    # calculate fluence
    data, _ = mesh.femdata(0, solver=solver, opt=solver_opt)
