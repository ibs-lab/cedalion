import numpy as np
import pytest
import xarray as xr

import cedalion
import cedalion.dataclasses as cdc
import cedalion.geometry.segmentation as segm


def _open_sphere_mask(centre_k: float):
    """A sphere that is cut by the volume border, like a head with the neck cut off."""
    shape = (41, 41, 30)
    ii, jj, kk = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    centre = np.array([20.0, 20.0, centre_k])
    radius = 14.0
    mask = (
        np.sqrt((ii - centre[0]) ** 2 + (jj - centre[1]) ** 2 + (kk - centre[2]) ** 2)
        <= radius
    ).astype(np.uint8)
    da = xr.DataArray(
        mask[None],
        dims=["segmentation_type", "i", "j", "k"],
        coords={"segmentation_type": ["s"]},
    )
    return da, centre


def _outward_fraction(mesh, centre):
    face = (((mesh.triangles_center - centre) * mesh.face_normals).sum(1) > 0).mean()
    vertex = (((mesh.vertices - centre) * mesh.vertex_normals).sum(1) > 0).mean()
    return face, vertex


@pytest.mark.parametrize("centre_k", [15.0, 6.0])
def test_surface_from_segmentation_faces_point_outward(centre_k):
    """Faces of extracted surfaces point outward, also for open (non-watertight) meshes.

    Regression test: trimesh.fix_normals cannot invert non-watertight meshes, so
    surfaces of masks touching the volume border had inward winding. Vertex normals
    recomputed from the faces (after copy()/apply_transform()) then pointed into the
    head and the forward model launched photons away from it.
    """
    da, centre = _open_sphere_mask(centre_k)
    surface = segm.surface_from_segmentation(da, ["s"])

    face_outward, vertex_outward = _outward_fraction(surface.mesh, centre)
    assert face_outward > 0.95
    assert vertex_outward > 0.95

    # the orientation must survive a transformation (which recomputes the normals)
    t = cdc.affine_transform_from_numpy(
        np.diag([2.0, 2.0, 2.0, 1.0]), "ijk", "ras", "1", "mm"
    )
    transformed = surface.fix_vertex_normals().apply_transform(t)
    face_outward, vertex_outward = _outward_fraction(transformed.mesh, 2 * centre)
    assert face_outward > 0.95
    assert vertex_outward > 0.95


def test_fix_vertex_normals_inverts_winding():
    """fix_vertex_normals orients an inverted mesh by its faces, not by its cache."""
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    # remove a cap to make the mesh non-watertight and invert the winding
    keep = sphere.triangles_center[:, 2] < 8.0
    inverted = trimesh.Trimesh(sphere.vertices, sphere.faces[keep][:, ::-1])
    assert not inverted.is_watertight

    surface = cdc.TrimeshSurface(inverted, "ijk", cedalion.units.Unit("mm"))
    centre = np.zeros(3)
    assert _outward_fraction(surface.mesh, centre)[0] < 0.05

    fixed = surface.fix_vertex_normals()
    assert _outward_fraction(fixed.mesh, centre)[0] > 0.95
    assert _outward_fraction(fixed.copy().mesh, centre)[1] > 0.95
