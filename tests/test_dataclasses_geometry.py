import trimesh
from numpy.testing import assert_allclose
import pytest

import cedalion
import cedalion.dataclasses as cdc
import cedalion.xrutils as xrutils


def test_normal_normalization():
    # normal vectors have length 1
    sphere_orig = trimesh.creation.icosphere(4)

    # mesh with scaled normal vectors
    sphere = trimesh.Trimesh(
        vertices=sphere_orig.vertices,
        faces=sphere_orig.faces,
        vertex_normals=2 * sphere_orig.vertex_normals,
    )

    s = cdc.TrimeshSurface(sphere, crs="crs", units=cedalion.units.millimeter)

    norm1 = xrutils.norm(s.get_vertex_normals(s.vertices, normalized=False), s.crs)
    norm2 = xrutils.norm(s.get_vertex_normals(s.vertices, normalized=True), s.crs)
    norm3 = xrutils.norm(s.get_vertex_normals(s.vertices), s.crs)

    assert_allclose(norm1, 2)
    assert_allclose(norm2, 1)
    assert_allclose(norm3, 1)

    # set one normal to zero
    vertex_normals = 2 * sphere_orig.vertex_normals.copy()
    vertex_normals[5, :] = 0.0

    sphere = trimesh.Trimesh(
        vertices=sphere_orig.vertices,
        faces=sphere_orig.faces,
        vertex_normals=vertex_normals,
    )

    s = cdc.TrimeshSurface(sphere, crs="crs", units=cedalion.units.millimeter)

    norm1 = xrutils.norm(s.get_vertex_normals(s.vertices, normalized=False), s.crs)

    with pytest.raises(ValueError):
        norm2 = xrutils.norm(s.get_vertex_normals(s.vertices, normalized=True), s.crs)
