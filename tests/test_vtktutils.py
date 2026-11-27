"""Tests for cedalion.vtktutils."""

import numpy as np
import pyvista as pv
import trimesh
from PIL import Image

from cedalion.vtktutils import trimesh_to_pv_textured_polydata


def _quad_trimesh_with_uv() -> trimesh.Trimesh:
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    img = Image.new("RGB", (4, 4), (255, 0, 0))
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=img)
    return mesh


def test_trimesh_to_pv_textured_polydata_uses_uvs():
    mesh = _quad_trimesh_with_uv()

    poly, tex = trimesh_to_pv_textured_polydata(mesh)

    # UV-mapped path: active texture coordinates are set, no per-vertex
    # RGB scalars.
    assert isinstance(poly, pv.PolyData)
    assert poly.active_texture_coordinates is not None
    assert poly.active_texture_coordinates.shape == (4, 2)
    assert poly.GetPointData().GetScalars() is None
    assert isinstance(tex, pv.Texture)


def test_trimesh_to_pv_textured_polydata_no_texture_returns_none():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    poly, tex = trimesh_to_pv_textured_polydata(mesh)

    assert isinstance(poly, pv.PolyData)
    assert tex is None
