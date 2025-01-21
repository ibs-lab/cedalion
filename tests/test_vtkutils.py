import cedalion
import cedalion.dataclasses as cdc
import trimesh
import numpy as np

def test_conversion():
    vertices = np.asarray([
        [0.,0.,0.],
        [1., 0. ,0.],
        [1., 1., 0.],
        [0., 1., 0.]
    ])
    faces = np.asarray([
        [0., 1., 2.],
        [1.,2.,3.]
    ])

    trimesh_surface = cdc.TrimeshSurface(
        trimesh.Trimesh(vertices, faces),
        crs = "my_crs",
        units = cedalion.units.mm
    )

    vtk_surface = cdc.VTKSurface.from_trimeshsurface(trimesh_surface)

    assert trimesh_surface.nfaces == vtk_surface.nfaces
    assert trimesh_surface.nvertices == vtk_surface.nvertices
    assert trimesh_surface.crs == vtk_surface.crs
    assert trimesh_surface.units == vtk_surface.units

    trimesh_surface2 = cdc.TrimeshSurface.from_vtksurface(vtk_surface)

    assert np.all(trimesh_surface2.mesh.vertices == vertices)
    assert np.all(trimesh_surface2.mesh.faces == faces)
