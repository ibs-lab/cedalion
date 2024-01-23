import vtk
from vtk.util.numpy_support import numpy_to_vtkIdTypeArray, numpy_to_vtk
import trimesh
import numpy as np


def trimesh_to_vtk_polydata(mesh: trimesh.Trimesh):
    ntris, ndim_cells = mesh.faces.shape
    nvertices, ndim_vertices = mesh.vertices.shape

    assert ndim_cells == 3  # operate only on triangle meshes
    assert ndim_vertices == 3  # operate only in 3D space

    # figure out if vtk uses 32 or 64 bits for IDs
    id_size = vtk.vtkIdTypeArray().GetDataTypeSize()
    id_dtype = np.int32 if id_size == 4 else np.int64

    cell_npoints = np.full(ntris, ndim_cells)
    # triangle definition: (points per cell, *point ids)
    point_ids = np.hstack((cell_npoints[:, None], mesh.faces)).astype(id_dtype).ravel()
    point_ids = numpy_to_vtkIdTypeArray(point_ids, deep=1)

    cells = vtk.vtkCellArray()
    cells.SetCells(ntris, point_ids)

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(mesh.vertices, deep=1))

    vtk_mesh = vtk.vtkPolyData()
    vtk_mesh.SetPoints(points)
    vtk_mesh.SetPolys(cells)

    # if the trimesh was textured copy the color information, too
    if isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        colors = mesh.visual.vertex_colors
    else:
        colors = mesh.visual.to_color().vertex_colors

    colors = numpy_to_vtk(colors)
    colors.SetName("colors")
    vtk_mesh.GetPointData().SetScalars(colors)

    return vtk_mesh
