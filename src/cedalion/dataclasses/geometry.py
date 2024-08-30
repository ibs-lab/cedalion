from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering
from typing import Any

import numpy as np
import pint
import trimesh
import vtk
import mne
import xarray as xr
from scipy.spatial import KDTree
from vtk.util.numpy_support import vtk_to_numpy
import pyvista as pv

import cedalion
import cedalion.typing as cdt
from cedalion.vtktutils import trimesh_to_vtk_polydata, pyvista_polydata_to_trimesh


@total_ordering
class PointType(Enum):
    UNKNOWN = 0
    SOURCE = 1
    DETECTOR = 2
    LANDMARK = 3

    # provide an ordering of PointTypes so that e.g. np.unique works
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        else:
            return NotImplemented


@dataclass
class Voxels():
    voxels: np.ndarray
    crs: str
    units: pint.Unit
    
    @property
    def vertices(self) -> cdt.LabeledPointCloud:
        result = xr.DataArray(
            self.voxels,
            dims=["label", self.crs],
            coords={"label": np.arange(len(self.voxels))},
            attrs={"units": self.units},
        )
        result = result.pint.quantify()

        return result

    @property
    def nvertices(self) -> int:
        return len(self.voxels)
    
    def apply_transform(self, transform: cdt.AffineTransform) -> "Voxels":
        # convert to homogeneous coordinates
        num, dim = self.voxels.shape
        hom = np.ones((num,dim+1))
        hom[:,:3] = self.voxels
        # apply transformation
        hom = (transform.pint.dequantify().values.dot(hom.T)).T  
        # backtransformation
        transformed = np.array([hom[i,:3] / hom[i,3] for i in range(hom.shape[0])])

        new_units = self.units * transform.pint.units
        new_crs = transform.dims[0]

        return Voxels(transformed, new_crs, new_units)


    def _build_kdtree(self):
        self._kdtree = KDTree(self.voxels)

    def __post_init__(self):
        self._kdtree = None

    @property
    def kdtree(self):
        if self._kdtree is None:
            self._build_kdtree()

        return self._kdtree


@dataclass
class Surface(ABC):
    mesh: Any
    crs: str
    units: pint.Unit

    @property
    @abstractmethod
    def vertices(self) -> cdt.LabeledPointCloud:
        raise NotImplementedError()

    @property
    @abstractmethod
    def nvertices(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def nfaces(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def apply_transform(self, transform: cdt.AffineTransform):
        raise NotImplementedError()

    @abstractmethod
    def _build_kdtree(self):
        raise NotImplementedError()

    def __post_init__(self):
        self._kdtree = None

    @property
    def kdtree(self):
        if self._kdtree is None:
            self._build_kdtree()

        return self._kdtree

    def snap(self, points: cdt.LabeledPointCloud):
        if self.crs != points.points.crs:
            raise ValueError("CRS mismatch")

        if self.units != points.pint.units:
            raise ValueError("units mismatch")

        snapped = points.copy()

        points_dq = points.pint.dequantify()
        _, vertex_indices = self.kdtree.query(points_dq.values)

        matched_vertices = self.vertices[vertex_indices]
        matched_vertices = matched_vertices.assign_coords(label=points.label)

        snapped[:] = matched_vertices

        return snapped


@dataclass
class TrimeshSurface(Surface):
    mesh: trimesh.Trimesh

    @property
    def vertices(self) -> cdt.LabeledPointCloud:
        result = xr.DataArray(
            self.mesh.vertices,
            dims=["label", self.crs],
            coords={"label": np.arange(len(self.mesh.vertices))},
            attrs={"units": self.units},
        )
        result = result.pint.quantify()

        return result

    @property
    def nvertices(self) -> int:
        return len(self.mesh.vertices)

    @property
    def nfaces(self) -> int:
        return len(self.mesh.faces)

    def _build_kdtree(self):
        self._kdtree = KDTree(self.mesh.vertices)

    def apply_transform(self, transform: cdt.AffineTransform) -> "TrimeshSurface":
        transformed = self.mesh.copy()

        new_units = self.units * transform.pint.units
        new_crs = transform.dims[0]

        transformed.apply_transform(transform.pint.dequantify().values)

        return TrimeshSurface(transformed, new_crs, new_units)

    def decimate(self, face_count: int) -> "TrimeshSurface":
        """Use quadric decimation to reduce the number of vertices.

        Args:
            face_count: the number of faces of the decimated mesh

        Returns:
            The surface with a decimated mesh
        """

        vertices, faces = mne.decimate_surface(
            self.mesh.vertices, self.mesh.faces, face_count, method="quadric"
        )
        decimated = trimesh.Trimesh(vertices, faces)

        return TrimeshSurface(decimated, self.crs, self.units)

    def smooth(self, lamb: float) -> "TrimeshSurface":
        """Apply a Taubin filter to smooth this surface."""

        smoothed = trimesh.smoothing.filter_taubin(self.mesh, lamb=lamb)
        return TrimeshSurface(smoothed, self.crs, self.units)

    def get_vertex_normals(self, points: cdt.LabeledPointCloud):
        """Get normals of vertices closest to the provided points."""

        assert points.points.crs == self.crs
        assert points.pint.units == self.units
        points = points.pint.dequantify()

        _, vertex_indices = self.kdtree.query(points.values, workers=-1)

        return xr.DataArray(
            self.mesh.vertex_normals[vertex_indices],
            dims=["label", self.crs],
            coords={"label": points.label},
        )


    def fix_vertex_normals(self):
        mesh = self.mesh
        # again make sure, that normals face outside
        cog2vert = mesh.vertices - np.mean(mesh.vertices, axis=0)
        projected_normals = (cog2vert * mesh.vertex_normals).sum(axis=1)
        flip = np.where(projected_normals < 0, -1., 1. )[:,None]
        flipped_normals = mesh.vertex_normals * flip

        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_normals=flipped_normals)
        vertex_normals = mesh.vertex_normals.copy()
        # fix wrong normals (mostly due to unsmoothed surfaces) by taking the average of the neighbors
        false_normals = np.argwhere(np.linalg.norm(mesh.vertex_normals, axis=1).round() != 1.0)
        for idx in false_normals:
            neighbors = {v for face in mesh.faces if idx[0] in face for v in face}
            normal = np.mean([vertex_normals[n] for n in neighbors], axis=0)
            if np.linalg.norm(normal) > 0.99:
                vertex_normals[idx] = normal / np.linalg.norm(normal)
            else:
                print("%d neighbors" % len(neighbors))
                print("%d vertex index" % idx[0])
            

        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_normals=vertex_normals)
        return TrimeshSurface(mesh, self.crs, self.units)

    @classmethod
    def from_vtksurface(cls, vtk_surface: "VTKSurface"):
        vtk_polydata = vtk_surface.mesh
        pyvista_polydata = pv.wrap(vtk_polydata)
        mesh = pyvista_polydata_to_trimesh(pyvista_polydata)

        return cls(mesh=mesh, crs=vtk_surface.crs, units=vtk_surface.units)


@dataclass
class VTKSurface(Surface):
    mesh: vtk.vtkPolyData

    @property
    def vertices(self) -> cdt.LabeledPointCloud:
        vertices = vtk_to_numpy(self.mesh.GetPoints().GetData())
        result = xr.DataArray(
            vertices,
            dims=["label", self.crs],
            coords={"label": np.arange(len(vertices))},
            attrs={"units": self.units},
        )
        result = result.pint.quantify()

        return result

    @property
    def nvertices(self) -> int:
        return self.mesh.GetNumberOfPoints()

    @property
    def nfaces(self) -> int:
        return self.mesh.GetNumberOfPolys()

    def _build_kdtree(self):
        self._kdtree = KDTree(self.mesh.GetPoints().GetData())

    def apply_transform(self, transform: cdt.AffineTransform):
        raise NotImplementedError()

    @classmethod
    def from_trimeshsurface(cls, tri_mesh: TrimeshSurface):
        mesh = tri_mesh.mesh
        vtk_mesh = trimesh_to_vtk_polydata(mesh)

        return cls(mesh=vtk_mesh, crs=tri_mesh.crs, units=tri_mesh.units)

    def decimate(self, reduction: float, **kwargs) -> "VTKSurface":
        """Use VTK's decimate_pro method to reduce the number of vertices.

        Args:
            reduction: Reduction factor. A value of 0.9 will leave 10% of the original
                number of vertices.
            **kwargs: additional keyword arguments are passed to decimate_pro

        Returns:
            The surface with a decimated mesh
        """

        pyvista_polydata = pv.wrap(self.mesh)
        decimated = pyvista_polydata.decimate_pro(reduction, **kwargs)

        return VTKSurface(decimated, self.crs, self.units)


def affine_transform_from_numpy(
    transform: np.ndarray, from_crs: str, to_crs: str, from_units: str, to_units: str
) -> cdt.AffineTransform:
    units = cedalion.units.Unit(to_units) / cedalion.units.Unit(from_units)

    return xr.DataArray(transform, dims=[to_crs, from_crs]).pint.quantify(units)
