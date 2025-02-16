"""Dataclasses for representing geometric objects."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import total_ordering
from typing import Any, List

import mne
import numpy as np
import pint
import pyvista as pv
import trimesh
import vtk
import xarray as xr
from scipy import sparse
from scipy.spatial import KDTree
from vtk.util.numpy_support import vtk_to_numpy

import cedalion
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion.errors import CRSMismatchError
from cedalion.vtktutils import pyvista_polydata_to_trimesh, trimesh_to_vtk_polydata


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
class Surface(ABC):
    """Abstract base class for surfaces."""

    mesh: Any
    """The mesh representing the surface."""

    crs: str
    """The coordinate reference system of the surface."""

    units: pint.Unit
    """The units of the surface."""

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
        """Snap points to the nearest vertices on the surface."""
        if self.crs != points.points.crs:
            raise CRSMismatchError.unexpected_crs(
                expected_crs=self.crs, found_crs=points.points.crs
            )

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
class Voxels():
    """3D voxels represented by a np.array."""

    voxels: np.ndarray
    """The voxels."""

    crs: str
    """The coordinate reference system of the voxels."""

    units: pint.Unit
    """The units of the voxels."""

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
class TrimeshSurface(Surface):
    """A surface represented by a trimesh object."""

    mesh: trimesh.Trimesh
    """The trimesh object representing the surface."""

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
        """Apply an affine transformation to this surface.

        Args:
            transform: The affine transformation to apply.

        Returns:
            TrimeshSurface: The transformed surface.
        """
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

    def get_vertex_normals(
        self,
        points: cdt.LabeledPointCloud,
        normalized: bool = True
    ):
        """Get normals of vertices closest to the provided points."""

        assert points.points.crs == self.crs
        assert points.pint.units == self.units
        points = points.pint.dequantify()

        _, vertex_indices = self.kdtree.query(points.values, workers=-1)

        normals = xr.DataArray(
            self.mesh.vertex_normals[vertex_indices],
            dims=["label", self.crs],
            coords={"label": points.label},
        )

        if normalized:
            norms = xrutils.norm(normals, dim=normals.points.crs)

            if not (norms > 0).all():
                raise ValueError("Cannot normalize normals with zero length.")

            normals /= norms

        return normals

    def fix_vertex_normals(self):
        mesh = self.mesh
        # again make sure, that normals face outside
        cog2vert = mesh.vertices - np.mean(mesh.vertices, axis=0)
        projected_normals = (cog2vert * mesh.vertex_normals).sum(axis=1)
        flip = np.where(projected_normals < 0, -1.0, 1.0)[:, None]
        flipped_normals = mesh.vertex_normals * flip

        mesh = trimesh.Trimesh(
            mesh.vertices, mesh.faces, vertex_normals=flipped_normals
        )
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


@dataclass
class SimpleMesh:
    pts: np.ndarray
    polys: np.ndarray


@dataclass
class PycortexSurface(Surface):
    """Provides the geodesic functionality of Pycortex.

    References:
        Functions in this class are based on the implementation
        in Pycortex (:cite:t:`Gao2015`).
        Gao JS, Huth AG, Lescroart MD and Gallant JL (2015)
        Pycortex: an interactive surface visualizer for fMRI.
        Front. Neuroinform. 9:23. doi: 10.3389/fninf.2015.00023
    """

    mesh: SimpleMesh

    def __init__(self, mesh: SimpleMesh, crs: str, units: pint.Unit):
        super().__init__(mesh, crs, units)
        self._cache = dict()
        self._rlfac_solvers = dict()
        self._nLC_solvers = dict()

    @property
    def vertices(self) -> cdt.LabeledPointCloud:
        result = xr.DataArray(
            self.mesh.pts,
            dims=["label", self.crs],
            coords={"label": np.arange(len(self.mesh.pts))},
            attrs={"units": self.units},
        )
        result = result.pint.quantify()

        return result

    @property
    def nvertices(self) -> int:
        return len(self.mesh.pts)

    @property
    def nfaces(self) -> int:
        return len(self.mesh.polys)

    def _build_kdtree(self):
        self._kdtree = KDTree(self.mesh.pts)

    def apply_transform(self, transform: cdt.AffineTransform) -> "PycortexSurface":
        transformed_pts = self.mesh.pts.copy()
        new_units = self.units * transform.pint.units
        new_crs = transform.dims[0]

        transformed_pts = transform.apply(transformed_pts)

        transformed_mesh = SimpleMesh(transformed_pts, self.mesh.polys)

        return PycortexSurface(mesh=transformed_mesh, crs=new_crs, units=new_units)

    def decimate(self, face_count: int) -> "PycortexSurface":
        raise NotImplementedError("Decimation not implemented for PycortexSurface")

    def get_vertex_normals(
        self,
        points: cdt.LabeledPointCloud,
        normalized: bool = True
    ):
        assert points.points.crs == self.crs
        assert points.pint.units == self.units
        points = points.pint.dequantify()

        _, vertex_indices = self.kdtree.query(points.values, workers=-1)

        # Calculate vertex normals
        face_normals = np.cross(
            self.mesh.pts[self.mesh.polys[:, 1]] - self.mesh.pts[self.mesh.polys[:, 0]],
            self.mesh.pts[self.mesh.polys[:, 2]] - self.mesh.pts[self.mesh.polys[:, 0]],
        )
        face_normals /= np.linalg.norm(face_normals, axis=1)[:, np.newaxis]
        vertex_normals = np.zeros_like(self.mesh.pts)
        for i, poly in enumerate(self.mesh.polys):
            for j in poly:
                vertex_normals[j] += face_normals[i]

        if normalized:
            vertex_normals /= np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis]

        return xr.DataArray(
            vertex_normals[vertex_indices],
            dims=["label", self.crs],
            coords={"label": points.label},
        )

    @property
    def ppts(self) -> np.ndarray:
        """3D matrix of points in each face.

        n faces x 3  per face x 3 coords per point.
        """
        return self.mesh.pts[self.mesh.polys]

    @property
    def connected(self) -> sparse.csr_matrix:
        """Sparse matrix of vertex-face associations."""
        npt = len(self.mesh.pts)
        npoly = len(self.mesh.polys)
        return sparse.coo_matrix(
            (
                np.ones((3 * npoly,)),  # data
                (
                    np.hstack(self.mesh.polys.T),  # row
                    np.tile(range(npoly), (1, 3)).squeeze(),
                ),
            ),  # col
            (npt, npoly),
        ).tocsr()  # size

    @property
    def adj(self) -> sparse.csr_matrix:
        """Sparse vertex adjacency matrix."""
        npt = len(self.mesh.pts)
        npoly = len(self.mesh.polys)
        adj1 = sparse.coo_matrix(
            (np.ones((npoly,)), (self.mesh.polys[:, 0], self.mesh.polys[:, 1])),
            (npt, npt),
        )
        adj2 = sparse.coo_matrix(
            (np.ones((npoly,)), (self.mesh.polys[:, 0], self.mesh.polys[:, 2])),
            (npt, npt),
        )
        adj3 = sparse.coo_matrix(
            (np.ones((npoly,)), (self.mesh.polys[:, 1], self.mesh.polys[:, 2])),
            (npt, npt),
        )
        alladj = (adj1 + adj2 + adj3).tocsr()
        return alladj + alladj.T

    @property
    def face_normals(self) -> np.ndarray:
        """Normal vector for each face."""
        # Compute normal vector direction
        nnfnorms = np.cross(
            self.ppts[:, 1] - self.ppts[:, 0], self.ppts[:, 2] - self.ppts[:, 0]
        )
        # Normalize to norm 1
        nfnorms = nnfnorms / np.sqrt((nnfnorms**2).sum(1))[:, np.newaxis]
        # Ensure that there are no nans. Shouldn't be a problem with well-formed srfcs.
        return np.nan_to_num(nfnorms)

    @property
    def vertex_normals(self) -> np.ndarray:
        """Normal vector for each vertex (average of normals for neighboring faces)."""
        # Average adjacent face normals
        nnvnorms = np.nan_to_num(
            self.connected.dot(self.face_normals) / self.connected.sum(1)
        ).A
        # Normalize to norm 1
        return nnvnorms / np.sqrt((nnvnorms**2).sum(1))[:, np.newaxis]

    @property
    def face_areas(self) -> np.ndarray:
        """Area of each face."""
        # Compute normal vector (length is face area)
        nnfnorms = np.cross(
            self.ppts[:, 1] - self.ppts[:, 0], self.ppts[:, 2] - self.ppts[:, 0]
        )
        # Compute vector length
        return np.sqrt((nnfnorms**2).sum(-1)) / 2

    @property
    def cotangent_weights(self) -> np.ndarray:
        """Cotangent of angle opposite each vertex in each face."""
        ppts = self.ppts
        cots1 = ((ppts[:, 1] - ppts[:, 0]) * (ppts[:, 2] - ppts[:, 0])).sum(
            1
        ) / np.sqrt(
            (np.cross(ppts[:, 1] - ppts[:, 0], ppts[:, 2] - ppts[:, 0]) ** 2).sum(1)
        )
        cots2 = ((ppts[:, 2] - ppts[:, 1]) * (ppts[:, 0] - ppts[:, 1])).sum(
            1
        ) / np.sqrt(
            (np.cross(ppts[:, 2] - ppts[:, 1], ppts[:, 0] - ppts[:, 1]) ** 2).sum(1)
        )
        cots3 = ((ppts[:, 0] - ppts[:, 2]) * (ppts[:, 1] - ppts[:, 2])).sum(
            1
        ) / np.sqrt(
            (np.cross(ppts[:, 0] - ppts[:, 2], ppts[:, 1] - ppts[:, 2]) ** 2).sum(1)
        )

        # Then we have to sanitize everything..
        cots = np.vstack([cots1, cots2, cots3])
        cots[np.isinf(cots)] = 0
        cots[np.isnan(cots)] = 0
        return cots

    @property
    def laplace_operator(self):
        """Laplace-Beltrami operator for this surface.

        A sparse adjacency matrix with
        edge weights determined by the cotangents of the angles opposite each edge.
        Returns a 4-tuple (B, D, W, V) where D is the 'lumped mass matrix', W is the
        weighted adjacency matrix, and V is a diagonal matrix that normalizes the
        adjacencies. The 'stiffness matrix', A, can be computed as V - W.

        The full LB operator can be computed as D^{-1} (V - W).

        B is the finite element method (FEM) 'mass matrix', which replaces D in FEM
        analyses.

        See 'Discrete Laplace-Beltrami operators for shape analysis and segmentation'
        by Reuter et al., 2009 for details.
        """
        # Lumped mass matrix
        D = self.connected.dot(self.face_areas) / 3.0

        # Stiffness matrix
        npt = len(self.mesh.pts)
        cots1, cots2, cots3 = self.cotangent_weights
        # W is weighted adjacency matrix
        W1 = sparse.coo_matrix(
            (cots1, (self.mesh.polys[:, 1], self.mesh.polys[:, 2])), (npt, npt)
        )
        W2 = sparse.coo_matrix(
            (cots2, (self.mesh.polys[:, 2], self.mesh.polys[:, 0])), (npt, npt)
        )
        W3 = sparse.coo_matrix(
            (cots3, (self.mesh.polys[:, 0], self.mesh.polys[:, 1])), (npt, npt)
        )
        W = (W1 + W1.T + W2 + W2.T + W3 + W3.T).tocsr() / 2.0

        # V is sum of each col
        V = sparse.dia_matrix((np.array(W.sum(0)).ravel(), [0]), (npt, npt))

        # A is stiffness matrix
        # A = W - V # negative operator -- more useful in practice

        # For FEM:
        Be1 = sparse.coo_matrix(
            (self.face_areas, (self.mesh.polys[:, 1], self.mesh.polys[:, 2])),
            (npt, npt),
        )
        Be2 = sparse.coo_matrix(
            (self.face_areas, (self.mesh.polys[:, 2], self.mesh.polys[:, 0])),
            (npt, npt),
        )
        Be3 = sparse.coo_matrix(
            (self.face_areas, (self.mesh.polys[:, 0], self.mesh.polys[:, 1])),
            (npt, npt),
        )
        Bd = self.connected.dot(self.face_areas) / 6
        dBd = sparse.dia_matrix((Bd, [0]), (len(D), len(D)))
        B = (Be1 + Be1.T + Be2 + Be2.T + Be3 + Be3.T) / 12 + dBd
        return B, D, W, V


    @property
    def avg_edge_length(self):
        """Average length of all edges in the surface."""
        adj = self.adj
        tadj = sparse.triu(adj, 1)  # only entries above main diagonal, in coo format
        edgelens = np.sqrt(
            ((self.mesh.pts[tadj.row] - self.mesh.pts[tadj.col]) ** 2).sum(1)
        )
        return edgelens.mean()


    def surface_gradient(
        self,
        scalars: np.ndarray,
        at_verts: bool = True
    ) -> np.ndarray:
        """Gradient of a function with values `scalars` at each vertex on the surface.

        If `at_verts`, returns values at each vertex. Otherwise, returns values at each
        face.

        Args:
            scalars : 1D ndarray, shape (total_verts,) a scalar-valued function across
                the cortex.
            at_verts : If True (default), values will be returned for each vertex.
                Otherwise, values will be returned for each face.

        Returns:
            gradu : 2D ndarray, shape (total_verts,3) or (total_polys,3)
                Contains the x-, y-, and z-axis gradients of the given `scalars` at
                either each vertex (if `at_verts` is True) or each face.
        """
        pu = scalars[self.mesh.polys]
        fe12, fe23, fe31 = [f.T for f in self._facenorm_cross_edge]
        pu1, pu2, pu3 = pu.T
        fa = self.face_areas

        gradu = np.nan_to_num(((fe12 * pu3 + fe23 * pu1 + fe31 * pu2) / (2 * fa)).T)


        if at_verts:
            return (self.connected.dot(gradu).T / self.connected.sum(1).A.squeeze()).T
        return gradu


    @property
    def _facenorm_cross_edge(self):
        ppts = self.ppts
        fnorms = self.face_normals
        fe12 = np.cross(fnorms, ppts[:, 1] - ppts[:, 0])
        fe23 = np.cross(fnorms, ppts[:, 2] - ppts[:, 1])
        fe31 = np.cross(fnorms, ppts[:, 0] - ppts[:, 2])
        return fe12, fe23, fe31


    def geodesic_distance(self, verts, m: float = 1.0, fem: bool = False) -> np.ndarray:
        """Calcualte the inimum mesh geodesic distance (in mm).

        The geodesic distance is calculated from each vertex in surface to any vertex in
        the collection `verts`.

        Geodesic distance is estimated using heat-based method (see 'Geodesics in Heat',
        Crane et al, 2012). Diffusion of heat along the mesh is simulated and then
        used to infer geodesic distance. The duration of the simulation is controlled
        by the parameter `m`. Larger values of `m` will smooth & regularize the distance
        computation. Smaller values of `m` will roughen and will usually increase error
        in the distance computation. The default value of 1.0 is probably pretty good.

        This function caches some data (sparse LU factorizations of the laplace-beltrami
        operator and the weighted adjacency matrix), so it will be much faster on
        subsequent runs.

        The time taken by this function is independent of the number of vertices in
        verts.

        Args:
            verts : 1D array-like of ints
                Set of vertices to compute distance from. This function returns the
                shortest distance to any of these vertices from every vertex in the
                surface.
            m : float, optional
                Reverse Euler step length. The optimal value is likely between 0.5 and
                1.5. Default is 1.0, which should be fine for most cases.
            fem : bool, optional
                Whether to use Finite Element Method lumped mass matrix. Wasn't used in
                Crane 2012 paper. Doesn't seem to help any.

        Returns:
            1D ndarray, shape (total_verts,)
            Geodesic distance (in mm) from each vertex in the surface to the closest
            vertex in `verts`.
        """
        npt = len(self.mesh.pts)
        if m not in self._rlfac_solvers or m not in self._nLC_solvers:
            B, D, W, V = self.laplace_operator
            nLC = W - V  # negative laplace matrix
            if not fem:
                # lumped mass matrix
                spD = sparse.dia_matrix((D, [0]), (npt, npt)).tocsr()
            else:
                spD = B

            t = m * self.avg_edge_length**2  # time of heat evolution
            lfac = spD - t * nLC  # backward Euler matrix

            # Exclude rows with zero weight (these break the sparse LU)
            goodrows = np.nonzero(~np.array(lfac.sum(0) == 0).ravel())[0]
            self._goodrows = goodrows
            self._rlfac_solvers[m] = sparse.linalg.factorized(
                lfac[goodrows][:, goodrows]
            )
            self._nLC_solvers[m] = sparse.linalg.factorized(
                nLC[goodrows][:, goodrows]
            )

        # I. "Integrate the heat flow ̇u = ∆u for some fixed time t"
        # ---------------------------------------------------------

        # Solve system to get u, the heat values
        u0 = np.zeros((npt,))  # initial heat values
        u0[verts] = 1.0
        goodu = self._rlfac_solvers[m](u0[self._goodrows])
        u = np.zeros((npt,))
        u[self._goodrows] = goodu

        # II. "Evaluate the vector field X = − ∇u / |∇u|"
        # -----------------------------------------------

        # Compute grad u at each face
        gradu = self.surface_gradient(u, at_verts=False)

        # Compute X (normalized grad u)
        gusum = np.sum(gradu ** 2, axis=1)
        X = np.nan_to_num((-gradu.T / np.sqrt(gusum)).T)

        # III. "Solve the Poisson equation ∆φ = ∇·X"
        # ------------------------------------------

        # Compute integrated divergence of X at each vertex
        c32, c13, c21 = self._cot_edge
        x1 = 0.5 * (c32 * X).sum(1)
        x2 = 0.5 * (c13 * X).sum(1)
        x3 = 0.5 * (c21 * X).sum(1)

        conn1, conn2, conn3 = self._polyconn
        divx = conn1.dot(x1) + conn2.dot(x2) + conn3.dot(x3)

        # Compute phi (distance)
        goodphi = self._nLC_solvers[m](divx[self._goodrows])
        phi = np.zeros((npt,))
        phi[self._goodrows] = goodphi - goodphi.min()

        # Ensure that distance is zero for selected verts
        phi[verts] = 0.0

        return phi


    def geodesic_path(
        self, a: int,
        b: int,
        max_len: int = 1000,
        d: np.ndarray = None,
        **kwargs
    ) -> List:
        """Finds the shortest path between two points `a` and `b`.

        This shortest path is based on geodesic distances across the surface.
        The path starts at point `a` and selects the neighbor of `a` in the
        graph that is closest to `b`. This is done iteratively with the last
        vertex in the path until the last point in the path is `b`.

        Other Parameters in kwargs are passed to the geodesic_distance
        function to alter how geodesic distances are actually measured

        Args:
            a : int
                Vertex that is the start of the path
            b : int
                Vertex that is the end of the path
            d : array
                array of geodesic distances, will be computed if not provided

            max_len : int, optional, default=1000
                Maximum path length before the function quits. Sometimes it can get
                stuck in loops, causing infinite paths.
            m : float, optional
                Reverse Euler step length. The optimal value is likely between 0.5 and
                1.5. Default is 1.0, which should be fine for most cases.
            fem : bool, optional
                Whether to use Finite Element Method lumped mass matrix. Wasn't used in
                Crane 2012 paper. Doesn't seem to help any.

            kwargs: other arugments are passed to self.geodesic_distance

        Returns:
            path : list
                List of the vertices in the path from a to b
        """
        path = [a]
        if d is None:
            d = self.geodesic_distance([b], **kwargs)
        while path[-1] != b:
            n = np.array([v for v in self.graph.neighbors(path[-1])])
            path.append(n[d[n].argmin()])
            if len(path) > max_len:
                return path
        return path

    @property
    def _cot_edge(self):
        ppts = self.ppts
        cots1, cots2, cots3 = self.cotangent_weights
        c3 = cots3[:,np.newaxis] * (ppts[:,1] - ppts[:,0])
        c2 = cots2[:,np.newaxis] * (ppts[:,0] - ppts[:,2])
        c1 = cots1[:,np.newaxis] * (ppts[:,2] - ppts[:,1])
        c32 = c3 - c2
        c13 = c1 - c3
        c21 = c2 - c1
        return c32, c13, c21

    @property
    def _polyconn(self):
        npt = len(self.mesh.pts)
        npoly = len(self.mesh.polys)
        o = np.ones((npoly,))

        c1 = sparse.coo_matrix((o, (self.mesh.polys[:,0], range(npoly))), (npt, npoly)).tocsr() # noqa: E501
        c2 = sparse.coo_matrix((o, (self.mesh.polys[:,1], range(npoly))), (npt, npoly)).tocsr() # noqa: E501
        c3 = sparse.coo_matrix((o, (self.mesh.polys[:,2], range(npoly))), (npt, npoly)).tocsr() # noqa: E501

        return c1, c2, c3

    @classmethod
    def from_trimeshsurface(cls, tri_mesh: TrimeshSurface):
        pts = tri_mesh.mesh.vertices
        polys = tri_mesh.mesh.faces

        pycortex_mesh = SimpleMesh(pts, polys)

        return cls(mesh=pycortex_mesh, crs=tri_mesh.crs, units=tri_mesh.units)

    @classmethod
    def from_vtksurface(cls, vtk_surface: VTKSurface):
        pts = vtk_to_numpy(vtk_surface.mesh.GetPoints().GetData())
        polys = vtk_to_numpy(vtk_surface.mesh.GetPolys().GetData()).reshape(-1, 4)[
            :, 1:
        ]

        pycortex_mesh = SimpleMesh(pts, polys)

        return cls(mesh=pycortex_mesh, crs=vtk_surface.crs, units=vtk_surface.units)


def affine_transform_from_numpy(
    transform: np.ndarray, from_crs: str, to_crs: str, from_units: str, to_units: str
) -> cdt.AffineTransform:
    """Create a AffineTransform object from a numpy array."""
    units = cedalion.units.Unit(to_units) / cedalion.units.Unit(from_units)

    return xr.DataArray(transform, dims=[to_crs, from_crs]).pint.quantify(units)
