import warnings
from typing import List, Optional

import numpy as np
import vtk
import vtk.util.numpy_support as vnp
import xarray as xr
import pyvista as pv

import cedalion.plots
from cedalion.dataclasses import (
    Surface,
    TrimeshSurface,
    VTKSurface,
    validate_schemas,
    PointType,
)
from cedalion.typing import LabeledPointCloud


def _sort_line_points(start_point: np.ndarray, points: np.ndarray):
    sorted_indices = []
    sorted_distances = []
    indices = list(range(len(points)))

    current_point = start_point
    for i in range(len(points)):
        dists = np.linalg.norm(points[indices] - current_point, axis=1)
        assert len(dists) == len(indices)
        closest_point_dists_index = np.argmin(dists)
        closest_point_points_index = indices.pop(closest_point_dists_index)
        sorted_indices.append(closest_point_points_index)
        sorted_distances.append(dists[closest_point_dists_index])
        current_point = points[closest_point_points_index]

    assert len(set(sorted_indices)) == len(points)

    cumulative_distance = np.cumsum(sorted_distances)
    cumulative_distance /= cumulative_distance[-1]

    return points[sorted_indices], cumulative_distance


def _intersect_mesh_with_triangle(
    vtk_mesh: vtk.vtkPolyData,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    select: Optional[List[float]] = None,
):
    """Find the line along the mesh through three points.

    Construct a line on a surface meshe from p0 through p1 to p2. To that end the
    mesh is intersected with a plane defined by the triangle (p0,p1,p2). Only points
    above the p0-p2-line are kept.
    """
    p0p2 = p2 - p0
    origin = p0 + 0.5 * p0p2

    normal_cut = np.cross(p0p2, p1 - p0)
    normal_clip = np.cross(normal_cut, p0p2)

    plane_cut = vtk.vtkPlane()
    plane_cut.SetOrigin(*origin)
    plane_cut.SetNormal(*normal_cut)

    plane_clip = vtk.vtkPlane()
    plane_clip.SetOrigin(*origin)
    plane_clip.SetNormal(*normal_clip)

    # cut through the mesh to get a closed line
    cut = vtk.vtkCutter()
    cut.SetCutFunction(plane_cut)
    cut.SetInputData(vtk_mesh)
    cut.Update()

    # clip the line to the area above p0-p2
    clip = vtk.vtkClipPolyData()
    clip.SetInputData(cut.GetOutput())
    clip.SetClipFunction(plane_clip)
    clip.Update()

    strip = vtk.vtkStripper()
    strip.SetInputData(clip.GetOutput())
    strip.Update()

    # fit a spline through the points and get equidistant points
    spline = vtk.vtkSplineFilter()
    spline.SetInputData(strip.GetOutput())
    spline.SetSubdivideToLength()
    spline.SetLength(0.5)  # FIXME assume mm here
    spline.Update()

    result = spline.GetOutput()
    points = vnp.vtk_to_numpy(result.GetPoints().GetData())

    # sort points in increasing distance from p0
    points, dists = _sort_line_points(p0, points)

    # find selected point indices along line
    if select is not None:
        indices = [np.searchsorted(dists, sel) for sel in select]
    else:
        indices = None

    points = np.copy(points)

    return points, dists, indices


class LandmarksBuilder1010:
    """Construct the 10-10-system on scalp surface based on :cite:t:`Oostenveld2001`.

    Args:
        scalp_surface: a triangle-mesh representing the scalp
        landmarks: positions of "Nz", "Iz", "LPA", "RPA"

    """

    @validate_schemas
    def __init__(self, scalp_surface: Surface, landmarks: LabeledPointCloud):
        if isinstance(scalp_surface, TrimeshSurface):
            scalp_surface = VTKSurface.from_trimeshsurface(scalp_surface)

        self.scalp_surface = scalp_surface

        required_landmarks = ["Nz", "Iz", "LPA", "RPA"]
        for label in required_landmarks:
            assert label in landmarks.label

        self.landmarks_mm = landmarks.pint.to("mm").pint.dequantify()

        self.vtk_mesh = self.scalp_surface.mesh

        self.lines = []

    def _estimate_cranial_vertex_by_height(self):
        """Find the highest point of the skull."""

        vertices = vnp.vtk_to_numpy(self.vtk_mesh.GetPoints().GetData())
        highest_vertices = vertices[vertices[:, 2] == vertices[:, 2].max()]

        # there may be serveral vertices with the same z-coordinate -> average
        return highest_vertices.mean(axis=0)

    def _estimate_cranial_vertex_from_lines(self):
        if "Cz" in self.landmarks_mm.label:
            cz1 = self.landmarks_mm.loc["Cz"].values
            # FIXME remove Cz from landmarks
        else:
            cz1 = self._estimate_cranial_vertex_by_height()

        points_a, dists_a, indices = _intersect_mesh_with_triangle(
            self.vtk_mesh,
            self.landmarks_mm.loc["Nz"].values,
            cz1,
            self.landmarks_mm.loc["Iz"].values,
            select=[0.5],
        )

        cz2 = points_a[indices[0]]

        points_r, dists_r, _ = _intersect_mesh_with_triangle(
            self.vtk_mesh,
            self.landmarks_mm.loc["LPA"].values,
            cz2,
            self.landmarks_mm.loc["RPA"].values,
        )

        candidates_r = points_r[(0.4 <= dists_r) & (dists_r <= 0.6)]
        candidates_a = points_a[(0.4 <= dists_a) & (dists_a <= 0.6)]

        # find smallest distance between candidate points
        dists = np.linalg.norm(
            (candidates_r[:, None, :] - candidates_a[None, :, :]), axis=2
        )
        idx_r, idx_a = np.unravel_index(np.argmin(dists), dists.shape)

        # average between the two closest points
        cz2 = np.vstack((candidates_r[idx_r], candidates_a[idx_a])).mean(axis=0)

        return cz2

    def _add_landmarks_along_line(
        self, triangle_labels: List[str], labels: List[str], dists: List[float]
    ):
        assert len(triangle_labels) == 3
        assert len(labels) == len(dists)
        assert all([label in self.landmarks_mm.label for label in triangle_labels])

        points, _, indices = _intersect_mesh_with_triangle(
            self.vtk_mesh,
            self.landmarks_mm.loc[triangle_labels[0]].values,
            self.landmarks_mm.loc[triangle_labels[1]].values,
            self.landmarks_mm.loc[triangle_labels[2]].values,
            select=dists,
        )

        crs = self.landmarks_mm.points.crs

        tmp = xr.DataArray(
            points[indices],
            dims=["label", crs],
            coords={
                "label": ("label", labels),
                "type": ("label", [PointType.LANDMARK] * len(labels)),
            },
            attrs={"units": "mm"},
        )

        self.landmarks_mm = xr.concat((self.landmarks_mm, tmp), dim="label")

        self.lines.append(points)

    def build(self):
        warnings.warn("WIP: distance calculation around ears")

        cz = self._estimate_cranial_vertex_from_lines()

        self.landmarks_mm = self.landmarks_mm.points.add("Cz", cz, PointType.LANDMARK)

        self._add_landmarks_along_line(
            ["Nz", "Cz", "Iz"],
            ["Fpz", "AFz", "Fz", "FCz", "CPz", "Pz", "POz", "Oz"],
            [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        )

        self._add_landmarks_along_line(["LPA", "Cz", "RPA"], ["T7", "T8"], [0.1, 0.9])

        self._add_landmarks_along_line(
            ["Fpz", "T7", "Oz"],
            ["Fp1", "AF7", "F7", "FT7", "TP7", "P7", "PO7", "O1"],
            [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        )

        self._add_landmarks_along_line(
            ["Fpz", "T8", "Oz"],
            ["Fp2", "AF8", "F8", "FT8", "TP8", "P8", "PO8", "O2"],
            [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9],
        )

        self._add_landmarks_along_line(
            ["T7", "Cz", "T8"],
            ["C5", "C3", "C1", "C2", "C4", "C6"],
            [1 / 8, 2 / 8, 3 / 8, 5 / 8, 6 / 8, 7 / 8],
        )

        self._add_landmarks_along_line(
            ["FT7", "FCz", "FT8"],
            ["FC5", "FC3", "FC1", "FC2", "FC4", "FC6"],
            [1 / 8, 2 / 8, 3 / 8, 5 / 8, 6 / 8, 7 / 8],
        )

        self._add_landmarks_along_line(
            ["F7", "Fz", "F8"],
            ["F5", "F3", "F1", "F2", "F4", "F6"],
            [1 / 8, 2 / 8, 3 / 8, 5 / 8, 6 / 8, 7 / 8],
        )

        self._add_landmarks_along_line(
            ["AF7", "AFz", "AF8"],
            ["AF5", "AF3", "AF1", "AF2", "AF4", "AF6"],
            [1 / 8, 2 / 8, 3 / 8, 5 / 8, 6 / 8, 7 / 8],
        )

        self._add_landmarks_along_line(
            ["TP7", "CPz", "TP8"],
            ["CP5", "CP3", "CP1", "CP2", "CP4", "CP6"],
            [1 / 8, 2 / 8, 3 / 8, 5 / 8, 6 / 8, 7 / 8],
        )

        self._add_landmarks_along_line(
            ["P7", "Pz", "P8"],
            ["P5", "P3", "P1", "P2", "P4", "P6"],
            [1 / 8, 2 / 8, 3 / 8, 5 / 8, 6 / 8, 7 / 8],
        )

        self._add_landmarks_along_line(
            ["PO7", "POz", "PO8"],
            ["PO5", "PO3", "PO1", "PO2", "PO4", "PO6"],
            [1 / 8, 2 / 8, 3 / 8, 5 / 8, 6 / 8, 7 / 8],
        )

        return self.landmarks_mm.pint.quantify()

    def plot(self):
        plt = pv.Plotter()
        cedalion.plots.plot_surface(plt, self.scalp_surface)
        cedalion.plots.plot_labeled_points(plt, self.landmarks_mm.pint.quantify())

        for points in self.lines:
            lines = pv.MultipleLines(points)
            plt.add_mesh(lines, color="m")

        plt.show()
