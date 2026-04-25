"""Blocks of plotting functionality to build visualizations."""

from __future__ import annotations

import itertools

import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
import xarray as xr
from matplotlib.patches import Rectangle
from vtk.util.numpy_support import numpy_to_vtk
from numpy.typing import ArrayLike

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion.dataclasses import PointType

from .colors import COLORBREWER_Q8


__all__ = [
    "plot_stim_markers",
    "plot_segments",
    "plot_surface",
    "plot_labeled_points",
    "plot_vector_field",
    "camera_at_cog",
]

########################################################################################
# matplotlib based
########################################################################################

def plot_stim_markers(
    ax, stim: pd.DataFrame, fmt: dict[str, dict] | None = None, y: float = 0.03
):
    """Add stimulus indicators to an Axes.

    For each trial a Rectangle is plotted in x from onset to onset+duration.
    The height of the rectangle is specified in axes coordinates. In the default
    setting a small bar at bottom of the axes is drawn. By setting y to 1. the
    stimulus marker covers the full height of the axes.

    Args:
        ax: the matplotlib axes to operate on
        stim: a stimulas data frame
        fmt: for each trial_type a dictioniary of keyword arguments can be provided.
            These kwargs are passed to matplotlib.patches.Rectangle to format
            the stimulus indicator.
        y : the height of the Rectangle in axes coordinates.

    Initial Contributors:
        - Eike Middell | middell@tu-berlin.de | 2024
    """
    trans = transforms.blended_transform_factory(
    ax.transData, ax.transAxes)

    base_fmt = {"fc" : 'None'}

    if fmt is None:
        fmt = {}
        trial_types = stim["trial_type"].drop_duplicates().values
        for trial_type, color in zip(trial_types, itertools.cycle(COLORBREWER_Q8)):
            fmt[trial_type] = {"ec": color, "fc": color, "alpha": 0.3}

    labeled_patches = []

    for _, row in stim.iterrows():
        trial_type = row["trial_type"]
        if trial_type in fmt:
            trial_fmt = base_fmt | fmt[trial_type]
        else:
            trial_fmt = base_fmt | {"c": "k"}

        rect = Rectangle(
            (row["onset"], 0),
            row["duration"],
            y,
            transform=trans,
            **trial_fmt,
        )

        # for each trial_type label one patch to put it in the legend
        if trial_type not in labeled_patches:
            rect.set_label(trial_type)
            labeled_patches.append(trial_type)

        ax.add_patch(rect)


########################################################################################


def plot_segments(
    ax,
    segments: list[tuple[float, float]],
    fmt: dict | None = None,
    y: float = 1.0,
    label: str | None = None,
):
    """Highlight time segments in line plots."""

    trans = transforms.blended_transform_factory(
    ax.transData, ax.transAxes)

    if fmt is None:
        color = COLORBREWER_Q8[0]
        fmt = {"ec": color, "fc": color, "alpha": 0.3}

    for i, (start, end) in enumerate(segments):
        rect = Rectangle(
            (start, 0),
            end-start,
            y,
            transform=trans,
            **fmt,
        )

        if (i == 0) and (label is not None):
            rect.set_label(label)

        ax.add_patch(rect)


########################################################################################
# pyvista/VTK based
########################################################################################

def plot_surface(
    plotter: pv.Plotter,
    surface: cdc.Surface,
    color: pv.ColorLike | ArrayLike |  None = None,
    opacity : float =1.0,
    pick_landmarks : list[str] | bool = False,
    **kwargs,
):
    #used for picking landmarks in photogrammetry example
    """Plots a surface mesh with optional landmark picking in a PyVista plotter.

    Args:
        plotter: A PyVista plotter instance used for rendering the surface.
        surface: The surface object to be plotted.
        color: Color of the mesh.
        opacity: Opacity of the mesh, ranging from 0 (transparent) to 1
            (opaque). Default is 1.0.
        pick_landmarks: If True, enables interactive picking of landmarks
            ('Nz', 'Iz', 'Cz', 'Lpa', 'Rpa') on the surface. If a list of
            strings is provided, these are used as the landmark labels instead.
            Default is False.
        **kwargs: Additional keyword arguments are passed to pv.add_mesh.

    Returns:
        function: If `pick_landmarks` is True, returns a function that when called,
        provides the current picked points and their labels. This function prints
        warnings if some labels are missing or are repeated.

    Initial Contributors:
        - Eike Middell | middell@tu-berlin.de | 2024
        - Masha Iudina | mashayudi@gmail.com | 2024
    """

    if isinstance(surface, cdc.VTKSurface):
        mesh = surface.mesh
    elif isinstance(surface, cdc.TrimeshSurface):
        mesh = cdc.VTKSurface.from_trimeshsurface(surface).mesh
    else:
        raise ValueError("unsupported mesh")

    scalars = mesh.GetPointData().GetScalars()

    if color is None:
        if scalars is not None:
            if scalars.GetNumberOfComponents() in [3, 4]:
                rgb = True
            else:
                rgb = False
        else:
            rgb = False
            color = "w"
    else:
        if isinstance(color, list):
            color = np.asarray(color)

        if hasattr(color, "shape") and len(color.shape) == 2:
            if color.shape[0] == surface.nvertices:
                mesh = pv.wrap(mesh)
                mesh["scalars"] = color
                kwargs["scalars"] = "scalars"
                color = None
                rgb = True
            else:
                raise ValueError(
                    "If color is a 2D array it must have as many rows "
                    "as there are vertices in the surface."
                )
        elif hasattr(color, "shape") and len(color.shape) == 1:
            if color.shape[0] == surface.nvertices:
                mesh = pv.wrap(mesh)
                mesh["scalars"] = color
                kwargs["scalars"] = "scalars"
                color = None
                rgb = False
            else:
                raise ValueError(
                    "If color is a 1D array it must have as many items "
                    "as there are vertices in the surface."
                )
        else:
            rgb = False

    if "pickable" not in kwargs:
        kwargs["pickable"] = True
    if "smooth_shading" not in kwargs:
        kwargs["smooth_shading"] = True
    if "split_sharp_edges" not in kwargs:
        kwargs["split_sharp_edges"] = True
    if "feature_angle" not in kwargs:
        kwargs["feature_angle"] = 50

    plotter.add_mesh(mesh, color=color, rgb=rgb, opacity=opacity, **kwargs)


    # Define landmark labels
    if isinstance(pick_landmarks, bool):
        landmark_labels = ['Nz', 'Iz', 'Cz', 'Lpa', 'Rpa']
    else:
        landmark_labels = pick_landmarks
        pick_landmarks = True
    picked_points = []
    labels = []
    point_actors = []
    label_actors = []

    def place_landmark(point):
        nonlocal picked_points, point_actors, label_actors, mesh, labels, plotter
        threshold_distance_squared = 25  # Using squared distance to avoid square root

        new_point = np.array(point)

        # Check if the clicked point is close to any existing point
        for i, existing_point in enumerate(picked_points):
            if np.sum((new_point - existing_point) ** 2) < threshold_distance_squared:
                current_label_index = landmark_labels.index(labels[i])
                next_label_index = (current_label_index + 1) % len(landmark_labels)
                next_label = landmark_labels[next_label_index]

                # Check if the next label is the first one in the list
                if next_label == landmark_labels[0]:
                    # Delete the point and its label
                    del picked_points[i]
                    plotter.remove_actor(label_actors[i])
                    plotter.remove_actor(point_actors[i])
                    del point_actors[i]
                    del label_actors[i]
                    del labels[i]
                    return

                labels[i] = next_label
                plotter.remove_actor(label_actors[i])  # Remove previous label
                label_actors[i] = plotter.add_point_labels(
                    existing_point, [next_label], font_size=30
                )
                return

        # If no point is close enough, create a new point and assign a label
        # Check if there are already 5 points placed
        if len(picked_points) >= len(landmark_labels):
            return

        landmark_label = landmark_labels[(len(picked_points) % len(landmark_labels))]
        # Add new point and label actors
        point_actor = plotter.add_mesh(pv.Sphere(radius=3, center=new_point),
                                       color='green', smooth_shading=True)
        point_actors.append(point_actor)
        label_actor = plotter.add_point_labels(
            new_point, [landmark_label], font_size=30
        )
        label_actors.append(label_actor)
        picked_points.append(new_point)
        labels.append(landmark_label)

    # Initialize the labels list
    # labels = [None] * 5  # Initialize with None for unassigned labels

    if pick_landmarks is True:
        def get_points_and_labels():
            if len(labels) < len(landmark_labels):
                print("Warning: Some labels are missing")
            elif len(set(labels)) != len(landmark_labels):
                print("Warning: Some labels are repeated!")
            
            landmarks = xr.DataArray(
                    np.vstack(picked_points),
                    dims=["label", "digitized"],
                    coords={
                        "label": ("label", labels),
                        "type": ("label", [cdc.PointType.LANDMARK]*len(labels)),
                        "group": ("label", ["L"]*len(labels)),
                        },
                ).pint.quantify("mm")
            return landmarks

        plotter.enable_surface_point_picking(
            callback=place_landmark,
            show_message="Right click to place or change the landmark label.\n" \
                         "Expected labels: "+str(landmark_labels)+"\n" \
                         "Close window when done.",
            show_point=False,
            tolerance=0.005,
        )

        return get_points_and_labels

########################################################################################

def plot_labeled_points(
    plotter: pv.Plotter,
    points: cdt.LabeledPoints,
    color: pv.ColorLike | None = None,
    show_labels: bool = False,
    ppoints: bool = None,
    labels: list[str] | None = None,
    meas_list: xr.DataArray | None = None,
):
    #used in selecting optode centers in Photogrammetry example.
    """Plots `LabeledPoints` with optional interaction for picking points.

        This function visualizes a point cloud where each point can have a label.
        Points can be interactively picked if enabled. Picked point is indicated by
        increased radius.

    Args:
        plotter: A PyVista plotter instance used for rendering the points.
        points: A labeled point cloud data structure containing points and optional
            labels.
        color: Override color for all points. If None, colors are assigned based on
            point types.
        show_labels: If True, labels are displayed next to the points.
        ppoints: A list to store indices of picked points, enables picking if not None.
        labels: List of labels to show if `show_labels` is True. If None and
            `show_labels` is True, the labels from `points` are used.
        meas_list: A DataArray containing channel information for plotting
            channels as lines between sources and detectors.

    Initial Contributors:
        - Eike Middell | middell@tu-berlin.de | 2024
    """

    # FIXME make these configurable
    default_point_colors = {
        PointType.UNKNOWN: "gray",
        PointType.SOURCE: "r",
        PointType.DETECTOR: "b",
        PointType.LANDMARK: "g",
        PointType.ELECTRODE: "pink",
    }
    default_point_sizes = {
        PointType.UNKNOWN: 2,
        PointType.SOURCE: 3,
        PointType.DETECTOR: 3,
        PointType.LANDMARK: 2,
        PointType.ELECTRODE: 3,
    }


    #labels = None
    if labels is None and show_labels:
        labels = points.label.values

    def on_pick(picked_point):
        nonlocal ppoints
        # Define how close points have to be to consider them "super close"
        threshold_distance = 5
        new_point = np.array(picked_point)

        # Check if new point is super close to any existing sphere
        for i, existing_point in enumerate(points):
            if np.linalg.norm(new_point - existing_point) < threshold_distance:
                s = pv.Sphere(radius=4, center=existing_point)
                plotter.add_mesh(s, color='r', smooth_shading=True)
                if i not in ppoints:
                    ppoints.append(i)
                return  # Stop the function after removing the sphere

    # points = points.pint.to("mm").pint.dequantify()  # FIXME unit handling
    points = points.pint.dequantify()  # FIXME unit handling
    # Iterate through each point and its corresponding label
    for i_point, point in enumerate(points):
        # Determine the point type
        if 'type' in point.coords:
            point_type = point.coords['type'].item()
        else:
            point_type = PointType.UNKNOWN

        # Create and add a sphere at the point's coordinates
        s = pv.Sphere(radius=default_point_sizes[point_type], center=point.values)
        plotter.add_mesh(
            s, color=color or default_point_colors[point_type], smooth_shading=True
        )
        # Add the label if required
        if show_labels and labels is not None:
            plotter.add_point_labels(point.values[np.newaxis], [str(labels[i_point])])

	# If measurement list is provided, plot lines between source and detector points
    if meas_list is not None:
        all_points = []
        connectivity = []

        for s, d in zip(meas_list['source'], meas_list['detector']):
            src = points.loc[s].values
            det = points.loc[d].values

            # Add source and detector points to the point list
            idx_offset = len(all_points)
            all_points.extend([src, det])

            # Create connectivity array: [2, pt_id0, pt_id1]
            connectivity.append([2, idx_offset, idx_offset + 1])

        # Create the combined line mesh 
        all_points = np.array(all_points)
        connectivity = np.hstack(connectivity)
        lines = pv.PolyData()
        lines.points = all_points
        lines.lines = connectivity

        plotter.add_mesh(lines, color="k", smooth_shading=True, line_width=2.0)

    if ppoints is not None:
        plotter.enable_surface_point_picking(callback=on_pick, show_point=False)


########################################################################################


def plot_vector_field(
    plotter: pv.Plotter,
    points: cdt.LabeledPoints,
    vectors: xr.DataArray,
    ppoints = None
):
    """Plots a vector field on a PyVista plotter.

    Args:
        plotter (pv.Plotter): A PyVista plotter instance used for rendering the vector
            field.
        points (cdt.LabeledPoints): A labeled point cloud data structure containing
            point coordinates.
        vectors (xr.DataArray): A data array containing the vector field.
        ppoints (list, optional): A list to store indices of picked points, enables
            picking if not None. Default is None.
    """
    assert len(points) == len(vectors)
    assert all(points.label.values == vectors.label.values)
    assert points.points.crs == vectors.dims[1]


    points = points.pint.to("mm").pint.dequantify()
    vectors = vectors.pint.dequantify()

    ugrid = vtk.vtkUnstructuredGrid()

    vpoints = vtk.vtkPoints()
    vpoints.SetData(numpy_to_vtk(points.values))
    ugrid.SetPoints(vpoints)
    ugrid.GetPointData().SetVectors(numpy_to_vtk(vectors.values))

    points = ugrid.GetPoints()

    hedgehog = vtk.vtkHedgeHog()
    hedgehog.SetInputData(ugrid)
    hedgehog.SetScaleFactor(10)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(hedgehog.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor([1.0, 0.0, 0.0])

    plotter.renderer.AddActor(actor)


########################################################################################


def camera_at_cog(
    plt: pv.Plotter,
    surface: cdc.Surface,
    rpos: tuple[float, float, float],
    fp_offset: tuple[float, float, float] = (0, 0, 0),
    up: tuple[float, float, float] = (0, 0, 1),
    fit_scene : bool = False
):
    """Point the camera at the center of gravity of the surface vertices.

    Args:
        plt: the pyvista plotter
        surface: the surface from which the COG should be calculated
        rpos: positin of the camera relative to the COG
        fp_offset: offset from the COG to calculate the focal point
        up: direction of upwards.
        fit_scene: if True the camera is moved along the position-to-focal-point line to
          fit all objects in the scene into the view.
    """

    # FIXME gracefully handle offset units, if given.
    cog = surface.vertices.pint.dequantify().mean("label").values
    plt.camera.position = cog + rpos
    plt.camera.focal_point = cog + fp_offset
    plt.camera.up = up

    if fit_scene:
        plt.reset_camera()
