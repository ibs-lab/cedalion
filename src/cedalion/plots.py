"""Plotting functions for visualization of montages, meshes, etc."""

from __future__ import annotations

import itertools
import sys

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as p
import matplotlib.transforms as transforms
from matplotlib.typing import ColorType
import numpy as np
import pandas as pd
import pyvista as pv
import vtk
import xarray as xr
from matplotlib.colors import ListedColormap, Normalize, LinearSegmentedColormap
from matplotlib.patches import Circle, Ellipse, Rectangle
from numpy.typing import ArrayLike
from PIL import Image
from vtk.util.numpy_support import numpy_to_vtk

import cedalion.data
import cedalion.dataclasses as cdc
import cedalion.geometry.registration as registration
import cedalion.nirs
import cedalion.typing as cdt
from cedalion import Quantity
from cedalion.dataclasses import PointType
from cedalion.imagereco.forward_model import TwoSurfaceHeadModel


def plot_montage3D(amp: xr.DataArray, geo3d: xr.DataArray):
    """Plots a 3D visualization of a montage.

    Args:
        amp (xr.DataArray): Time series data array.
        geo3d (xr.DataArray): Landmark coordinates.
    """
    geo3d = geo3d.pint.dequantify()

    f = p.figure()
    ax = f.add_subplot(projection="3d")
    colors = ["r", "b", "gray"]
    sizes = [20, 20, 2]
    for i, (type, x) in enumerate(geo3d.groupby("type")):
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors[i], s=sizes[i])

    for i in range(amp.sizes["channel"]):
        src = geo3d.loc[amp.source[i], :]
        det = geo3d.loc[amp.detector[i], :]
        ax.plot([src[0], det[0]], [src[1], det[1]], [src[2], det[2]], c="k")

    # if available mark Nasion in yellow
    if "Nz" in geo3d.label:
        ax.scatter(
            geo3d.loc["Nz", 0], geo3d.loc["Nz", 1], geo3d.loc["Nz", 2], c="y", s=25
        )

    ax.view_init(elev=30, azim=145)
    p.tight_layout()


def plot3d(
    brain_mesh,
    scalp_mesh,
    geo3d,
    timeseries,
    poly_lines=[],
    brain_scalars=None,
    plotter=None,
):
    """Plots a 3D visualization of brain and scalp meshes.

    Args:
        brain_mesh (TrimeshSurface): The brain mesh as a TrimeshSurface object.
        scalp_mesh (TrimeshSurface): The scalp mesh as a TrimeshSurface object.
        geo3d (xarray.Dataset): Dataset containing 3-dimentional point centers.
        timeseries: Time series data array.
        poly_lines: List of lists of points to be plotted as polylines.
        brain_scalars: Scalars to be used for coloring the brain mesh.
        plotter (pv.Plotter, optional): An existing PyVista plotter instance to use for
            plotting. If None, a new PyVista plotter instance is created. Default: None.

    Initial Contributors:
        - Eike Middell | middell@tu-berlin.de | 2024
    """

    if plotter is None:
        plt = pv.Plotter()
    else:
        plt = plotter

    if brain_mesh:
        pv_brain = pv.wrap(brain_mesh)
        if brain_scalars is None:
            plt.add_mesh(pv_brain, color="w", smooth_shading=True)
        else:
            plt.add_mesh(pv_brain, scalars=brain_scalars, smooth_shading=True)
    if scalp_mesh:
        pv_scalp = pv.wrap(scalp_mesh)
        plt.add_mesh(pv_scalp, color="w", opacity=0.4, smooth_shading=True)

    point_colors = {
        PointType.SOURCE: "r",
        PointType.DETECTOR: "b",
        PointType.LANDMARK: "green",
        PointType.ELECTRODE: "pink",
    }
    point_sizes = {
        PointType.SOURCE: 3,
        PointType.DETECTOR: 3,
        PointType.LANDMARK: 2,
        PointType.ELECTRODE: 3,
    }
    if geo3d is not None:
        labels = geo3d.label.values
    else:
        labels = None

    if geo3d is not None:
        geo3d = geo3d.pint.to("mm").pint.dequantify()  # FIXME unit handling
        for type, x in geo3d.groupby("type"):
            labels = x.label.values
            for i_point in range(len(x)):
                s = pv.Sphere(radius=point_sizes[type], center=x[i_point])
                plt.add_mesh(s, color=point_colors[type], smooth_shading=True)
                if labels is not None:
                    plt.add_point_labels(x[i_point].values, [str(labels[i_point])])

        # FIXME labels are not rendered
        # plt.add_point_labels(
        #    geo3d.values,
        #    [str(i) for i in geo3d.label.values],
        #    point_size=10,
        #    font_size=20,
        #    always_visible=True,
        # )

    if timeseries is not None:
        for i_chan in range(timeseries.sizes["channel"]):
            src = geo3d.loc[timeseries.source[i_chan], :]
            det = geo3d.loc[timeseries.detector[i_chan], :]
            line = pv.Line(src, det)
            plt.add_mesh(line, color="k", smooth_shading=True)

    for points in poly_lines:
        lines = pv.MultipleLines(points)
        plt.add_mesh(lines, color="m", smooth_shading=True)



def plot_surface(
    plotter: pv.Plotter,
    surface: cdc.Surface,
    color: pv.ColorLike | None = None,
    opacity : float =1.0,
    pick_landmarks : bool = False,
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
        pick_landmarks: If True, enables interactive picking of landmarks on the
            surface. Default is False.
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
        rgb = False

    plotter.add_mesh(mesh, color=color, rgb=rgb, opacity=opacity, smooth_shading=True,
                     pickable=True, **kwargs)


    # Define landmark labels
    landmark_labels = ['Nz', 'Iz', 'Cz', 'Lpa', 'Rpa']
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
        if len(picked_points) >= 5:
            return

        landmark_label = landmark_labels[0]
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

            if len(labels) < 5:
                print("Warning: Some labels are missing")
            elif len(set(labels)) != 5:
                print("Warning: Some labels are repeated!")
            return picked_points, labels

        plotter.enable_surface_point_picking(
            callback=place_landmark,
            show_message="Right click to place or change the landmark label",
            show_point=False,
            tolerance=0.005,
        )

        return get_points_and_labels

def plot_labeled_points(
    plotter: pv.Plotter,
    points: cdt.LabeledPointCloud,
    color: pv.ColorLike = None,
    show_labels: bool = False,
    ppoints: bool = None,
    labels: list[str] | None = None,
):
    #used in selecting optode centers in Photogrammetry example.
    """Plots a labeled point cloud with optional interaction for picking points.

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
        point_type = point.coords['type'].item()
        # Create and add a sphere at the point's coordinates
        s = pv.Sphere(radius=default_point_sizes[point_type], center=point.values)
        plotter.add_mesh(
            s, color=color or default_point_colors[point_type], smooth_shading=True
        )
        # Add the label if required
        if show_labels and labels is not None:
            plotter.add_point_labels(point.values[np.newaxis], [str(labels[i_point])])

    if ppoints is not None:
        plotter.enable_surface_point_picking(callback=on_pick, show_point=False)



def plot_vector_field(
    plotter: pv.Plotter,
    points: cdt.LabeledPointCloud,
    vectors: xr.DataArray,
    ppoints = None
):
    """Plots a vector field on a PyVista plotter.

    Args:
        plotter (pv.Plotter): A PyVista plotter instance used for rendering the vector
            field.
        points (cdt.LabeledPointCloud): A labeled point cloud data structure containing
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

class OptodeSelector:
    """A class for visualizing point clouds with interactive features in PyVista.

    This class provides functionality to visualize and interact with labeled point
    clouds using a PyVista plotter. It allows points to be dynamically added or removed
    by picking them directly from the plot interface.

    Attributes:
        surface (cdc.Surface): The surface of a head for normals.
        points (cdt.LabeledPointCloud): The point cloud data containing point
            coordinates.
        normals (xr.DataArray): Normal vectors to the points.
        plotter (pv.Plotter): A PyVista plotter instance for rendering the point cloud.
        labels (list of str, optional): Labels corresponding to the points, displayed
            if provided.
        actors (list): List of PyVista actor objects representing the points in the
            visualization.
        color (str or tuple, optional): Default color for points if not specified by
            point type.

    Methods:
        plot(): Renders the point cloud using the current settings.
        on_pick(picked_point): Callback function for picking points in the visualization
        update_visualization(): Clears the existing plot and re-renders the point cloud.
        enable_picking(): Enables interactive picking of points on the plot.

    Initial Contributors:
        - Masha Iudina | mashayudi@gmail.com | 2024
    """
    def __init__(self, surface, points, normals=None, plotter=None, labels = None):
        self.points = points
        self.normals = normals
        self.surface = surface
        self.plotter = plotter if plotter else pv.Plotter()
        self.labels = labels
        self.actors = []
        self.color = None

        self.cog = surface.mesh.vertices.mean(axis=0)

    def plot(self):
        plotter = self.plotter
        points = self.points.pint.dequantify()
        color = 'r'
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

        # points = points.pint.to("mm").pint.dequantify()  # FIXME unit handling
        # points = points.pint.dequantify()  # FIXME unit handling
        for type, x in points.groupby("type"):
            for i_point in range(len(x)):

                s = pv.Sphere(radius=default_point_sizes[type], center=x[i_point])
                if color is None:
                    sphere_actor = plotter.add_mesh(
                        s, color=default_point_colors[type], smooth_shading=True
                    )
                else:
                    sphere_actor = plotter.add_mesh(s, color=color, smooth_shading=True)
                self.actors.append(sphere_actor)
                if self.labels is not None:
                    plotter.add_point_labels(
                        x[i_point].values, [str(self.labels[i_point])]
                    )


    def on_pick(self, picked_point):
        plotter = self.plotter
        points = self.points.pint.dequantify()
        # Define how close points have to be to consider them "super close"
        threshold_distance = 5
        new_point = np.array(picked_point)

        # Check if new point is super close to any existing sphere
        for i, existing_point in enumerate(points.values):
            if np.linalg.norm(new_point - existing_point) < threshold_distance:
                idx_to_remove = i
                indexes = np.arange(len(self.points.label))
                selected_indexes = np.delete(indexes, idx_to_remove)

                self.points = self.points.isel(label=selected_indexes)
                if self.normals is not None:
                    self.normals = self.normals.isel(label=selected_indexes)
                self.plotter.remove_actor(self.actors[idx_to_remove])
                del self.actors[idx_to_remove]

                return  # Stop the function after removing the sphere

        existing_labels = self.points.coords['label'].values
        # Generate a new unique label
        new_label_number = (
            max([int(label.split("-")[-1]) for label in existing_labels]) + 1
        )
        new_label = f'O-{new_label_number}'
        new_group = self.points.coords['group'].values[0]
        new_type = cdc.PointType.LANDMARK if new_group == 'O' else cdc.PointType.UNKNOWN

        # Create the new entry DataArray
        new_center_coords = new_point

        s = pv.Sphere(radius=2, center=new_point)
        sphere_actor = plotter.add_mesh(s, color='r', smooth_shading=True)
        self.actors.append(sphere_actor)

        new_normal = self.find_surface_normal(new_point)
        self.points = self.points.points.add(
            new_label, new_center_coords, new_type, new_group
        )
        self.normals = self.update_normals(new_normal, new_label)

    def update_visualization(self):
        # Clear existing plot and re-plot with the updated self.points
        self.plotter.clear()
        self.plot()

    def enable_picking(self):
        self.plotter.enable_surface_point_picking(
            callback=self.on_pick,
            show_message="Right click to place or remove optode",
            show_point=False,
            tolerance=0.005,
        )

    def find_surface_normal(self, picked_point, radius=6):
        def pca(vertices: np.ndarray):
            eigenvalues, eigenvecs = np.linalg.eigh(np.cov(vertices.T))

            # sort by increasing eigenvalue
            indices = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[indices]
            eigenvecs = eigenvecs[:, indices]

            return eigenvalues, eigenvecs
        # Calculate distances from picked point to all vertices in the mesh
        distances = np.linalg.norm(self.surface.mesh.vertices - picked_point, axis=1)

        # Select vertices within the specified radius
        close_vertices = self.surface.mesh.vertices[distances < radius]

        # calculate normal from eigenvector
        eigenvalues, eigenvecs = pca(close_vertices)
        normal_vector = eigenvecs[:, 0]

        # Verify the direction of the normal
        if np.dot(normal_vector, picked_point - self.cog) < 0:
            normal_vector = -normal_vector  # Ensure the normal points outward
        return normal_vector

    def update_normals(self, normal_at_picked_point, label):
        new_normals = xr.DataArray(
            np.vstack([normal_at_picked_point]),
            dims=["label", self.surface.crs],
            coords={
                "label": ("label", [label]),
                "group": ("label", ["O"]),
            },
        ).pint.quantify("1")

        return xr.concat((self.normals, new_normals), dim="label")



COLORBREWER_Q8 = [
    "#e41a1c",
    "#4daf4a",
    "#377eb8",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
]


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

def plot_segments(
    ax,
    segments: list[tuple[float, float]],
    fmt: dict | None = None,
    y: float = 1.0,
    label: str | None = None,
):
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


def scalp_plot(
    ts: cdt.NDTimeSeries,
    geo3d: cdt.LabeledPointCloud,
    metric: xr.DataArray | ArrayLike,
    ax,
    title: str | None = None,
    y_title: float = None,
    vmin: float | None = None,
    vmax: float | None = None,
    center : float | None = None,
    cmap: str | matplotlib.colors.Colormap = "bwr",
    norm: Normalize | None = None,
    bad_color: ColorType = [0.7, 0.7, 0.7],
    min_dist: Quantity | None = None,
    min_metric: float | None = None,
    channel_lw: float = 2.0,
    optode_size: float = 36.0,
    optode_labels: bool = False,
    cb_label: str | None = None,
    cb_ticks_labels: list[(float, str)] | None = None,
    add_colorbar: bool = True,
    zorder : str | None = None,
):
    """Creates a 2D plot of the head with channels coloured according to a given metric.

    Args:
        ts: a NDTimeSeries to provide channel definitions
        geo3d: a LabeledPointCloud to provide the probe geometry
        metric ((:class:`DataArray`, (channel,) | ArrayLike)): the scalar metric to be
            plotted for each channel. If provided as a DataArray it needs a channel
            dimension. If provided as a plain array or list it must have the same
            length as ts.channel and the matching is done by position.
        ax: the matplotlib.Axes object into which to draw
        title: the axes title
        y_title: the y position of the title in axes coordinates
        vmin: the minimum value of the metric
        vmax: the maximum value of the metric
        center: when calculating vmin and vmax, center the value range at this value.
        cmap: the name of the colormap
        norm: normalization for color map
        bad_color: the color to use when the metric contains NaNs
        min_dist: if provided channels below this distance threshold are not drawn
        min_metric: if provided channels below this metric threshold are toned down
        channel_lw: channel line width
        optode_size: optode marker size
        optode_labels: if True draw optode labels instead of markers
        cb_label: colorbar label
        cb_ticks_labels: ticks and labels for colorbar
        add_colorbar: if true a colorbar is added to the plot
        zorder: 'ascending' or 'descending' or None. Controls whether channels
            with high or low metric values are plotted on top.

    Initial Contributors:
        - Laura Carlton | lcarlton@bu.edu | 2024
        - Eike Middell | middell@tu-berlin.de | 2024
    """


    geo2d = registration.simple_scalp_projection(geo3d)
    channel_dists = cedalion.nirs.channel_distances(ts, geo3d)


    if not isinstance(metric, xr.DataArray):
        if len(metric) != ts.sizes["channel"]:
            raise ValueError("metric is not a DataArray and does not match in size.")

        metric = xr.DataArray(metric, dims=["channel"], coords={"channel": ts.channel})

    metric_channels = set(metric.channel.values)

    # FIXME use metric unit in colorbar label?
    metric = metric.pint.dequantify()

    channel = ts.channel.values
    source = ts.source.values
    detector = ts.detector.values

    if norm is None:
        if vmin is None:
            vmin = np.nanmin(metric)
        if vmax is None:
            vmax = np.nanmax(metric)

        if center is not None:
            delta = max(abs(vmin-center), abs(vmax-center))
            vmin = center - delta
            vmax = center + delta
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        if (vmin is not None) or (vmax is not None) or (center is not None):
            raise ValueError("Specify either norm or vmin/vmax/center.")

    if isinstance(cmap, str):
        cmap = p.cm.get_cmap(cmap)

    cmap.set_bad(bad_color)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal", adjustable="datalim")

    # head and ears
    ax.add_patch(Circle((0,0), 1., ec="k", fc="None"))
    ax.add_patch(Ellipse((1.05, 0), .1, .3, ec="k", fc="None"))
    ax.add_patch(Ellipse((-1.05, 0), .1, .3, ec="k", fc="None"))

    # nose marker
    angles = [np.pi/2 + .05, np.pi/2, np.pi/2 + -.05]
    r = [1., 1.1, 1.0]
    ax.plot( r * np.cos(angles), r * np.sin(angles), "k-")


    # draw lines for channels
    used_sources = set()
    used_detectors = set()

    extend_upper = False
    extend_lower = False

    for ch,src,det,dist in zip(channel, source, detector, channel_dists):
        s = geo2d.loc[src]
        d = geo2d.loc[det]

        if (min_dist is not None) and (dist.item() < min_dist):
            continue

        used_sources.add(str(src))
        used_detectors.add(str(det))

        if ch in metric_channels:
            v = metric.sel(channel=ch).item()
        else:
            v = np.nan

        normed_v = norm(v)

        # check if any channel metric exceeds vmin or vmax
        if (normed_v > 1).any():
            extend_upper = True
        if (normed_v < 0).any():
            extend_lower = True

        c = cmap(normed_v)
        line_fmt = {'c' : c, 'ls' : '-', 'lw' : channel_lw, 'alpha' : 1.0}

        if (min_metric is not None) and (v < min_metric):
            line_fmt['alpha'] = 0.4

        if zorder is None:
            zorder_line = 0
        elif zorder == "ascending":
            zorder_line = normed_v
        elif zorder == "descending":
            zorder_line = 1 - normed_v
        else:
            raise ValueError(f"unexpected value '{zorder}' for zorder.")


        ax.plot([s[0], d[0]], [s[1], d[1]], zorder=zorder_line, **line_fmt)

    # draw markers or labels for sources and detectors
    # /!\ isin with np strings and sets is tricky. probably because of the hash
    s = geo2d.sel(label=geo2d.label.isin(list(used_sources)))
    d = geo2d.sel(label=geo2d.label.isin(list(used_detectors)))

    COLOR_SOURCE = "#e41a1c" # colorbrewer red
    COLOR_DETECTOR = "#377eb8" # colorbrewer blue

    if optode_labels:
        for sd, color in [(s, COLOR_SOURCE), (d, COLOR_DETECTOR)]:
            for i in range(len(sd)):
                ax.text(
                    sd[i, 0],
                    sd[i, 1],
                    sd.label.values[i],
                    ha="center",
                    va="center",
                    fontsize="small",
                    weight="semibold",
                    color=color,
                    zorder=200)
    else:
        ax.scatter(
            s[:, 0],
            s[:, 1],
            s=optode_size,
            marker="s",
            fc=COLOR_SOURCE,
            zorder=100,
        )
        ax.scatter(
            d[:, 0],
            d[:, 1],
            s=optode_size,
            marker="s",
            fc=COLOR_DETECTOR,
            zorder=100,
        )

    # remove axes and ticks
    ax.set_axis_off()

    # colorbar
    if add_colorbar:
        if extend_upper and extend_lower:
            extend = "both"
        elif extend_upper and not extend_lower:
            extend = "max"
        elif not extend_upper and extend_lower:
            extend = "min"
        else:
            extend = "neither"

        cb = p.colorbar(
            matplotlib.cm.ScalarMappable(cmap=cmap,norm=norm),
            ax=ax,
            shrink=0.6,
            extend=extend
        )
        cb.set_label(cb_label)
        if cb_ticks_labels is not None:
            cb.set_ticks([tick for tick, _ in cb_ticks_labels])
            cb.set_ticklabels([label for _, label in cb_ticks_labels])

    if title:
        ax.set_title(title, y=y_title)


    #cb.set_ticks([vmin, (vmin+vmax)//2, vmax])



def brain_plot(
    ts: cdt.NDTimeSeries,
    geo3d: cdt.LabeledPointCloud,
    metric: xr.DataArray | ArrayLike,
    brain_surface: cdc.TrimeshSurface,
    ax,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | matplotlib.colors.Colormap = "RdBu_r",
    bad_color: ColorType = [0.7, 0.7, 0.7],
    cb_label: str = "",
    camera_pos: ArrayLike | str | None = None,
):
    if vmin is None:
        vmin = np.nanmin(metric)
    if vmax is None:
        vmax = np.nanmax(metric)

    cmap = p.cm.get_cmap(cmap)
    cmap.set_bad(bad_color)

    vertices = brain_surface.mesh.vertices
    center_brain = np.mean(vertices, axis=0)

    brain_surface = cdc.VTKSurface.from_trimeshsurface(brain_surface)
    brain_surface = pv.wrap(brain_surface.mesh)

    plt = pv.Plotter(off_screen=True)

    plt.add_mesh(
        brain_surface,
        scalars=metric,
        cmap=cmap,
        clim=(vmin, vmax),
        scalar_bar_args={"title": cb_label},
        smooth_shading=True,
    )

    if camera_pos is not None:
        if isinstance(camera_pos, str):
            if camera_pos not in geo3d.label:
                raise ValueError(f"camera_pos was set to '{camera_pos}' but this label"
                                 " does not exist in geo3d.")
            lm_pos = geo3d.sel(label=camera_pos).values
            camera_pos = center_brain + 6 * (lm_pos - center_brain)

        plt.camera.position = camera_pos
        plt.camera.focal_point = center_brain
        plt.camera.up = [0, 0, 1]

    if title:
        plt.add_text(title, position="upper_edge", font_size=20)

    # determine size of the axes in pixels
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    width = int(bbox.width * ax.figure.dpi * 2)
    height = int(bbox.height * ax.figure.dpi * 2)

    # FIXME plt.screenshot uses vtk functionality, which hijacks sys.stdout by replacing
    # it with vtkPythonStdStreamCaptureHelper. We don't want this.
    _stdout = sys.stdout

    # render 3D scene and create image
    image = plt.screenshot(window_size=(width, height))

    # reset stdout to previous one
    sys.stdout = _stdout

    # show image in matplotlib axes
    ax.imshow(image)

    # remove ticks
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])


def scalp_plot_gif(
        data_ts: cdt.NDTimeSeries,
        geo3d: cdt.LabeledPointCloud,
        filename: str,
        time_range: tuple = None,
        cmap: str | matplotlib.colors.Colormap = 'seismic',
        scl=None,
        fps: int =10,
        optode_size: float = 6,
        optode_labels: bool =False,
        str_title: str =''
        ):
    """Generate a GIF of scalp topographies over time from time-series data.

    Args:
        data_ts : xarray.DataArray
            A 2D DataArray with dimensions (channel, time). Must include coordinate
            labels for 'source' and 'detector' in the 'channel' dimension.
        geo3d : cedalion.core.LabeledPointCloud
            3D geometry defining optode locations for projecting onto the scalp surface.
        filename : str
            Full path to the output GIF file without file extension.
        time_range: tuple, optional
           Provides (start_time, stop_time, step_time) in quantity 's' for generating
           animation.
        cmap : string, optional
            A matplotlib colormap name or a Colormap object. Default is 'seismic'.
        scl : tuple of (float, float), optional
            Tuple defining the (vmin, vmax) for the color scale. If None, the color
            scale is set to ± the maximum absolute value of the data.
        fps : int, optional
            Frames per second for the output GIF. Default is 10.
        optode_size : float, optional
            Size of optode markers on the plot. Default is 6.
        optode_labels : bool, optional
            Whether to show text labels for optodes instead of markers. Default: False.
        str_title : str, optional
            Extra string to append to the title of each frame.

    Returns:
        None.
        The function saves a GIF file to the specified location.

    Initial Contributors:
        - David Boas | dboas@bu.edu | 2025
        - Alexander von Lühmann | vonluehmann@tu-berlin.de | 2025
    """

    data_ts = data_ts.pint.dequantify()

    if ("time" in data_ts.dims and data_ts.sizes["time"] > 1) or (
        "reltime" in data_ts.dims and data_ts.sizes["reltime"] > 1
    ):

        # If time_range is not provided, default to using the range in X_ts
        if time_range is None:
            start_time = float(data_ts.time.values[0])
            end_time = float(data_ts.time.values[-1])
            step_time = (end_time - start_time) / max((data_ts.sizes["time"] - 1), 1)
        else:
            # Convert each element from the time_range tuple to seconds
            start_time = time_range[0].to('s').magnitude
            end_time = time_range[1].to('s').magnitude
            step_time = time_range[2].to('s').magnitude

        # Create an array of time points to iterate over
        time_points = np.arange(start_time, end_time + step_time, step_time)
        # Select the subset of data within the given time range
        X_subset = data_ts.sel(time=slice(start_time, end_time))

        # Initialize using the first time point
        # (using nearest in case of slight mismatches)
        X_frame = X_subset.sel(time=time_points[0], method="nearest")

    filename = filename+'.gif'

    if scl is None:
        absmax = np.max(np.abs(data_ts.values)) * (1+1e-6) # eps to avoid cb-extension
        scl = (-absmax,absmax)

    frames = []

    # Iterate over the time points
    for current_time in time_points:
        # Select the frame closest to the current time point
        X_frame = X_subset.sel(time=current_time, method="nearest")

        f,ax = p.subplots(1, 1, figsize=(8, 8))
        # reset position to avoid inset growth from colorbar
        ax.set_position([0.1, 0.1, 0.8, 0.8])
        scalp_plot(
            data_ts,
            geo3d,
            X_frame.values,
            ax,
            cmap=cmap,
            vmin=scl[0],
            vmax=scl[1],
            optode_labels=optode_labels,
            title=f"Time: {float(current_time):0.1f}s\n{str_title}",
            optode_size=optode_size,
            add_colorbar=True,
        )
        ax.figure.canvas.draw()
        rgba = np.asarray(ax.figure.canvas.buffer_rgba())
        image = Image.fromarray(rgba)
        frames.append(image)
        p.close(f)

    frames[0].save(
        filename, save_all=True, append_images=frames[1:], duration=1000 / fps, loop=0
    )


def image_recon(
    X: cdt.NDTimeSeries,
    head: TwoSurfaceHeadModel,
    cmap: str | matplotlib.colors.Colormap = 'seismic',
    clim=None,
    view_type: str ='hbo_brain',
    view_position: str ='superior',
    p0=None,
    title_str: str = None,
    off_screen: bool =False,
    plotshape=(1, 1),
    iax=(0, 0),
    show_scalar_bar: bool = False,
    wdw_size: tuple = (1024, 768)
):
    """Render a single frame of brain or scalp activity on a specified view.

    This function creates (or reuses) a PyVista plotter, applies a custom colormap,
    sets the camera view according to the given view_position, adds the surface mesh
    with the scalar data (extracted from X), and returns the plotter, the mesh, and a
    text label.

    Args:
        X: cdt.NDTimeSeries (or similar)
            Scalar data for the current frame. Expected to have a boolean attribute
            `is_brain` indicating brain vs. non-brain vertices, and HbO / HbR
            chromophore dimension
        head: TwoSurfaceHeadModel
            A head model containing attributes such as `head.brain` and `head.scalp`.
        cmap: str or matplotlib.colors.Colormap, default 'seismic'
            The colormap to use.
        clim: tuple, optional
            Color limits. If None, they are computed from the data.
        view_type: str, default 'hbo_brain'
            Indicates whether to plot brain ('hbo_brain' or 'hbr_brain') or scalp
            ('hbo_scalp' or 'hbr_scalp') data.
        view_position: str, default 'superior'
            The view direction. Options are:
            'superior', 'anterior', 'posterior','left', 'right', and 'scale_bar'.
        p0: PyVista Plotter instance, optional
            If provided the mesh is added to this plotter; else a new plotter is created
        title_str: str, optional
            Title to use on the scalar bar.
        off_screen: bool, default False
            Whether to use off-screen rendering.
        plotshape: tuple, default (1, 1)
            The subplot grid shape.
        iax: tuple, default (0, 0)
            The target subplot index (row, col).
        show_scalar_bar: bool, optional
            Flag to control scalar bar visibility
        wdw_size: tuple, default (1024, 768)
            The window size for the plotter (the plot resolution)

    Returns:
        A tuple (p0, surf, surf_label) where:
          - p0: the PyVista Plotter instance.
          - surf: the wrapped surface mesh (a pyvista mesh).
          - surf_label: a text actor (e.g., the scalar bar label).

    Initial Contributors:
    - David Boas | dboas@bu.edu | 2025
    - Laura Carlton | lcarlton@bu.edu | 2025
    - Alexander von Lühmann | vonluehmann@tu-berlin.de | 2025
    """
    # Create colormap and custom version
    cmap_obj = p.get_cmap(cmap, 1024)
    new_cmap_colors = np.vstack((cmap_obj(np.linspace(0, 1, 256))))
    custom_cmap = ListedColormap(new_cmap_colors)

    # Separate the scalar data
    X_hbo_brain = X.sel(chromo='HbO')[X.is_brain.values]
    X_hbr_brain = X.sel(chromo='HbR')[X.is_brain.values]
    X_hbo_scalp = X.sel(chromo='HbO')[~X.is_brain.values]
    X_hbr_scalp = X.sel(chromo='HbR')[~X.is_brain.values]

    # Define view directions
    positions = {
        'superior': [0, 0, 1],
        'left': [-1, 0, 0],
        'right': [1, 0, 0],
        'anterior': [0, 1, 0],
        'posterior': [0, -1, 0],
        'scale_bar': [0, 0, 1]
    }
    camera_direction = positions.get(view_position, [0, 0, 1])

    # Create a new plotter if none is provided
    if p0 is None:
        p0 = pv.Plotter(
            shape=(plotshape[0], plotshape[1]),
            window_size=wdw_size,
            off_screen=off_screen,
        )
    p0.subplot(iax[0], iax[1])

    # Select the appropriate head surface based on flag_hbx
    if view_type in ['hbo_brain', 'hbr_brain']:
        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
    elif view_type in ['hbo_scalp', 'hbr_scalp']:
        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
    else:
        raise ValueError(f"Invalid flag_hbx: {view_type}")
    surf = pv.wrap(surf.mesh)
    centroid = np.mean(surf.points, axis=0)

    # Set the scalar data on the mesh and compute clim if needed
    if view_type == 'hbo_brain':
        surf['brain'] = X_hbo_brain
        if clim is None:
            clim = (-X_hbo_brain.max(), X_hbo_brain.max())
        p0.add_mesh(surf, scalars='brain', cmap=custom_cmap, clim=clim,
                    show_scalar_bar=False, nan_color=(0.9, 0.9, 0.9),
                    smooth_shading=True, interpolate_before_map=False)
    elif view_type == 'hbr_brain':
        surf['brain'] = X_hbr_brain
        if clim is None:
            clim = (-X_hbr_brain.max(), X_hbr_brain.max())
        p0.add_mesh(surf, scalars='brain', cmap=custom_cmap, clim=clim,
                    show_scalar_bar=False, nan_color=(0.9, 0.9, 0.9),
                    smooth_shading=True, interpolate_before_map=False)
    elif view_type == 'hbo_scalp':
        surf['brain'] = X_hbo_scalp
        if clim is None:
            clim = (-X_hbo_scalp.max(), X_hbo_scalp.max())
        p0.add_mesh(surf, scalars='brain', cmap=custom_cmap, clim=clim,
                    show_scalar_bar=False, nan_color=(0.9, 0.9, 0.9),
                    smooth_shading=True, interpolate_before_map=False)
    elif view_type == 'hbr_scalp':
        surf['brain'] = X_hbr_scalp
        if clim is None:
            clim = (-X_hbr_scalp.max(), X_hbr_scalp.max())
        p0.add_mesh(surf, scalars='brain', cmap=custom_cmap, clim=clim,
                    show_scalar_bar=False, nan_color=(0.9, 0.9, 0.9),
                    smooth_shading=True, interpolate_before_map=False)

    # Set camera: adjust 'view_up' depending on the view position
    view_up = [0, 1, 0] if view_position == 'superior' else [0, 0, 1]
    p0.camera_position = [
        centroid + np.array(camera_direction) * 500,
        centroid,
        view_up,
    ]

    # Add the scalar bar or view label for multiview plot
    if iax == (1, 1):
        p0.clear_actors()
        p0.add_scalar_bar(
            title=title_str,
            vertical=False,
            position_x=0.1,
            position_y=0.5,
            height=0.1,
            width=0.8,
            fmt="%.1e",
            label_font_size=16,
            title_font_size=32,
        )
        surf_label = p0.add_text('', position='upper_left', font_size=10)
    else:
        surf_label = p0.add_text(view_position, position='lower_left', font_size=10)
    # add scalar bar to (each) single view if flag is set
    if show_scalar_bar:
        p0.add_scalar_bar(
            title=title_str, fmt="%.1e", label_font_size=24, title_font_size=32
        )

    return p0, surf, surf_label



def image_recon_view(
    X_ts: cdt.NDTimeSeries,
    head: TwoSurfaceHeadModel,
    cmap: str | matplotlib.colors.Colormap = 'seismic',
    clim = None,
    view_type: str ='hbo_brain',
    view_position: str ='superior',
    title_str: str = None,
    filename: str =None,
    SAVE: bool = False,
    time_range: tuple = None,
    fps: int = 6,
    geo3d_plot: cdt.LabeledPointCloud = None,
    wdw_size: tuple = (1024, 768)
):
    """Generate a single-view visualization of head activity.

    For static data (2D: vertex × channel) the function can display (or save) a single
    frame. For time series data (3D: vertex × channel × time) the function can create an
    animated GIF by looping over the specified frame indices.

    Args:
        X_ts: xarray.DataArray or NDTimeSeries
            Activity data. If 2D, a single static frame is plotted; if 3D, a time series
            is used. Expected to have a boolean attribute `is_brain` indicating brain
            vs. non-brain vertices, and HbO / HbR chromophore dimension
        head: TwoSurfaceHeadModel
            The head mesh data to plot activity on.
        cmap: str or matplotlib.colors.Colormap, default 'seismic'
            The colormap to use.
        view_position: str, default 'superior'
            The view to render.
        clim: tuple, optional
            Color limits. If None, they are computed from the data.
        view_type: str, default 'hbo_brain'
            Indicates whether to plot brain ('hbo_brain' or 'hbr_brain') or scalp
            ('hbo_scalp' or 'hbr_scalp') data.
        view_position: str, default 'superior'
            The view direction. Options are:
            'superior', 'anterior', 'posterior','left', 'right', and 'scale_bar'.
        title_str: str, optional
            Title to use on the scalar bar.
        filename: str, optional
            The output filename (without extension) for saving the image/GIF.
        SAVE: bool, default False
            If True, the resulting still image is saved, otherwise only shown. Rendered
            gifs are always saved.
        time_range: tuple, optional
           Provides (start_time, stop_time, step_time) in quantity 's' for generating
           animation.
        fps: int, default 6
            Frames per second for the GIF.
        geo3d_plot: cdt.LabeledPointCloud, optional
            A 3D point cloud for plotting labeled points (e.g. optodes) on the mesh.
        wdw_size: tuple, default (1024, 768)
            The window size for the plotter (the plot resolution)

    Returns: Nothing

    Initial Contributors:
    - David Boas | dboas@bu.edu | 2025
    - Laura Carlton | lcarlton@bu.edu | 2025
    - Alexander von Lühmann | vonluehmann@tu-berlin.de | 2025
    """

    # Animated case (time dimension exists with more than one element):
    # check for frame indices
    if ("time" in X_ts.dims and X_ts.sizes["time"] > 1) or (
        "reltime" in X_ts.dims and X_ts.sizes["reltime"] > 1
    ):
        # If time_range is not provided, default to using the range in X_ts
        if time_range is None:
            start_time = float(X_ts.time.values[0])
            end_time = float(X_ts.time.values[-1])
            step_time = (end_time - start_time) / max((X_ts.sizes["time"] - 1), 1)
        else:
            # Convert each element from the time_range tuple to seconds
            start_time = time_range[0].to('s').magnitude
            end_time = time_range[1].to('s').magnitude
            step_time = time_range[2].to('s').magnitude

        # Create an array of time points to iterate over
        time_points = np.arange(start_time, end_time + step_time, step_time)
        # Select the subset of data within the given time range
        X_subset = X_ts.sel(time=slice(start_time, end_time))

        # Initialize using the first time point
        # (using nearest in case of slight mismatches)
        X_frame = X_subset.sel(time=time_points[0], method="nearest")

        # Initialize using the first time point
        # (using nearest in case of slight mismatches)
        X_frame = X_subset.sel(time=time_points[0], method="nearest")

        p0, surf, label = image_recon(
            X_frame, head, cmap=cmap, clim=clim, view_type=view_type,
            view_position=view_position, title_str=title_str, off_screen=True,
            show_scalar_bar=True, wdw_size=wdw_size
        )

        # add labeled points if they were handed in
        if geo3d_plot is not None:
            plot_labeled_points(p0, geo3d_plot)

        if SAVE and filename:
            # Open GIF output with desired fps; filename will have a .gif extension
            p0.open_gif(filename + '.gif', fps=fps)
        else:
            assert filename is None, (
                "Filename must be provided to generate and save GIF."
            )

        # Loop over frames, update the mesh's scalar data, and update the text label
        for current_time in time_points:
            X_frame = X_subset.sel(time=current_time, method="nearest")
            if view_type == 'hbo_brain':
                new_data = X_frame.sel(chromo='HbO').where(X_ts.is_brain, drop=True)
            elif view_type == 'hbr_brain':
                new_data = X_frame.sel(chromo='HbR').where(X_ts.is_brain, drop=True)
            elif view_type == 'hbo_scalp':
                new_data = X_frame.sel(chromo='HbO').where(~X_ts.is_brain, drop=True)
            elif view_type == 'hbr_scalp':
                new_data = X_frame.sel(chromo='Hbr').where(~X_ts.is_brain, drop=True)
            else:
                new_data = None

            surf['brain'] = new_data
            if label:
                # Update the label text with the current time
                # (assumes X_ts has a 'time' coordinate)
                label.set_text('upper_left', f"Time = {float(current_time):0.1f} sec")
            p0.write_frame()

        p0.close()  # This finalizes and writes the GIF file.

    # Static image: no time dimension or only one time step available
    else:
        p0, _, _ = image_recon(
                X_ts, head, cmap=cmap, clim=clim, view_type=view_type,
                view_position=view_position, title_str=title_str, off_screen=False,
                show_scalar_bar=True, wdw_size=wdw_size
            )
        # add labeled points if they were handed in
        if geo3d_plot is not None:
            plot_labeled_points(p0, geo3d_plot)

        if SAVE and filename:
            p0.show()
            p0.screenshot(filename + '.png')
        else:
            p0.show()



def image_recon_multi_view(
    X_ts: cdt.NDTimeSeries,
    head: TwoSurfaceHeadModel,
    cmap: str | matplotlib.colors.Colormap = 'seismic',
    clim = None,
    view_type: str ='hbo_brain',
    title_str: str = None,
    filename: str =None,
    SAVE: bool = True,
    time_range: tuple = None,
    fps: int = 6,
    geo3d_plot: cdt.LabeledPointCloud = None,
    wdw_size: tuple = (1024, 768)
):
    """Generate a multi-view (2×3 grid) vis. of head activity across different views.

    For static data (2D: vertex × channel) the function can display (or save) a single
    frame. For time series data (3D: vertex × channel × time) the function creates an
    animated GIF where each frame updates all views.

    Args:
        X_ts: xarray.DataArray or NDTimeSeries
            Activity data. If 2D, a single static frame is plotted; if 3D, a time series
            is used. Expected to have a boolean attribute `is_brain` indicating brain
            vs. non-brain vertices, and HbO / HbR chromophore dimension
        head: TwoSurfaceHeadModel
            The head mesh data to plot activity on.
        cmap: str or matplotlib.colors.Colormap, default 'seismic'
            The colormap to use.
        view_position: str, default 'superior'
            The view to render.
        clim: tuple, optional
            Color limits. If None, they are computed from the data.
        view_type: str, default 'hbo_brain'
            Indicates whether to plot brain ('hbo_brain' or 'hbr_brain') or scalp
            ('hbo_scalp' or 'hbr_scalp') data.
        title_str: str, optional
            Title to use on the scalar bar.
        filename: str, optional
            The output filename (without extension) for saving the image/GIF.
        SAVE: bool, default False
            If True, the resulting still image is saved, otherwise only shown. Rendered
            gifs are always saved.
        time_range: tuple, optional
           Provides (start_time, stop_time, step_time) in quantity 's' for generating
           animation.
        fps: int, default 6
            Frames per second for the GIF.
        geo3d_plot: cdt.LabeledPointCloud, optional
            A 3D point cloud for plotting labeled points (e.g. optodes) on the mesh.
        wdw_size: tuple, default (1024, 768)
            The window size for the plotter (the plot resolution)

    Returns: Nothing

    Initial Contributors:
    - David Boas | dboas@bu.edu | 2025
    - Laura Carlton | lcarlton@bu.edu | 2025
    - Alexander von Lühmann | vonluehmann@tu-berlin.de | 2025
    """


    subplot_shape = (2, 3)
    # Define the subplot positions for each view
    views_positions = {
        'scale_bar': (1, 1),
        'left': (0, 0),
        'superior': (0, 1),
        'right': (0, 2),
        'anterior': (1, 0),
        'posterior': (1, 2)
    }

    # Animated case (time dimension exists with more than one element):
    # check for frame indices
    if ("time" in X_ts.dims and X_ts.sizes["time"] > 1) or (
        "reltime" in X_ts.dims and X_ts.sizes["reltime"] > 1
    ):

        # If time_range is not provided, default to using the range in X_ts
        if time_range is None:
            start_time = float(X_ts.time.values[0])
            end_time = float(X_ts.time.values[-1])
            step_time = (end_time - start_time) / max((X_ts.sizes["time"] - 1), 1)
        else:
            # Convert each element from the time_range tuple to seconds
            start_time = time_range[0].to('s').magnitude
            end_time = time_range[1].to('s').magnitude
            step_time = time_range[2].to('s').magnitude

        # Create an array of time points to iterate over
        time_points = np.arange(start_time, end_time + step_time, step_time)
        # Select the subset of data within the given time range
        X_subset = X_ts.sel(time=slice(start_time, end_time))

        # Initialize using the first time point
        # (using nearest in case of slight mismatches)
        X_frame = X_subset.sel(time=time_points[0], method="nearest")

        p0 = None
        subplots = {}
        labels = {}
        # Create all subviews
        for view, iax in views_positions.items():
            ts_title = title_str if view == 'scale_bar' else None
            p0, surf, lab = image_recon(
                X_frame, head, cmap=cmap, clim=clim, view_type=view_type,
                view_position=view, p0=p0, title_str=ts_title, off_screen=True,
                plotshape=subplot_shape, iax=iax, show_scalar_bar=False,
                wdw_size=wdw_size
            )
            subplots[view] = surf
            labels[view] = lab
            # add labeled points if they were handed in
            if geo3d_plot is not None:
                plot_labeled_points(p0, geo3d_plot)

        if SAVE and filename:
            # Open GIF output with desired fps; filename will have a .gif extension
            p0.open_gif(filename + '.gif', fps=fps)
        else:
            assert filename is None, (
                "Filename must be provided to generate and save GIF."
            )

        # Iterate over the time points
        for current_time in time_points:
            # Select the frame closest to the current time point
            X_frame = X_subset.sel(time=current_time, method="nearest")
            if view_type in ['hbo_brain', 'hbr_brain']:
                new_data = (
                    X_frame.sel(chromo="HbO").where(X_ts.is_brain, drop=True)
                    if view_type == "hbo_brain"
                    else X_frame.sel(chromo="HbR").where(X_ts.is_brain, drop=True)
                )
            elif view_type in ['hbo_scalp', 'hbr_scalp']:
                new_data = (
                    X_frame.sel(chromo="HbO").where(~X_ts.is_brain, drop=True)
                    if view_type == "hbo_scalp"
                    else X_frame.sel(chromo="HbR").where(~X_ts.is_brain, drop=True)
                )
            else:
                new_data = None

            for view, surf in subplots.items():
                surf['brain'] = new_data

            # Update the scalar bar text (for the central 'scale_bar' view)
            if 'scale_bar' in labels:
                labels["scale_bar"].set_text(
                    "upper_left", f"Time = {float(current_time):0.1f} sec"
                )
            p0.write_frame()

        p0.close()  # This finalizes and writes the GIF file.

    # Static image: no time dimension or only one time step available
    else:
        p0 = None
        subplots = {}
        labels = {}
        for view, iax in views_positions.items():
            # For the central view (scale_bar) we pass the title_str
            ts_title = title_str if view == 'scale_bar' else None
            p0, surf, lab = image_recon(
                X_ts, head, cmap=cmap, clim=clim, view_type=view_type,
                view_position=view, p0=p0, title_str=ts_title, off_screen=False,
                plotshape=subplot_shape, iax=iax, wdw_size=wdw_size
            )
            subplots[view] = surf
            labels[view] = lab
            # add labeled points if they were handed in
            if geo3d_plot is not None:
                plot_labeled_points(p0, geo3d_plot)

        if SAVE and filename:
            p0.screenshot(filename + '.png')
        else:
            p0.show()


def segmented_cmap(
    name : str,
    vmin: float,
    vmax: float,
    segments: list[tuple[float, ColorType]],
    over : None | ColorType = None,
    under : None | ColorType = None,
    bad : None | ColorType = None,
) -> tuple[Normalize, LinearSegmentedColormap]:
    """Create a linear segmented colormap."""

    norm = Normalize(vmin, vmax)

    segments = [(norm(v), c) for v, c in segments]

    cmap = LinearSegmentedColormap.from_list(name, segments)

    if over is not None:
        cmap.set_over(over)
    if under is not None:
        cmap.set_under(under)
    if bad is not None:
        cmap.set_bad(bad)

    return norm, cmap
