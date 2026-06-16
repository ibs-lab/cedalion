import sys

import matplotlib.colors
import matplotlib.pyplot as p
import numpy as np
import pyvista as pv
import xarray as xr
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.vis.blocks as vbx
from cedalion.dataclasses import PointType


def plot_brain_and_scalp(
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


def plot_brain_in_axes(
    ts: cdt.NDTimeSeries,
    geo3d: cdt.LabeledPoints,
    metric: xr.DataArray | ArrayLike,
    brain_surface: cdc.TrimeshSurface,
    ax : matplotlib.axes.Axes,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | matplotlib.colors.Colormap = "RdBu_r",
    bad_color: ColorType = [0.7, 0.7, 0.7],
    cb_label: str = "",
    camera_pos: ArrayLike | str | None = None,
    **kwargs
):
    """Using pyvista render a brain, colored by a metric, and display it in MPL axes."""

    metric = metric.pint.dequantify()
    geo3d = geo3d.pint.dequantify()

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

    if "smooth_shading" not in kwargs:
        kwargs["smooth_shading"] = True
    if "split_sharp_edges" not in kwargs:
        kwargs["split_sharp_edges"] = True
    if "feature_angle" not in kwargs:
        kwargs["feature_angle"] = 90


    plt = pv.Plotter(off_screen=True)

    plt.add_mesh(
        brain_surface,
        scalars=metric,
        cmap=cmap,
        clim=(vmin, vmax),
        scalar_bar_args={"title": cb_label},
        **kwargs
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


def camera_for_view(center, view, distance=350):
    """Return camera parameters for a named orthogonal brain view.

    Args:
        center: 3-element array-like with the focal point coordinates (e.g. brain
            centroid) in the same units as ``distance``.
        view: One of ``"superior"``, ``"left"``, ``"right"``, ``"anterior"``,
            ``"posterior"``.
        distance: Distance from ``center`` to the camera position along the view
            axis. Defaults to 350.

    Returns:
        tuple: ``(position, focal_point, up)`` where each element is a numpy
        array suitable for assignment to ``pyvista.Camera`` attributes.
    """
    cameras = {
        "superior": ([0, 0, distance], [0, 1, 0]),
        "left": ([-distance, 0, 0], [0, 0, 1]),
        "right": ([distance, 0, 0], [0, 0, 1]),
        "anterior": ([0, distance, 0], [0, 0, 1]),
        "posterior": ([0, -distance, 0], [0, 0, 1]),
    }
    position_offset, up = cameras[view]
    return center + np.asarray(position_offset), center, up


def plot_brain_views_grid(
    brain_surface, vertex_colors, window_size=(1000, 600), reset_camera=False
):
    """Render the brain surface from five standard views in a grid layout.

    Displays superior, anterior, posterior, left, and right views arranged in a
    2-row PyVista plotter window.

    Args:
        brain_surface: A surface object whose ``.vertices`` attribute is a
            pint-aware xarray with a ``"label"`` dimension.
        vertex_colors: Per-vertex color array passed to ``vbx.plot_surface``.
        window_size: ``(width, height)`` in pixels for the plotter window.
            Defaults to ``(1000, 600)``.
        reset_camera: If ``True``, call ``plt.reset_camera()`` after setting
            each view to fit the surface tightly. Defaults to ``False``.
    """
    brain_center = brain_surface.vertices.pint.dequantify().mean("label").values
    plt = pv.Plotter(
        shape=(2, 6),
        groups=(
            (0, slice(0, 2)),
            (0, slice(2, 4)),
            (0, slice(4, 6)),
            (1, slice(0, 3)),
            (1, slice(3, 6)),
        ),
        window_size=window_size,
    )
    views = ("superior", "anterior", "posterior", "left", "right")
    positions = [(0, 0), (0, 2), (0, 4), (1, 0), (1, 3)]
    for view, subplot in zip(views, positions):
        plt.subplot(*subplot)
        vbx.plot_surface(
            plt,
            brain_surface,
            color=vertex_colors,
        )
        plt.add_text(view, font_size=10)
        plt.camera.position, plt.camera.focal_point, plt.camera.up = camera_for_view(
            brain_center, view
        )
        if reset_camera:
            plt.reset_camera()
    plt.subplot(1, 2)
    plt.add_text("", font_size=10)
    plt.show()


def get_vertex_colors_from_coord(
    brain_surface : cdc.TrimeshSurface,
    label_coord: str,
    color_mapping: dict,
    default_color="w",
    labels: list[str] | None = None,
):
    """Build a per-vertex color list from a named coordinate on the brain surface.

    Each vertex is colored according to the value of ``label_coord`` at that vertex,
    looked up in ``color_mapping``. Vertices whose coordinate value is not present
    in the mapping (or whose label is filtered out) receive ``default_color``.

    Args:
        brain_surface: Surface object whose ``.vertices`` attribute is an xarray
            DataArray with named coordinates.
        label_coord: Name of the coordinate on ``brain_surface.vertices`` whose
            values are used as keys into ``color_mapping``.
        color_mapping: Controls how coordinate values map to colors:
            * ``None`` — generate a deterministic random color per unique label.
            * ``dict`` — map each label to a matplotlib-compatible color spec.
            * Any other single color spec assigns the same color to every label.
        default_color: Matplotlib-compatible color used for vertices whose label
            is absent from the resolved mapping. Defaults to ``"w"`` (white).
        labels: If provided, only labels in this list are kept in the mapping;
            all other vertices fall back to ``default_color``.

    Returns:
        list: One RGB tuple per vertex, in the same order as
        ``brain_surface.vertices``.
    """
    coords = brain_surface.vertices.coords[label_coord].values
    default_color = matplotlib.colors.to_rgb(default_color)

    def normalize_colors(c):
        if (isinstance(c, tuple) or isinstance(c, list)) and all(
            [isinstance(v, int) for v in c]
        ):
            c = [k/255. for k in c]

        return matplotlib.colors.to_rgb(c)

    if color_mapping is None:
        # generate random colors
        rng = np.random.default_rng(43)
        color_mapping = {
            k: rng.uniform(0.3, 1.0, size=3).tolist() for k in sorted(set(coords))
        }
    elif isinstance(color_mapping, dict):
        color_mapping = {
            k: normalize_colors(v) for k, v in color_mapping.items()
        }
    elif not isinstance(color_mapping, dict):
        # all coord values get the same color
        c = matplotlib.colors.to_rgb(color_mapping)
        color_mapping = {k: c for k in coords}
    elif callable(color_mapping):
        # support any kind of mapping
        raise not NotImplementedError()
    else:
        raise ValueError("could not interprete color_mapping")

    if labels is not None:
        color_mapping = {k: v for k, v in color_mapping.items() if k in labels}

    vertex_colors = [color_mapping.get(pp, default_color) for pp in coords]

    return vertex_colors
