from cedalion.dataclasses import PointType
import numpy as np
import pyvista as pv
import matplotlib.colors
from matplotlib.typing import ColorType
import cedalion.typing as cdt
from numpy.typing import ArrayLike
import cedalion.dataclasses as cdc
import xarray as xr
import sys
import matplotlib.pyplot as p



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

