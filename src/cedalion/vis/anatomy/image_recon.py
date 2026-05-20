"""Functions to plot image reconstruction results."""

import matplotlib.colors
import matplotlib.pyplot as p
import numpy as np
import pyvista as pv
from matplotlib.colors import ListedColormap

import cedalion.dataclasses as cdc
import cedalion.dot
import cedalion.typing as cdt

from ..blocks import plot_labeled_points

########################################################################################

def image_recon(
    X: cdt.NDTimeSeries,
    head: "cedalion.dot.TwoSurfaceHeadModel",
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

    X = X.pint.dequantify()

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


########################################################################################


def image_recon_view(
    X_ts: cdt.NDTimeSeries,
    head: "cedalion.dot.TwoSurfaceHeadModel",
    cmap: str | matplotlib.colors.Colormap = 'seismic',
    clim = None,
    view_type: str ='hbo_brain',
    view_position: str ='superior',
    title_str: str = None,
    filename: str =None,
    SAVE: bool = False,
    time_range: tuple = None,
    fps: int = 6,
    geo3d_plot: cdt.LabeledPoints | None = None,
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
        geo3d_plot: cdt.LabeledPoints, optional
            For plotting labeled points (e.g. optodes) on the mesh.
        wdw_size: tuple, default (1024, 768)
            The window size for the plotter (the plot resolution)

    Returns: Nothing

    Initial Contributors:
    - David Boas | dboas@bu.edu | 2025
    - Laura Carlton | lcarlton@bu.edu | 2025
    - Alexander von Lühmann | vonluehmann@tu-berlin.de | 2025
    """

    X_ts = X_ts.pint.dequantify()

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


########################################################################################


def image_recon_multi_view(
    X_ts: cdt.NDTimeSeries,
    head: "cedalion.dot.TwoSurfaceHeadModel",
    cmap: str | matplotlib.colors.Colormap = 'seismic',
    clim = None,
    view_type: str ='hbo_brain',
    title_str: str = None,
    filename: str =None,
    SAVE: bool = True,
    time_range: tuple = None,
    fps: int = 6,
    geo3d_plot: cdt.LabeledPoints | None = None,
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
        geo3d_plot: cdt.LabeledPoints, optional
            For plotting labeled points (e.g. optodes) on the mesh.
        wdw_size: tuple, default (1024, 768)
            The window size for the plotter (the plot resolution)

    Returns: Nothing

    Initial Contributors:
    - David Boas | dboas@bu.edu | 2025
    - Laura Carlton | lcarlton@bu.edu | 2025
    - Alexander von Lühmann | vonluehmann@tu-berlin.de | 2025
    """

    X_ts = X_ts.pint.dequantify()

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


