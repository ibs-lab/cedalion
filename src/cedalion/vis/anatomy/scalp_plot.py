"""Static and animated 2D plots of metrics projected on the scalp."""

import matplotlib.colors
import matplotlib.pyplot as p
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike
from PIL import Image

import cedalion.geometry.registration as registration
import cedalion.nirs
import cedalion.typing as cdt
from cedalion import Quantity


def scalp_plot(
    ts: cdt.NDTimeSeries,
    geo3d: cdt.LabeledPoints,
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
    draw_arcs : bool = False,
):
    """Creates a 2D plot of the head with channels coloured according to a given metric.

    Args:
        ts: a NDTimeSeries to provide channel definitions
        geo3d: a LabeledPoints to provide the probe geometry
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
        draw_arcs: if true, draw channel lines as arcs instead of straight lines to
            avoid overlap.

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

    if isinstance(vmin, Quantity):
        vmin = vmin.magnitude
    if isinstance(vmax, Quantity):
        vmax = vmax.magnitude

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

    smallest_channel_dist = channel_dists.min().item()
    channel_dist_range = channel_dists.max().item() - smallest_channel_dist

    for ch,src,det,dist in zip(channel, source, detector, channel_dists):
        s = geo2d.loc[src]
        d = geo2d.loc[det]
        dist = dist.item()

        if (min_dist is not None) and (dist < min_dist):
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

        # color
        c = cmap(normed_v)

        # opacity
        alpha = 1.0
        if (min_metric is not None) and (v < min_metric):
            alpha = 0.4

        # zorder
        if zorder is None:
            zorder_line = 0
        elif zorder == "ascending":
            zorder_line = normed_v
        elif zorder == "descending":
            zorder_line = 1 - normed_v
        else:
            raise ValueError(f"unexpected value '{zorder}' for zorder.")

        # curvature
        if draw_arcs:
            rad = ((dist - smallest_channel_dist) / channel_dist_range).magnitude * 0.15
        else:
            rad = 0.

        arrow = FancyArrowPatch(
            posA=s.values,
            posB=d.values,
            connectionstyle=f"arc3,rad={rad:.3f}",
            arrowstyle="-",
            color=c,
            linewidth=channel_lw,
            alpha=alpha,
            zorder=zorder_line,
        )
        ax.add_patch(arrow)

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


def scalp_plot_gif(
        data_ts: cdt.NDTimeSeries,
        geo3d: cdt.LabeledPoints,
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
        geo3d : 3D geometry defining optode locations for projecting onto the scalp
            surface.
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

