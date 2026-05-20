"""Color and colormap definitions."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.typing import ColorType

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


def segmented_cmap(
    name: str,
    vmin: float,
    vmax: float,
    segments: list[tuple[float, ColorType]],
    over: None | ColorType = None,
    under: None | ColorType = None,
    bad: None | ColorType = None,
) -> tuple[Normalize, LinearSegmentedColormap]:
    """Create a linear segmented colormap from (value, color) breakpoints.

    Args:
        name: Name for the colormap (used by matplotlib internally).
        vmin: Data value corresponding to the bottom of the colormap.
        vmax: Data value corresponding to the top of the colormap.
        segments: List of ``(value, color)`` pairs defining the colour
            breakpoints.  Values are normalised by ``vmin``/``vmax`` before
            being passed to :class:`~matplotlib.colors.LinearSegmentedColormap`.
        over: Color for out-of-range values above ``vmax`` (optional).
        under: Color for out-of-range values below ``vmin`` (optional).
        bad: Color for masked/NaN values (optional).

    Returns:
        Tuple ``(norm, cmap)`` — a :class:`~matplotlib.colors.Normalize` instance
        and the constructed :class:`~matplotlib.colors.LinearSegmentedColormap`.
    """

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


def p_values_cmap() -> tuple[Normalize, LinearSegmentedColormap]:
    """A colormap for log10(p-values).

    Gray for p>0.05 and with different colors for common thresholds.
    """

    norm, cmap = segmented_cmap(
        "logpvalue",
        vmin=-4,
        vmax=0,
        segments=[
            (np.log10(0.0001), "blue"),
            (np.log10(0.001), "red"),
            (np.log10(0.005), "orange"),
            (np.log10(0.01), "yellow"),
            (np.log10(0.05), "green"),
            (np.log10(0.05), "lightgray"),
            (np.log10(1), "darkgray"),
        ],
        under="magenta",
    )
    return norm, cmap


def threshold_cmap(
    name,
    vmin,
    vmax,
    threshold,
    higher_is_better=True,
    colors=["#000000", "#DC3220", "#5D3A9B", "#0C7BDC"],
) -> tuple[Normalize, LinearSegmentedColormap]:
    """Create a red-and-blue colormap with a sharp break at a quality threshold.

    Args:
        name: Colormap name.
        vmin: Minimum data value.
        vmax: Maximum data value.
        threshold: Value at which the colour transitions sharply.
        higher_is_better: If ``True`` (default), colours below the threshold
            are "bad" (warm) and colours above are "good" (cool).  Reversed
            when ``False``.
        colors: Four colours ``[vmin, threshold-, threshold+, vmax]``.

    Returns:
        Tuple ``(norm, cmap)`` — a :class:`~matplotlib.colors.Normalize` and the
        constructed :class:`~matplotlib.colors.LinearSegmentedColormap`.
    """

    values = [vmin, threshold, threshold, vmax]

    if not higher_is_better:
        colors = colors[::-1]

    norm, cmap = segmented_cmap(
        name,
        vmin=vmin,
        vmax=vmax,
        segments=zip(values, colors),
        bad="magenta",
        over="magenta",
        under="magenta",
    )

    return norm, cmap


def mask_cmap(
    true_is_good=True, colors=["#DC3220", "#DC3220", "#0C7BDC", "#0C7BDC"]
) -> tuple[Normalize, LinearSegmentedColormap]:
    """Create a binary red/blue colormap for boolean quality masks.

    Args:
        true_is_good: If ``True`` (default), ``True`` values are shown in blue
            (good) and ``False`` values in red (bad).  Reversed when ``False``.
        colors: Four colours defining the two-level step (``[0, 0.5, 0.5, 1]``
            breakpoints).

    Returns:
        Tuple ``(norm, cmap)`` — a :class:`~matplotlib.colors.Normalize` for
        ``[0, 1]`` and the constructed
        :class:`~matplotlib.colors.LinearSegmentedColormap`.
    """

    if not true_is_good:
        colors = colors[::-1]

    norm, cmap = segmented_cmap(
        "mask_cmap",
        vmin=0,
        vmax=1.0,
        segments=zip([0, 0.5, 0.5, 1], colors),
        bad="magenta",
        over="magenta",
        under="magenta",
    )

    return norm, cmap
