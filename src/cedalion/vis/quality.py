"""Tools for visualizing signal quality metrics and masks."""

import matplotlib.pyplot as p
import numpy as np

import cedalion.vis.colors as colors


def plot_quality_mask(
    mask,
    cb_label: str,
    bool_labels=["TAINTED", "CLEAN"],
    true_is_good=True,
    figsize=(12, 10),
):
    """Plot a boolean quality mask as a colour-coded channel × time heatmap.

    Args:
        mask: Boolean DataArray with ``"channel"`` and ``"time"`` dimensions.
        cb_label: Label for the colorbar.
        bool_labels: Two-element list giving the colorbar tick labels for
            ``False`` and ``True`` values respectively (default:
            ``["TAINTED", "CLEAN"]``).
        true_is_good: Passed to :func:`~cedalion.vis.colors.mask_cmap`; if
            ``True`` (default), ``True`` is rendered in blue (good).
        figsize: Matplotlib figure size tuple.
    """
    mask_norm, mask_cmap = colors.mask_cmap(true_is_good)

    # plot the binary heatmap
    f, ax = p.subplots(1, 1, figsize=figsize)

    m = ax.pcolormesh(
        mask.time,
        np.arange(len(mask.channel)),
        mask,
        shading="nearest",
        norm=mask_norm,
        cmap=mask_cmap,
    )
    cb = p.colorbar(m, ax=ax)
    p.tight_layout()
    ax.yaxis.set_ticks(np.arange(len(mask.channel)))
    ax.yaxis.set_ticklabels(mask.channel.values, fontsize=7)
    cb.set_label(cb_label)
    ax.set_xlabel("time / s")
    cb.set_ticks([0.25, 0.75])
    cb.set_ticklabels(bool_labels)

