"""Tools for visualizing data on brain and scalp surface representations."""

from .brain_and_scalp import plot_brain_in_axes, plot_brain_and_scalp
from .image_recon import image_recon, image_recon_multi_view, image_recon_view
from .montage import plot_montage3D
from .optode_selector import OptodeSelector
from .scalp_plot import scalp_plot, scalp_plot_gif

import cedalion.vis.anatomy.sensitivity_matrix

__all__ = [
    "plot_brain_in_axes",
    "plot_brain_and_scalp",
    "image_recon",
    "image_recon_multi_view",
    "image_recon_view",
    "plot_montage3D",
    "OptodeSelector",
    "scalp_plot",
    "scalp_plot_gif",
]
