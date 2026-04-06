"""Interactive UI components for nasion selection.

Provides a PyVista-based interactive picker for manual nasion (Nz)
selection as a fallback when automatic detection fails.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging

import numpy as np
import pyvista as pv

import cedalion.dataclasses as cdc

logger = logging.getLogger("cedalion")


def pick_nasion(surface: cdc.TrimeshSurface) -> np.ndarray | None:
    """Open an interactive window for the user to click the nasion point.

    Displays the textured mesh and lets the user click on the nasion (Nz).
    The clicked point is snapped to the nearest mesh vertex.

    Args:
        surface: TrimeshSurface from photogrammetry scan.

    Returns:
        Nasion position as numpy array of shape (3,) in mm, or None if
        the user closes the window without clicking.
    """
    from scipy.spatial import KDTree

    picked_point = [None]  # mutable container for closure

    def _on_pick(point):
        if point is not None:
            picked_point[0] = np.array(point)

    plotter = pv.Plotter(notebook=False)

    # Add textured mesh using cedalion's plot_surface for consistency
    import cedalion.plots
    cedalion.plots.plot_surface(plotter, surface, opacity=1.0)

    plotter.enable_surface_point_picking(
        callback=_on_pick,
        left_clicking=True,
        show_point=True,
        point_size=20,
        color="red",
        show_message="Left-click Nz (nasion), then close the window",
    )

    plotter.add_text(
        "Left-click NASION (bridge of the nose), then close window",
        position="upper_left",
        font_size=14,
    )

    plotter.show()

    if picked_point[0] is None:
        return None

    # Snap to nearest mesh vertex
    vertices = surface.mesh.vertices
    tree = KDTree(vertices)
    _, idx = tree.query(picked_point[0])
    nasion = vertices[idx].copy()
    logger.debug(f"Manual nasion selected: {nasion}")
    return nasion
