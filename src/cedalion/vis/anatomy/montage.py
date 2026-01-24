import cedalion.typing as cdt
from cedalion.dataclasses.geometry import PointType
import matplotlib.pyplot as p


def plot_montage3D(
    amp: cdt.NDTimeSeries, 
    geo3d: cdt.LabeledPoints, 
    landmarks: list[str] | None = None
):
    """Plots a 3D visualization of a montage.

    Args:
        amp: Time series data array.
        geo3d: Landmark coordinates.
        landmarks: Landmarks to highlight in the plot. Can be:
            - None (default): Shows canonical registration landmarks (Nz, Iz, LPA, RPA, Cz)
            - list of str: Shows specific landmarks (only if they exist in geo3d)
            - []: Empty list shows no landmarks
    """
    geo3d = geo3d.pint.dequantify()

    f = p.figure()
    ax = f.add_subplot(projection="3d")
    colors = ["r", "b", "gray"]
    sizes = [20, 20, 2]
    for i, (point_type, x) in enumerate(geo3d.groupby("type")):
        if len(x) > 0:
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors[i], s=sizes[i])

    # Draw lines connecting sources to detectors for each channel
    for i in range(amp.sizes["channel"]):
        src = geo3d.loc[amp.source[i], :]
        det = geo3d.loc[amp.detector[i], :]
        ax.plot([src[0], det[0]], [src[1], det[1]], [src[2], det[2]], c="k")

    # Determine which landmarks to highlight
    if landmarks is None:
        # Default: show canonical registration landmarks
        canonical_landmarks = ["Nz", "Iz", "LPA", "RPA", "Cz"]
        landmarks_to_plot = [
            label for label in canonical_landmarks 
            if label in geo3d.label.values
        ]
    else:
        # Show specified landmarks (filter non-existent ones)
        landmarks_to_plot = [
            label for label in landmarks 
            if label in geo3d.label.values
        ]
    
    landmark_colors = ["y", "m", "c", "orange", "lime", "pink", "brown", "purple"]
    for idx, label in enumerate(landmarks_to_plot):
        color = landmark_colors[idx % len(landmark_colors)]
        ax.scatter(
            geo3d.loc[label, 0], 
            geo3d.loc[label, 1], 
            geo3d.loc[label, 2], 
            c=color, 
            s=50,
            label=label
        )
    
    if landmarks_to_plot:
        ax.legend(bbox_to_anchor=(0, 0.5), loc='center right')

    ax.view_init(elev=30, azim=145)
    p.tight_layout()
