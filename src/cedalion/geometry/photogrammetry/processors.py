"""Vertex classifiers."""

import colorsys
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from cv2 import minEnclosingCircle
import matplotlib.pyplot as p
import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import units, Quantity

logger = logging.getLogger("cedalion")


class ScanProcessor(ABC):
    """Base class for all processors of photogrammetric scans."""

    @abstractmethod
    def process(self, surface: cdc.TrimeshSurface) -> xr.DataArray:
        raise NotImplementedError()


def pca(vertices: np.ndarray):
    eigenvalues, eigenvecs = np.linalg.eigh(np.cov(vertices.T))

    # sort by increasing eigenvalue
    indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[indices]
    eigenvecs = eigenvecs[:, indices]

    return eigenvalues, eigenvecs


@dataclass
class ColoredStickerProcessorDetails:
    cluster_coords: list[np.ndarray]
    cluster_circles: list[tuple[float, float, float]]
    cluster_colors: list[np.ndarray]
    vertex_colors: np.ndarray
    vertex_hue: np.ndarray
    vertex_value: np.ndarray

    cfg_colors: dict[str, float]
    vertex_sat: np.ndarray

    def plot_cluster_circles(self):
        """Plot for each cluster the vertex coordinates in the sticker plan."""
        nclusters = len(self.cluster_coords)
        figure_layout = (5, 5)
        clusters_per_figure = figure_layout[0] * figure_layout[1]
        nfigures = int(np.ceil(nclusters / clusters_per_figure))

        for i_plot in range(nfigures):
            start = i_plot * clusters_per_figure

            f, ax = p.subplots(
                figure_layout[0],
                figure_layout[1],
                figsize=(3 * figure_layout[0], 3 * figure_layout[1]),
            )
            ax = ax.flatten()
            for i_ax in range(clusters_per_figure):
                i_cluster = start + i_ax
                if i_cluster >= nclusters:
                    break

                coords = self.cluster_coords[i_cluster]
                a, b, r = self.cluster_circles[i_cluster]
                colors = self.cluster_colors[i_cluster]

                std_x = np.std(coords[:, 1])
                std_y = np.std(coords[:, 2])

                ax[i_ax].scatter(coords[:, 1], coords[:, 2], c=colors)
                circle = p.Circle((a, b), r, color="r", fill=False)
                ax[i_ax].add_patch(circle)
                ax[i_ax].set_xlim(-12, 12)
                ax[i_ax].set_ylim(-12, 12)
                ax[i_ax].set_title(
                    rf"$\sigma_x$ {std_x:.2f} $\sigma_y$ {std_y:.2f} $rad$ {r:.2f}"
                )

    # FIXME obsolete?
    def plot_vertex_colors1(self):
        f, ax = p.subplots(1, 1, figsize=(20, 10))
        ax.scatter(
            self.vertex_hue,
            self.vertex_value,
            s=4,
            c=self.vertex_colors.astype(float) / 255,
        )

        for group_label, (h_min, h_max, v_min, v_max) in self.cfg_colors.items():
            rect = p.Rectangle(
                (h_min, v_min), h_max - h_min, v_max - v_min, color="k", fill=False
            )
            ax.add_patch(rect)

    def plot_vertex_colors(self):
        f = p.figure(figsize=(20, 20))
        ax1 = f.add_subplot(211)  # First subplot with 3D projection
        ax2 = f.add_subplot(212, projection="3d")  # Second subplot with 3D projection

        # Plot for the first subplot
        ax1.scatter(
            self.vertex_hue,
            self.vertex_value,
            s=4,
            c=self.vertex_colors.astype(float) / 255,
        )

        ax1.set_xlabel("Vertex Hue")
        ax1.set_ylabel("Vertex Value")

        for group_label, (h_min, h_max, v_min, v_max) in self.cfg_colors.items():
            rect = p.Rectangle(
                (h_min, v_min), h_max - h_min, v_max - v_min, color="k", fill=False
            )
            ax1.add_patch(rect)

        # Plot for the second subplot (3D scatter plot)
        ax2.scatter(
            self.vertex_hue,
            self.vertex_value,
            self.vertex_sat,
            s=4,
            c=self.vertex_colors.astype(float) / 255,
        )
        ax2.set_xlabel("Vertex Hue")
        ax2.set_ylabel("Vertex Value")
        ax2.set_zlabel("Vertex Saturation")

        p.show()


class ColoredStickerProcessor(ScanProcessor):
    """Searches for colored circular patches."""

    def __init__(
        self,
        colors: dict[str, tuple[float, float, float, float]],
        # hue_threshold: float = 0.05,
        # value_threshold: float = 150 / 255.0,
        sticker_radius: Quantity = 6.5 * units.mm,
        min_nvertices: int = 50,
    ):
        """Initiliaze the classifier by specifying colors and classnames.

        Args:
            colors: maps class name to hue value [0,1.0]
            #hue_threshold: absolute hue difference to still be classified
            #value_threshold: minimum value to still be classified
            sticker_radius: the radius of the colored stickers
            min_nvertices: minimum number of vertices during clustering

        """
        self.colors = colors
        # self.hue_threshold = hue_threshold
        # self.value_threshold = value_threshold
        self.sticker_radius = sticker_radius
        self.min_nvertices = min_nvertices

    def process(
        self, surface: cdc.TrimeshSurface, details: bool = False
    ) -> (
        tuple[cdt.LabeledPointCloud, xr.DataArray]
        | tuple[cdt.LabeledPointCloud, xr.DataArray, ColoredStickerProcessorDetails]
    ):
        """Process a scanned surface.

        Args:
            surface: the textured scan
            details: return additional information

        Returns:
            extracted sticker positions
            surface normal vectors
            aux. detail object if detail == True
        """
        assert surface.units == units.mm  # FIXME Einstar yields mm. Allow other units.

        vertex_colors = surface.mesh.visual.to_color().vertex_colors

        rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)

        # vertex colors as float HSV values. h,s in [0,1.] v in [0,255]
        h, s, v = rgb_to_hsv(
            vertex_colors[:, 0],
            vertex_colors[:, 1],
            vertex_colors[:, 2],
        )
        v = v / 255.0

        head_cog = np.mean(surface.mesh.vertices, axis=0)
        radius_mm = float(self.sticker_radius.to("mm").magnitude)

        sticker_centers = []
        sticker_normals = []
        groups = []
        labels = []
        types = []

        detail_coords = []
        detail_circles = []
        detail_colors = []

        for group_name, (h_min, h_max, v_min, v_max) in self.colors.items():
            group_counter = 1
            group_mask = (v_min <= v) & (v <= v_max)
            group_mask &= (h_min <= h) & (h <= h_max)
            group_mask &= 0.6 <= s

            class_vertices = surface.mesh.vertices[group_mask]  # shape=(nvertices, 3)

            print(surface.mesh.vertices)
            print(class_vertices)
            print(group_name, (h_min, h_max, v_min, v_max))

            cluster_labels = DBSCAN(eps=0.2 * radius_mm).fit_predict(class_vertices)

            for label in np.unique(cluster_labels):
                cluster_mask = cluster_labels == label
                cluster_vertices = class_vertices[cluster_mask]
                cluster_colors = vertex_colors[group_mask][cluster_mask]
                nverts = len(cluster_vertices)

                if label == -1:  # vertices assigned to no cluster
                    logging.debug(f"{nverts} vertices belong to no cluster.")
                    continue

                if len(cluster_vertices) < self.min_nvertices:
                    logging.debug(
                        f"skipping cluster {label} because of too few vertices "
                        f"({nverts} < {self.min_nvertices})."
                    )
                    continue

                logging.debug(f"{group_name} - {label}: {nverts} vertices")

                # tentative center. the stickers are not always uniformly sampled
                # and the cog is not necessarily at the sticker center.
                tentative_center = cluster_vertices.mean(axis=0)

                # select vertices in the proximity of the tentative center. At the edge
                # of the optode 'cluster_vertices' may leave the plane but in the
                # proximity of the center they should be flat
                rel_verts = cluster_vertices - tentative_center
                rel_dists = np.linalg.norm(rel_verts, axis=1)
                proximity_mask = rel_dists < 0.66 * radius_mm

                hopefully_flat_verts = cluster_vertices[proximity_mask]

                if len(hopefully_flat_verts) == 0:
                    logging.info(
                        f"skipping cluster {label} because there are no "
                        f"vertices close to the tentative center."
                    )
                    continue

                # the eigenvector corresponding to the smallest eigenvalue denotes the
                # direction of the smallest extent of the vertices. For circular
                # stickers this corresponds to the surface normal. The other two
                # eigenvectors span the plan in which the vertices should lie.
                eigenvalues, eigenvecs = pca(hopefully_flat_verts)

                # FIXME? add a criterion based on the magnitude of eigenvalues[0].
                # too large eigenvalues indicate non-flat vertices.

                normal = eigenvecs[:, 0]

                # calculate coords relative to eigenvector basis
                # coords[:,1] and coords[:;2] are in the tentative plane of the sticker
                coords = np.vstack(
                    [eigenvecs[:, i][None, :] @ (rel_verts).T for i in range(3)]
                ).T

                # since stickers are circular the standard deviation of the relative
                # coordinates in both directions should be similar close to ~68% of the
                # sticker radius. Skip clusters with small extent in either dimension.
                # FIXME this breaks when stickers of very different sizes are used.
                std_x = np.std(coords[:, 1])
                std_y = np.std(coords[:, 2])
                std_threshold = 0.25 * radius_mm
                if (std_x < std_threshold) or (std_y < std_threshold):
                    logger.debug(f"skipping non-circuluar cluster {label}")
                    # continue

                # find the minimum enclosing circle to find the sticker center.
                # cv2 expects integer coordinates with shape
                # (npoints, 1, 2). Upscale float coords by 100 to avoid rounding errors.
                coords *= 100

                # First pass. The center (a,b) should be closer to the real sticker
                # center. It could still be affected by outlier vertices, though.
                (a, b), r = minEnclosingCircle(
                    coords[:, 1:][:, None, :].astype(np.int32)
                )

                # Second pass. Calculate distances to new center and get rid of the N
                # most distant vertices to remove remaining outlier vertices not in the
                # sticker circle.
                skip_to_N = 20
                rel_dists = np.linalg.norm(coords[:, 1:] - [a, b], axis=1)
                indices = np.argsort(rel_dists)[:-skip_to_N]
                coords2 = coords[indices]

                (a, b), r = minEnclosingCircle(
                    coords2[:, 1:][:, None, :].astype(np.int32)
                )

                # scale back coordinates
                a /= 100
                b /= 100
                coords /= 100
                r /= 100

                center = tentative_center + a * eigenvecs[:, 1] + b * eigenvecs[:, 2]

                detail_coords.append(coords)
                detail_circles.append((a, b, r))
                detail_colors.append(cluster_colors / 255.0)

                # FIXME? add a check that cluster_vertices fill the sphere
                # FIXME? select all vertices in found cylinder volume to catch vertices
                # missed by the color criterion.

                # make sure that normal points away from the head's center
                if np.dot(center - head_cog, normal) < 0:
                    normal *= -1

                sticker_centers.append(center)
                sticker_normals.append(normal)
                groups.append(group_name)
                labels.append(f"{group_name}-{group_counter:02d}")
                types.append(
                    cdc.PointType.LANDMARK
                    if group_name == "L"
                    else cdc.PointType.UNKNOWN
                )
                group_counter += 1

        radii = np.array([circle[2] for circle in detail_circles])
        mean_radius = np.mean(radii)

        # Calculate the absolute differences from the mean radius
        differences_from_mean = np.abs(radii - mean_radius)
        print(differences_from_mean)
        print(mean_radius)
        sorted_indices = np.argsort(differences_from_mean)

        # sorted_indices = sorted_indices[:46]
        # FIXME the following errors lead tp ValueErros
        """ detail_circles = np.array(detail_circles)[sorted_indices]
        detail_coords = np.array(detail_coords)[sorted_indices]
        detail_colors = np.array(detail_colors)[sorted_indices] """
        sticker_centers = np.array(sticker_centers)[sorted_indices]
        sticker_normals = np.array(sticker_normals)[sorted_indices]
        labels = np.array(labels)[sorted_indices]
        groups = np.array(groups)[sorted_indices]
        types = np.array(types)[sorted_indices]

        sticker_centers = xr.DataArray(
            np.vstack(sticker_centers),
            dims=["label", surface.crs],
            coords={
                "label": ("label", labels),
                "type": ("label", types),
                "group": ("label", groups),
            },
        ).pint.quantify("mm")

        sticker_normals = xr.DataArray(
            np.vstack(sticker_normals),
            dims=["label", surface.crs],
            coords={
                "label": ("label", labels),
                "group": ("label", groups),
            },
        ).pint.quantify("1")

        print("surface.crs", surface.crs)

        if details:
            csdetails = ColoredStickerProcessorDetails(
                detail_coords,
                detail_circles,
                detail_colors,
                vertex_colors,
                h,
                v,
                self.colors,
                s,
            )
            return sticker_centers, sticker_normals, csdetails
        else:
            return sticker_centers, sticker_normals


@cdc.validate_schemas
def geo3d_from_scan(
    scalp_coords: cdt.LabeledPointCloud, landmarks: cdt.LabeledPointCloud
):
    """Creates a geo3D with optode and landmark positions from photogrammetric scans.

    Args:
        scalp_coords: 3D coordinates of optodes on the scalp from photogrammetric
            coregistration.
        landmarks: 3D coordinates of landmarks on the scalp from photogrammetric
            coregistration.

    Returns:
        geo3D with optode and landmark positions
    """

    # FIXME: what to do with 10-10 landmarks in input geo3d?

    # merge landmarks and scalp_coords
    geo3d = xr.concat([scalp_coords, landmarks], dim="label")

    # iterate through all points and in type set PointType if PointType.UNKNOWN
    for label in geo3d.label:
        point = geo3d.sel(label=label)
        if point.coords["type"] == cdc.PointType.UNKNOWN:
            # Check if point is a source, detector or landmark by looking up its label.
            # Set the PointType accordingly.
            if (np.char.find(point.coords["label"].values, "S") != -1).any():
                geo3d["type"].loc[{"label": label}] = cdc.PointType.SOURCE
            elif (np.char.find(point.coords["label"].values, "D") != -1).any():
                geo3d["type"].loc[{"label": label}] = cdc.PointType.DETECTOR
            elif (np.char.find(point.coords["label"].values, "L") != -1).any():
                geo3d["type"].loc[{"label": label}] = cdc.PointType.LANDMARK

    # add mm units
    geo3d.attrs.update({"units": "mm"})

    return geo3d
