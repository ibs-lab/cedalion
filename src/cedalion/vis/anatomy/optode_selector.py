import pyvista as pv
from cedalion.dataclasses import PointType
import cedalion.dataclasses as cdc
import numpy as np
import xarray as xr


class OptodeSelector:
    """A class for visualizing point clouds with interactive features in PyVista.

    This class provides functionality to visualize and interact with labeled point
    clouds using a PyVista plotter. It allows points to be dynamically added or removed
    by picking them directly from the plot interface.

    Attributes:
        surface (cdc.Surface): The surface of a head for normals.
        points (cdt.LabeledPoints): The point cloud data containing point
            coordinates.
        normals (xr.DataArray): Normal vectors to the points.
        plotter (pv.Plotter): A PyVista plotter instance for rendering the point cloud.
        labels (list of str, optional): Labels corresponding to the points, displayed
            if provided.
        actors (list): List of PyVista actor objects representing the points in the
            visualization.
        color (str or tuple, optional): Default color for points if not specified by
            point type.

    Methods:
        plot(): Renders the point cloud using the current settings.
        on_pick(picked_point): Callback function for picking points in the visualization
        update_visualization(): Clears the existing plot and re-renders the point cloud.
        enable_picking(): Enables interactive picking of points on the plot.

    Initial Contributors:
        - Masha Iudina | mashayudi@gmail.com | 2024
    """
    def __init__(self, surface, points, normals=None, plotter=None, labels = None):
        self.points = points
        self.normals = normals
        self.surface = surface
        self.plotter = plotter if plotter else pv.Plotter()
        self.labels = labels
        self.actors = []
        self.color = None

        self.cog = surface.mesh.vertices.mean(axis=0)

    def plot(self):
        plotter = self.plotter
        points = self.points.pint.dequantify()
        color = 'r'
        # FIXME make these configurable
        default_point_colors = {
            PointType.UNKNOWN: "gray",
            PointType.SOURCE: "r",
            PointType.DETECTOR: "b",
            PointType.LANDMARK: "g",
            PointType.ELECTRODE: "pink",
        }
        default_point_sizes = {
            PointType.UNKNOWN: 2,
            PointType.SOURCE: 3,
            PointType.DETECTOR: 3,
            PointType.LANDMARK: 2,
            PointType.ELECTRODE: 3,
        }

        # points = points.pint.to("mm").pint.dequantify()  # FIXME unit handling
        # points = points.pint.dequantify()  # FIXME unit handling
        for type, x in points.groupby("type"):
            for i_point in range(len(x)):

                s = pv.Sphere(radius=default_point_sizes[type], center=x[i_point])
                if color is None:
                    sphere_actor = plotter.add_mesh(
                        s, color=default_point_colors[type], smooth_shading=True
                    )
                else:
                    sphere_actor = plotter.add_mesh(s, color=color, smooth_shading=True)
                self.actors.append(sphere_actor)
                if self.labels is not None:
                    plotter.add_point_labels(
                        x[i_point].values, [str(self.labels[i_point])]
                    )


    def on_pick(self, picked_point):
        plotter = self.plotter
        points = self.points.pint.dequantify()
        # Define how close points have to be to consider them "super close"
        threshold_distance = 5
        new_point = np.array(picked_point)

        # Check if new point is super close to any existing sphere
        for i, existing_point in enumerate(points.values):
            if np.linalg.norm(new_point - existing_point) < threshold_distance:
                idx_to_remove = i
                indexes = np.arange(len(self.points.label))
                selected_indexes = np.delete(indexes, idx_to_remove)

                self.points = self.points.isel(label=selected_indexes)
                if self.normals is not None:
                    self.normals = self.normals.isel(label=selected_indexes)
                self.plotter.remove_actor(self.actors[idx_to_remove])
                del self.actors[idx_to_remove]

                return  # Stop the function after removing the sphere

        existing_labels = self.points.coords['label'].values
        # Generate a new unique label
        new_label_number = (
            max([int(label.split("-")[-1]) for label in existing_labels]) + 1
        )
        new_label = f'O-{new_label_number}'
        new_group = self.points.coords['group'].values[0]
        new_type = cdc.PointType.LANDMARK if new_group == 'O' else cdc.PointType.UNKNOWN

        # Create the new entry DataArray
        new_center_coords = new_point

        s = pv.Sphere(radius=2, center=new_point)
        sphere_actor = plotter.add_mesh(s, color='r', smooth_shading=True)
        self.actors.append(sphere_actor)

        new_normal = self.find_surface_normal(new_point)
        self.points = self.points.points.add(
            new_label, new_center_coords, new_type, new_group
        )
        self.normals = self.update_normals(new_normal, new_label)

    def update_visualization(self):
        # Clear existing plot and re-plot with the updated self.points
        self.plotter.clear()
        self.plot()

    def enable_picking(self):
        self.plotter.enable_surface_point_picking(
            callback=self.on_pick,
            show_message="Right click to place or remove optode",
            show_point=False,
            tolerance=0.005,
        )

    def find_surface_normal(self, picked_point, radius=6):
        def pca(vertices: np.ndarray):
            eigenvalues, eigenvecs = np.linalg.eigh(np.cov(vertices.T))

            # sort by increasing eigenvalue
            indices = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[indices]
            eigenvecs = eigenvecs[:, indices]

            return eigenvalues, eigenvecs
        # Calculate distances from picked point to all vertices in the mesh
        distances = np.linalg.norm(self.surface.mesh.vertices - picked_point, axis=1)

        # Select vertices within the specified radius
        close_vertices = self.surface.mesh.vertices[distances < radius]

        # calculate normal from eigenvector
        eigenvalues, eigenvecs = pca(close_vertices)
        normal_vector = eigenvecs[:, 0]

        # Verify the direction of the normal
        if np.dot(normal_vector, picked_point - self.cog) < 0:
            normal_vector = -normal_vector  # Ensure the normal points outward
        return normal_vector

    def update_normals(self, normal_at_picked_point, label):
        new_normals = xr.DataArray(
            np.vstack([normal_at_picked_point]),
            dims=["label", self.surface.crs],
            coords={
                "label": ("label", [label]),
                "group": ("label", ["O"]),
            },
        ).pint.quantify("1")

        return xr.concat((self.normals, new_normals), dim="label")
