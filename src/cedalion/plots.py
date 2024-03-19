import matplotlib.pyplot as p
import xarray as xr

from cedalion.dataclasses import PointType
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import pyvista as pv
from vtk.util.numpy_support import numpy_to_vtk
import vtk

import numpy as np
import pint_xarray



def plot_montage3D(amp: xr.DataArray, geo3d: xr.DataArray):
    geo3d = geo3d.pint.dequantify()

    f = p.figure()
    ax = f.add_subplot(projection="3d")
    colors = ["r", "b", "gray"]
    sizes = [20, 20, 2]
    for i, (type, x) in enumerate(geo3d.groupby("type")):
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors[i], s=sizes[i])

    for i in range(amp.sizes["channel"]):
        src = geo3d.loc[amp.source[i], :]
        det = geo3d.loc[amp.detector[i], :]
        ax.plot([src[0], det[0]], [src[1], det[1]], [src[2], det[2]], c="k")

    # if available mark Nasion in yellow
    if "Nz" in geo3d.label:
        ax.scatter(
            geo3d.loc["Nz", 0], geo3d.loc["Nz", 1], geo3d.loc["Nz", 2], c="y", s=25
        )

    ax.view_init(elev=30, azim=145)
    p.tight_layout()


def plot3d(
    brain_mesh, scalp_mesh, geo3d, timeseries, poly_lines=[], brain_scalars=None, labels = None, plotter = None, ppoints = None
):
    pv.set_jupyter_backend("server")
    
    if plotter is None:
        plt = pv.Plotter()
    else:
        plt = plotter

    if brain_mesh:
        pv_brain = pv.wrap(brain_mesh)
        if brain_scalars is None:
            plt.add_mesh(pv_brain, color="w")
        else:
            plt.add_mesh(pv_brain, scalars=brain_scalars)
    if scalp_mesh:
        pv_scalp = pv.wrap(scalp_mesh)
        plt.add_mesh(pv_scalp, color="w", opacity=0.4)

    point_colors = {
        PointType.SOURCE: "r",
        PointType.DETECTOR: "b",
        PointType.LANDMARK: "gray",
    }
    point_sizes = {
        PointType.SOURCE: 3,
        PointType.DETECTOR: 3,
        PointType.LANDMARK: 2,
    }

    if geo3d is not None:
        geo3d = geo3d.pint.to("mm").pint.dequantify()  # FIXME unit handling
        for type, x in geo3d.groupby("type"):
            for i_point in range(len(x)):
                s = pv.Sphere(radius=point_sizes[type], center=x[i_point])
                plt.add_mesh(s, color=point_colors[type])
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
            plt.add_mesh(line, color="k")

    for points in poly_lines:
        lines = pv.MultipleLines(points)
        plt.add_mesh(lines, color="m")
        
    def callback(point):
        """Create a cube and a label at the click point."""
        mesh = pv.Sphere(radius=3, center=point)
        plt.add_mesh(mesh, style='wireframe', color='r')
        plt.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])
    if ppoints is not None:
        plt.enable_surface_point_picking(callback=callback, show_point=False)

    #plt.show()


def plot_surface(
    plotter: pv.Plotter,
    surface: cdc.Surface,
    color=None,
    opacity=1.0,
    ppoints = None,
    **kwargs,
):
    pv.set_jupyter_backend("server") 
    if isinstance(surface, cdc.VTKSurface):
        mesh = surface.mesh
    elif isinstance(surface, cdc.TrimeshSurface):
        mesh = cdc.VTKSurface.from_trimeshsurface(surface).mesh
    else:
        raise ValueError("unsupported mesh")

    scalars = mesh.GetPointData().GetScalars()

    if color is None:
        if scalars is not None:
            if scalars.GetNumberOfComponents() in [3, 4]:
                rgb = True
            else:
                rgb = False
        else:
            rgb = False
            color = "w"
    else:
        rgb = False

    plotter.add_mesh(mesh, color=color, rgb=rgb, opacity=opacity, pickable=True, **kwargs)
    
    actors = []  # Store sphere actors for removal

    def place_or_remove_sphere(point):
        nonlocal ppoints, actors
        threshold_distance = 5  # Define how close points have to be to consider them "super close"
        new_point = np.array(point)
        print(point)

        # Check if new point is super close to any existing sphere
        for i, existing_point in enumerate(ppoints):
            if np.linalg.norm(new_point - existing_point) < threshold_distance:
                # Point is super close to an existing sphere, remove the sphere
                plotter.remove_actor(actors[i])
                del ppoints[i]
                del actors[i]
                return  # Stop the function after removing the sphere
        # If no sphere is super close, add a new sphere
        sphere_actor = plotter.add_mesh(pv.Sphere(radius=3, center=point), color='green')
        #plotter.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])
        actors.append(sphere_actor)
        ppoints.append(list(new_point))
    
    # Function to handle the point placement
    def place_point(picked_point):
        if picked_point is not None:
            point = mesh.points[picked_point]
            # Add a sphere at the picked point
            plotter.add_mesh(pv.Sphere(radius=0.1, center=point), color='red')
            
    def place_point2(point):
        """Create a cube and a label at the click point."""
        mesh = pv.Sphere(radius=3, center=point)
        plotter.add_mesh(mesh, color='green')
        ppoints.append(mesh)
        print(point)
        plotter.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])
        
    def callback(point):
        """Create a cube and a label at the click point."""
        mesh = pv.Sphere(radius=3, center=point)
        plotter.add_mesh(mesh, style='wireframe', color='r')
        plotter.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])
        
    if ppoints is not None:
        plotter.enable_surface_point_picking(callback=place_or_remove_sphere, show_point=False)
    #plotter.show()



def plot_labeled_points(
    plotter: pv.Plotter,
    points: cdt.LabeledPointCloud,
    color=None,
    labels = None,
    ppoints = None,
):
    # FIXME make these configurable
    default_point_colors = {
        PointType.UNKNOWN: "gray",
        PointType.SOURCE: "r",
        PointType.DETECTOR: "b",
        PointType.LANDMARK: "g",
    }
    default_point_sizes = {
        PointType.UNKNOWN: 2,
        PointType.SOURCE: 3,
        PointType.DETECTOR: 3,
        PointType.LANDMARK: 2,
    }
    def on_pick(picked_point):
        nonlocal ppoints
        threshold_distance = 5  # Define how close points have to be to consider them "super close"
        new_point = np.array(picked_point)
        
        # Check if new point is super close to any existing sphere
        for i, existing_point in enumerate(points):
            if np.linalg.norm(new_point - existing_point) < threshold_distance:
                s = pv.Sphere(radius=4, center=existing_point)
                plotter.add_mesh(s, color='r')
                if i not in ppoints:
                    ppoints.append(i)
                return  # Stop the function after removing the sphere

    # points = points.pint.to("mm").pint.dequantify()  # FIXME unit handling
    points = points.pint.dequantify()  # FIXME unit handling
    for type, x in points.groupby("type"):
        for i_point in range(len(x)):
            s = pv.Sphere(radius=default_point_sizes[type], center=x[i_point])
            if color is None:
                plotter.add_mesh(s, color=default_point_colors[type])
            else:
                plotter.add_mesh(s, color=color)
            if labels is not None:
                plotter.add_point_labels(x[i_point].values, [str(labels[i_point])])
                
    if ppoints is not None: 
        plotter.enable_surface_point_picking(callback=on_pick, show_point=False)



def plot_vector_field(
    plotter: pv.Plotter,
    points: cdt.LabeledPointCloud,
    vectors: xr.DataArray,
    ppoints = None
):
    assert len(points) == len(vectors)
    assert all(points.label.values == vectors.label.values)
    assert points.points.crs == vectors.dims[1]
    

    points = points.pint.to("mm").pint.dequantify()
    vectors = vectors.pint.dequantify()

    ugrid = vtk.vtkUnstructuredGrid()

    vpoints = vtk.vtkPoints()
    vpoints.SetData(numpy_to_vtk(points.values))
    ugrid.SetPoints(vpoints)
    ugrid.GetPointData().SetVectors(numpy_to_vtk(vectors.values))
    
    points = ugrid.GetPoints()
    highlight_coords = points.GetPoint(0)

    hedgehog = vtk.vtkHedgeHog()
    hedgehog.SetInputData(ugrid)
    hedgehog.SetScaleFactor(10)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(hedgehog.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor([1.0, 0.0, 0.0])

    plotter.renderer.AddActor(actor)

    
def plot_labeled_points_1(
    plotter: pv.Plotter,
    points: cdt.LabeledPointCloud,
    color=None,
    labels = None,
    ppoints = None,
):
    # FIXME make these configurable
    default_point_colors = {
        PointType.UNKNOWN: "gray",
        PointType.SOURCE: "r",
        PointType.DETECTOR: "b",
        PointType.LANDMARK: "g",
    }
    default_point_sizes = {
        PointType.UNKNOWN: 2,
        PointType.SOURCE: 3,
        PointType.DETECTOR: 3,
        PointType.LANDMARK: 2,
    }
    
    actors = []
    
    def on_pick(picked_point):
        nonlocal ppoints, actors, points
        threshold_distance = 5  # Define how close points have to be to consider them "super close"
        new_point = np.array(picked_point)
        
        # Check if new point is super close to any existing sphere
        for i, existing_point in enumerate(points):
            if np.linalg.norm(new_point - existing_point) < threshold_distance:
                s = pv.Sphere(radius=4, center=existing_point)
                plotter.add_mesh(s, color='r')
                if i not in ppoints:
                    ppoints.append(i)
                    
                idx_to_remove = i
                indexes = np.arange(len(points.label))
                selected_indexes = np.delete(indexes, idx_to_remove)

                points = points.isel(label=selected_indexes)
                display(points)
                return  # Stop the function after removing the sphere

    # points = points.pint.to("mm").pint.dequantify()  # FIXME unit handling
    points = points.pint.dequantify()  # FIXME unit handling
    for type, x in points.groupby("type"):
        for i_point in range(len(x)):
            s = pv.Sphere(radius=default_point_sizes[type], center=x[i_point])
            if color is None:
                sphere_actor = plotter.add_mesh(s, color=default_point_colors[type])
            else:
                sphere_actor = plotter.add_mesh(s, color=color)
            actors.append(sphere_actor)
            if labels is not None:
                plotter.add_point_labels(x[i_point].values, [str(labels[i_point])])
                
    if ppoints is not None: 
        plotter.enable_surface_point_picking(callback=on_pick, show_point=False)

class PointCloudVisualizer:
    def __init__(self, points, plotter=None, labels = None):
        self.points = points  # Your LabeledPointCloud or similar
        self.plotter = plotter if plotter else pv.Plotter()
        self.picked_points = []
        self.labels = labels
        self.actors = []
        self.color = None
    
    def plot(self):
        plotter = self.plotter
        points = self.points
        color = 'r'
        # FIXME make these configurable
        default_point_colors = {
            PointType.UNKNOWN: "gray",
            PointType.SOURCE: "r",
            PointType.DETECTOR: "b",
            PointType.LANDMARK: "g",
        }
        default_point_sizes = {
            PointType.UNKNOWN: 2,
            PointType.SOURCE: 3,
            PointType.DETECTOR: 3,
            PointType.LANDMARK: 2,
        }

        # points = points.pint.to("mm").pint.dequantify()  # FIXME unit handling
        points = points.pint.dequantify()  # FIXME unit handling
        for type, x in points.groupby("type"):
            for i_point in range(len(x)):
                
                s = pv.Sphere(radius=default_point_sizes[type], center=x[i_point])
                if color is None:
                    sphere_actor = plotter.add_mesh(s, color=default_point_colors[type])
                else:
                    sphere_actor = plotter.add_mesh(s, color=color)
                self.actors.append(sphere_actor)
                if self.labels is not None:
                    plotter.add_point_labels(x[i_point].values, [str(labels[i_point])])

        #if self.picked_points is not None: 
        #    plotter.enable_surface_point_picking(callback=self.on_pick, show_point=False)


    def on_pick(self, picked_point):
        plotter = self.plotter
        points = self.points
        ppoints = self.picked_points
        threshold_distance = 5  # Define how close points have to be to consider them "super close"
        new_point = np.array(picked_point)
        
        # Check if new point is super close to any existing sphere
        for i, existing_point in enumerate(points.values):
            if np.linalg.norm(new_point - existing_point) < threshold_distance:
                #s = pv.Sphere(radius=4, center=existing_point)
                #plotter.add_mesh(s, color='r')
                #if i not in ppoints:
                #    ppoints.append(i)

                idx_to_remove = i
                indexes = np.arange(len(self.points.label))
                selected_indexes = np.delete(indexes, idx_to_remove)

                self.points = self.points.isel(label=selected_indexes)
                self.plotter.remove_actor(self.actors[idx_to_remove])
                del self.actors[idx_to_remove]
                
                return  # Stop the function after removing the sphere
            
        existing_labels = self.points.coords['label'].values
        # Generate a new unique label
        new_label_number = max([int(label.split('-')[-1]) for label in existing_labels]) + 1
        new_label = f'O-{new_label_number}'
        new_group = self.points.coords['group'].values[0]
        new_type = cdc.PointType.LANDMARK if new_group == 'L' else cdc.PointType.UNKNOWN

        # Create the new entry DataArray
        new_center_coords = new_point  # Example new coordinates for the single entry
        
        new_entry = xr.DataArray(
            [new_center_coords],  # Encapsulate in a list to fit the dimensionality
            dims=["label", "digitized"],  # Adjust "coordinate" to match your specific dimension names
            coords={
                "label": [new_label],  # Use the newly generated unique label
                # Assuming all entries share the same 'type' and 'group', you can directly reuse from existing
                "type": ("label", [new_type]),
                "group": ("label", [new_group]),
            }
        ).pint.quantify(units="mm")
        
        #self.points.points.add(new_label, new_center_coords, new_type)
        
        s = pv.Sphere(radius=2, center=new_point)
        
        sphere_actor = plotter.add_mesh(s, color='r')
        
        self.actors.append(sphere_actor)
        
        #self.points = xr.concat([self.points, new_entry], dim='label')
       
        display(self.points)
        self.points = self.points.points.add(new_label, new_center_coords, new_type, new_group)
        display(self.points)

    def update_visualization(self):
        # Clear existing plot and re-plot with the updated self.points
        self.plotter.clear()
        self.plot()  # Re-plot the updated points

    def enable_picking(self):
        self.plotter.enable_surface_point_picking(callback=self.on_pick, show_message=True, show_point=False)

