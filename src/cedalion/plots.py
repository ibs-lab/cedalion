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

# Laura's scalp_plot additional imports
import math
import pandas as pd
import matplotlib
import scipy.stats as stats
import matplotlib.colors as clrs
from scipy import signal 
import seaborn as sns
import cedalion 
import cedalion.nirs
import cedalion.xrutils as xrutils


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
    brain_mesh, scalp_mesh, geo3d, timeseries, poly_lines=[], brain_scalars=None, plotter = None):
    """Plots a 3D visualization of brain and scalp meshes.

    Args:
        brain_mesh (TrimeshSurface): The brain mesh as a TrimeshSurface object.
        scalp_mesh (TrimeshSurface): The scalp mesh as a TrimeshSurface object.
        geo3d (xarray.Dataset): Dataset containing 3-dimentional point centers.
        timeseries
        poly_lines
        brain_scalars
        plotter (pv.Plotter, optional): An existing PyVista plotter instance to use for plotting. If None, a new
            PyVista plotter instance is created. Default is None.
    """
    #pv.set_jupyter_backend("server")

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
        labels = geo3d.label.values
    else:
        labels = None

    if geo3d is not None:
        geo3d = geo3d.pint.to("mm").pint.dequantify()  # FIXME unit handling
        for type, x in geo3d.groupby("type"):
            labels = x.label.values
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

    #def callback(point):
    #    mesh = pv.Sphere(radius=3, center=point)
    #    plt.add_mesh(mesh, style='wireframe', color='r')
    #    plt.add_point_labels(point, [f"{point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f}"])

    #if ppoints is not None:
    #    plt.enable_surface_point_picking(callback=callback, show_point=False)

    #plt.show()


def plot_surface(
    plotter: pv.Plotter,
    surface: cdc.Surface,
    color=None,
    opacity=1.0,
    pick_landmarks = False,
    **kwargs,
):
    #used for picking landmarks in photogrammetry example
    """Plots a surface mesh with optional landmark picking in a PyVista plotter.

    Args:
        plotter (pv.Plotter): A PyVista plotter instance used for rendering the surface.
        surface (cdc.Surface): The surface object to be plotted.
        color (str or tuple, optional): Color of the mesh.
        opacity (float): Opacity of the mesh, ranging from 0 (transparent) to 1 (opaque). Default is 1.0.
        pick_landmarks (bool): If True, enables interactive picking of landmarks on the surface. Default is False.

    Returns:
        function: If `pick_landmarks` is True, returns a function that when called, provides the current picked points
        and their labels. This function prints warnings if some labels are missing or are repeated.
    """
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


    # Define landmark labels
    landmark_labels = ['Nz', 'Iz', 'Cz', 'Lpa', 'Rpa']
    picked_points = []
    labels = []
    point_actors = []
    label_actors = []

    def place_landmark(point):
        nonlocal picked_points, point_actors, label_actors, mesh, labels, plotter
        threshold_distance_squared = 25  # Using squared distance to avoid square root calculation

        new_point = np.array(point)

        # Check if the clicked point is close to any existing point
        for i, existing_point in enumerate(picked_points):
            if np.sum((new_point - existing_point) ** 2) < threshold_distance_squared:
                current_label_index = landmark_labels.index(labels[i])
                next_label_index = (current_label_index + 1) % len(landmark_labels)
                next_label = landmark_labels[next_label_index]

                # Check if the next label is the first one in the list
                if next_label == landmark_labels[0]:
                    # Delete the point and its label
                    del picked_points[i]
                    plotter.remove_actor(label_actors[i])
                    plotter.remove_actor(point_actors[i])
                    del point_actors[i]
                    del label_actors[i]
                    del labels[i]
                    return

                labels[i] = next_label
                plotter.remove_actor(label_actors[i])  # Remove previous label
                label_actors[i] = plotter.add_point_labels(existing_point, [next_label], font_size=30)
                return

        # If no point is close enough, create a new point and assign a label
        # Check if there are already 5 points placed
        if len(picked_points) >= 5:
            return

        landmark_label = landmark_labels[0]
        # Add new point and label actors
        point_actor = plotter.add_mesh(pv.Sphere(radius=3, center=new_point), color='green')
        point_actors.append(point_actor)
        label_actor = plotter.add_point_labels(new_point, [landmark_label], font_size=30)
        label_actors.append(label_actor)
        picked_points.append(new_point)
        labels.append(landmark_label)

    # Initialize the labels list
    # labels = [None] * 5  # Initialize with None for unassigned labels

    if pick_landmarks is True:
        def get_points_and_labels():

            if len(labels) < 5:
                print("Warning: Some labels are missing")
            elif len(set(labels)) != 5:
                print("Warning: Some labels are repeated!")
            return picked_points, labels

        plotter.enable_surface_point_picking(callback=place_landmark, show_message = "Right click to place or change the landmark label", show_point=False, tolerance = 0.005)

        return get_points_and_labels

def plot_labeled_points(
    plotter: pv.Plotter,
    points: cdt.LabeledPointCloud,
    color=None,
    show_labels = False,
    ppoints = None,
    labels = None
):
    #used in selecting optode centers in Photogrammetry example. 
    """Plots a labeled point cloud in a PyVista plotter with optional interaction for picking points.

        This function visualizes a point cloud where each point can have a label. Points can be interactively picked if enabled. Picked point is indicated by increased radius.

    Args:
        plotter (pv.Plotter): A PyVista plotter instance used for rendering the points.
        points (cdt.LabeledPointCloud): A labeled point cloud data structure containing points and optional labels.
        color (str or tuple, optional): Override color for all points. If None, colors are assigned based on point types.
        show_labels (bool): If True, labels are displayed next to the points. Default is False.
        ppoints (list, optional): A list to store indices of picked points, enables picking if not None. Default is None.
        labels (list of str, optional): List of labels to show if `show_labels` is True. If None and `show_labels` is True,
            the labels from `points` are used.
    """
    pv.set_jupyter_backend("server")
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


    #labels = None
    if labels == None and show_labels == True:
        labels = points.label.values

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
    # Iterate through each point and its corresponding label
    for i_point, point in enumerate(points):
        # Determine the point type
        point_type = point.coords['type'].item()
        # Create and add a sphere at the point's coordinates
        s = pv.Sphere(radius=default_point_sizes[point_type], center=point.values)
        plotter.add_mesh(s, color=color or default_point_colors[point_type])
        # Add the label if required
        if show_labels and labels is not None:
            plotter.add_point_labels(point.values[np.newaxis], [str(labels[i_point])])

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

class OptodeSelector:
    """A class for visualizing point clouds with interactive features in PyVista.

    This class provides functionality to visualize and interact with labeled point clouds using a PyVista plotter.
    It allows points to be dynamically added or removed by picking them directly from the plot interface. 

    Attributes:
        surface (cdc.Surface): The surface of a head for normals.
        points (cdt.LabeledPointCloud): The point cloud data containing point coordinates.
        normals (xr.DataArray): Normal vectors to the points.
        plotter (pv.Plotter): A PyVista plotter instance for rendering the point cloud.
        labels (list of str, optional): Labels corresponding to the points, displayed if provided.
        actors (list): List of PyVista actor objects representing the points in the visualization.
        color (str or tuple, optional): Default color for points if not specified by point type.

    Methods:
        plot(): Renders the point cloud using the current settings.
        on_pick(picked_point): Callback function for picking points in the visualization.
        update_visualization(): Clears the existing plot and re-renders the point cloud.
        enable_picking(): Enables interactive picking of points on the plot.
    """
    def __init__(self, surface, points, normals=None, plotter=None, labels = None):
        self.points = points
        self.normals = normals
        self.surface = surface
        self.plotter = plotter if plotter else pv.Plotter()
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


    def on_pick(self, picked_point):
        plotter = self.plotter
        points = self.points
        threshold_distance = 5  # Define how close points have to be to consider them "super close"
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
        new_label_number = max([int(label.split('-')[-1]) for label in existing_labels]) + 1
        new_label = f'O-{new_label_number}'
        new_group = self.points.coords['group'].values[0]
        new_type = cdc.PointType.LANDMARK if new_group == 'O' else cdc.PointType.UNKNOWN

        # Create the new entry DataArray
        new_center_coords = new_point 

        new_entry = xr.DataArray(
            [new_center_coords],  
            dims=["label", "digitized"],  
            coords={
                "label": [new_label],              
                "type": ("label", [new_type]),
                "group": ("label", [new_group]),
            }
        ).pint.quantify(units="mm")

        s = pv.Sphere(radius=2, center=new_point)
        sphere_actor = plotter.add_mesh(s, color='r')
        self.actors.append(sphere_actor)

        new_normal = self.find_surface_normal(new_point)
        self.points = self.points.points.add(new_label, new_center_coords, new_type, new_group)
        self.normals = self.update_normals(new_normal, new_label)

    def update_visualization(self):
        # Clear existing plot and re-plot with the updated self.points
        self.plotter.clear()
        self.plot()  

    def enable_picking(self):
        self.plotter.enable_surface_point_picking(callback=self.on_pick, show_message = "Right click to place or remove optode", show_point=False, tolerance = 0.005)

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
        close_vertices = self.surface.mesh.vertices[distances < radius]  # Select vertices within the specified radius      
        eigenvalues, eigenvecs = pca(close_vertices)
        normal_vector = eigenvecs[:, 0]
        # Optionally, verify the direction of the normal
        if np.dot(normal_vector, picked_point - np.mean(close_vertices, axis=0)) < 0:
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




def scalp_plot(snirfObj, metric, ax, colormap=p.cm.bwr, title=None, threshold_ind = None, threshold_col = None, saturation=None, vmin=0, vmax=1, savePath = None, remove_short=0, flagFontSize=0, flagMarkerSize=8):
    '''
    CREATE A 2D MONTAGE OF OPTODES WITH CHANNELS COLOURED ACCORDING TO A GIVEN METRIC
    
    Parameters:
        snirfObj -> valid snirfObj to extract the measurement list
        ax -> axis object to plot the montage
        metric -> metric to plot onto the channels
        colormap -> colormap to use to color the channels (default = jet)
        title -> title for the plot (default = SQE)
        threshold -> metrics values below threshold will be plotted as dotted lines (default = 0)
        vmin -> minimum value for the colorbar (default = 0)
        vmax -> maximum value for the colorbar (default = 1)
    '''

    def cart2sph(x, y, z):
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        return az, el, r
    
    def pol2cart(theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y
    
    def convert_optodePos3D_to_circular2D(pos, tranformation_matrix, norm_factor):
        pos = np.append(pos, np.ones((pos.shape[0],1)), axis=1)
        pos_sphere = np.matmul(pos,tranformation_matrix)
        pos_sphere_norm = np.sqrt(np.sum(np.square(pos_sphere), axis=1))
        pos_sphere_norm= pos_sphere_norm.reshape(-1,1)
        pos_sphere = np.divide(pos_sphere,pos_sphere_norm)
        azimuth, elevation, r = cart2sph(pos_sphere[:,0], pos_sphere[:,1], pos_sphere[:,2])
        elevation = math.pi/2-elevation;
        x, y = pol2cart(azimuth,elevation)
        x = x/norm_factor
        y = y/norm_factor
        return x, y
    
    channels_df = pd.read_excel('10-5-System_Mastoids_EGI129.xlsx') 
    probe_landmark_pos3D = []
    circular_landmark_pos3D = []
    geo3d = snirfObj.geo3d
    
    # landmarks = geo3d.loc[geo3d.type == cedalion.io.snirf.PointType.LANDMARK]
    # sources = geo3d.loc[geo3d.type == cedalion.io.snirf.PointType.SOURCE]
    # detectors = geo3d.loc[geo3d.type == cedalion.io.snirf.PointType.DETECTOR]
    

    # extracts all datapoints from geo3d that have the label 'SOURCE'
    sourcetype = cedalion.dataclasses.geometry.PointType(1)
    # yields a boolean xarray with True where the type is equal to the sourcetype
    bool_idx = geo3d.coords['type'] == sourcetype
    # extracts the data from geo3D where x is True. Result is another geo3D xarray reduced to this subset
    sources = geo3d.where(bool_idx, drop=True)

    detectortype = cedalion.dataclasses.geometry.PointType(2)
    bool_idx = geo3d.coords['type'] == detectortype
    detectors = geo3d.where(bool_idx, drop=True)

    landmarktype = cedalion.dataclasses.geometry.PointType(3)
    bool_idx = geo3d.coords['type'] == landmarktype
    landmarks = geo3d.where(bool_idx, drop=True)

    #### find the landmarks in the probe ####
    for u in range(len(landmarks)):
        idx_list = channels_df.index[channels_df['Label']==landmarks.label[u]].tolist()
        if idx_list:
            circular_landmark_pos3D.append([channels_df['X'][idx_list[0]],channels_df['Y'][idx_list[0]], channels_df['Z'][idx_list[0]]])
            landmark_pos3D = landmarks[u,0:3].to_numpy().tolist()
            landmark_pos3D.extend([1])
            probe_landmark_pos3D.append(landmark_pos3D)
            
    landmarks2D_theta = (channels_df['Theta']*2*math.pi/360).to_numpy()
    landmarks2D_phi = ((90-channels_df['Phi'])*2*math.pi/360).to_numpy()
    x,y = pol2cart(landmarks2D_theta, landmarks2D_phi)
    
    norm_factor = max(np.sqrt(np.add(np.square(x),np.square(y))))
    temp = np.linalg.inv(np.matmul(np.transpose(probe_landmark_pos3D),probe_landmark_pos3D))
    tranformation_matrix = np.matmul(temp,np.matmul(np.transpose(probe_landmark_pos3D),circular_landmark_pos3D))        
    tranformation_matrix = tranformation_matrix
    # tranformation_matrix = np.linalg.lstsq(probe_landmark_pos3D, circular_landmark_pos3D, rcond=None)
        
    # measurements = snirfObj.nirs[0].data[0].measurementList
    skipped_channels = []
    skipped_detectors = []
    skipped_metrics = []
    data = snirfObj["amp"] # .data[0]
    nMeas = len(data.channel)
    
    if remove_short == 1: # then remove any channels that are less than 10mm 
        
    
        for u in range(nMeas):
            
            sourceIndex =  data.source[u]
            detectorIndex =  data.detector[u]
            
            dist = xrutils.norm(geo3d.loc[data.source[u]] - geo3d.loc[data.detector[u]], dim="pos")


            # x = snirfObj.nirs[0].probe.sourcePos3D[sourceIndex-1]
            # y= snirfObj.nirs[0].probe.detectorPos3D[detectorIndex-1]
            # dist = math.dist(x,y)
                
            if dist < 10:
                    skipped_channels.append([sourceIndex, detectorIndex])
                    skipped_detectors.append(detectorIndex)
                    skipped_metrics.append(u)
    
    # if the metrics/threshold_col given include those for short channels, remove them from the array 
    if len(metric) == nMeas//2:
        metric = np.delete(metric,skipped_metrics)
    
    if type(threshold_col) == list:
        if len(threshold_col) == nMeas//2:
            threshold_col = np.delete(threshold_col,skipped_metrics)

    #### scale indices #####
    sourcePos2DX , sourcePos2DY = convert_optodePos3D_to_circular2D(sources, tranformation_matrix, norm_factor)
    detectorPos2DX , detectorPos2DY = convert_optodePos3D_to_circular2D(detectors, tranformation_matrix, norm_factor)
    
    # sourcePos2D = snirfObj.nirs[0].probe.sourcePos2D
    # sourcePos2DX = sourcePos2D[:,0]
    # sourcePos2DY = sourcePos2D[:,1]
    # detectorPos2D = snirfObj.nirs[0].probe.detectorPos2D
    # detectorPos2DX = detectorPos2D[:,0]
    # detectorPos2DY = detectorPos2D[:,1]
    
    
    scale = 1.3
    sourcePos2DX = sourcePos2DX*scale
    detectorPos2DX = detectorPos2DX*scale
    sourcePos2DY = sourcePos2DY*scale
    detectorPos2DY = detectorPos2DY*scale
        
    #### plot the positions on the unit circle ####
    t = np.linspace(0, 2 * np.pi, 100)
    head_x = [math.sin(i) for i in t]
    head_y = [math.cos(i) for i in t]
        
    
    #### plot the channels according to the metrics ####
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = matplotlib.cm.ScalarMappable(cmap=colormap,norm=norm)
    fontDict_src = dict(color='r', fontweight = 'bold', fontstretch= 'expanded',fontsize = flagFontSize, ha='center',va='center')
    fontDict_det = dict(color='b', fontweight = 'bold', fontstretch= 'expanded',fontsize = flagFontSize, ha='center',va='center')
    
    i =0
    for u in range(nMeas):
        sourceIndex =  data.source[u]
        detectorIndex =  data.detector[u]
        
        # skip the short_channels 
        if [sourceIndex, detectorIndex] in skipped_channels:
            continue
        
        
        iS = int(sourceIndex.to_numpy().tolist()[1:])
        iD = int(detectorIndex.to_numpy().tolist()[1:])
        x = [sourcePos2DX[iS-1], detectorPos2DX[iD-1]]
        y = [sourcePos2DY[iS-1], detectorPos2DY[iD-1]]
        

        try:
            assert(threshold_col == None)
        except:
            if threshold_col[i] == 1: #metric[u] < threshold: 
                linestyle = '-'
                alpha = 0.4
            else:
                linestyle = '-'
                alpha = 1
        else:
            linestyle = '-'
            alpha=1
        
        try:
            assert(saturation == None)
        except:
            if saturation[i] == 1:
                color = '0.7'
                alpha = 1
            else:
                color = colormap(norm(metric[i]))
        else:
            color = colormap(norm(metric[i]))
            
        ax.plot(x,y, color=color,linestyle=linestyle, linewidth = 2, alpha=alpha)
        if flagFontSize>0:
            ax.text(x[0], y[0],str(iS),fontdict=fontDict_src) # bbox=dict(color = 'r',boxstyle = "round, pad=0.3", alpha=0.05))
            ax.text(x[1], y[1], str(iD),fontdict=fontDict_det) # bbox=dict(color='b',boxstyle = "round, pad=0.3", alpha=0.05))
        i+=1
    
    ax.plot(head_x,head_y,'k')
    
    if flagMarkerSize>0:
        for u in range(len(sourcePos2DX)):
            ax.plot(sourcePos2DX[u] , sourcePos2DY[u], 'r.', markersize=flagMarkerSize)
        
        for u in range(len(detectorPos2DX)):
            if u+1 in skipped_detectors:
                continue
            ax.plot(detectorPos2DX[u] , detectorPos2DY[u], 'b.',markersize=flagMarkerSize)
     
    if threshold_ind != None:
        ticks = [vmin, (vmin+vmax)//2, threshold_ind, vmax]
    else:   
        ticks = [vmin, (vmin+vmax)//2, vmax]
        
    ax.plot(0, 1 , marker="^",markersize=16)
    p.colorbar(sm,shrink =0.6, ticks=ticks)
    ax.set_title(title)
    p.tight_layout()
    p.axis('equal')
    p.axis('off')
    
    if savePath is not None: 
        p.savefig(savePath, dpi=1200)
    






