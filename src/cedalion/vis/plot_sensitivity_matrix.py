import numpy as np
import matplotlib.pyplot as p
from matplotlib.colors import ListedColormap
import pyvista as pv

import cedalion
import cedalion.dataclasses as cdc

class Main():
    def __init__(self, sensitivity, brain_surface, 
                 head_surface=None, labeled_points=None, wavelength=760):
        
        # Initialize
        self.brain = brain_surface
        self.head = head_surface
        self.sensitivity = sensitivity
        self.wavelength = wavelength
        self.labeled_points = labeled_points

        self.plt = pv.Plotter()

    def plot(self, low_th=-3, high_th=0):

        cedalion.plots.plot_surface(self.plt, self.brain, color="w")
        if self.head is not None:
            cedalion.plots.plot_surface(self.plt, self.head, opacity=.1)
        if self.labeled_points is not None:
            cedalion.plots.plot_labeled_points(self.plt, self.labeled_points)


        b = cdc.VTKSurface.from_trimeshsurface(self.brain)
        b = pv.wrap(b.mesh)

        sensitivity_matrix = self.sensitivity.where(self.sensitivity['is_brain'], drop=True)
        sensitivity_matrix = sensitivity_matrix.sel(wavelength = self.wavelength).sum(dim='channel').values


        sensitivity_matrix[sensitivity_matrix<=0] = sensitivity_matrix[sensitivity_matrix>0].min()
        sensitivity_matrix = np.log10(sensitivity_matrix)

        # Original colormap
        cmap = p.cm.get_cmap("jet", 256)

        gray = [1, 1, 1, 1]  # RGBA for gray
        new_cmap_colors = np.vstack((gray, cmap(np.linspace(0, 1, 255))))
        custom_cmap = ListedColormap(new_cmap_colors)

        self.plt.add_mesh(b, scalars=sensitivity_matrix, cmap=custom_cmap, clim=(low_th, high_th), 
                    scalar_bar_args={'title':'Sensitivity (m⁻¹): Logarithmic Scale', 'shadow':True},)
        
        


        