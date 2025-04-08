
import os
import cedalion.plots as plots
import matplotlib.pyplot as p
import numpy as np
import imageio
from PIL import Image



def scalp_plot_gif( data_ts, geo3d, frame_range, filename, scl=None, fps=10, optode_size=6, optode_labels=False, str_title='' ):
    # data_ts - xarray with dimensions channel x time. Can be either a wavelength or chromo. 
    #           It needs to have source and detector labels.

    if scl is None:
        scl = (-np.max(np.abs(data_ts.values)),np.max(np.abs(data_ts.values)))

    f,axs = p.subplots(1, 1, figsize=(8, 8))

    ax1 = axs

    ax1.figure.canvas.draw()
    frames = []

    for idx_frame in range(frame_range[0], frame_range[1], frame_range[2]):

        ax1.cla()
        ax1.set_position([0.1, 0.1, 0.8, 0.8])  # reset position to avoid inset growth from colorbar
        plots.scalp_plot( 
            data_ts,
            geo3d,
            data_ts.isel(time=idx_frame).values,
            ax1,
            cmap='jet',#'gist_rainbow',
            vmin=scl[0],
            vmax=scl[1],
            optode_labels=optode_labels,
            title=f"Time: {float(data_ts.time[idx_frame].values):0.1f}s\n{str_title}",
            optode_size=optode_size,
            add_colorbar=False
        )
        ax1.figure.canvas.draw()
        image = Image.frombytes('RGB', ax1.figure.canvas.get_width_height(), ax1.figure.canvas.tostring_rgb())
        frames.append(image)

    frames[0].save(filename, save_all=True, append_images=frames[1:], duration=1000/fps, loop=0)

