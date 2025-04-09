import matplotlib.pyplot as p
import pyvista as pv
from matplotlib.colors import ListedColormap
import cedalion.typing as cdt
from cedalion.imagereco.forward_model import TwoSurfaceHeadModel
import numpy as np

import cedalion.dataclasses as cdc 



def plot_image_recon( X: cdt.NDTimeSeries, head: TwoSurfaceHeadModel, shape, iax, clim=(0,1), flag_hbx='hbo_brain', view_position='superior',
                     p0 = None, title_str = None, off_screen= True ):

    cmap = p.get_cmap("seismic", 1024) # or "jet"
    new_cmap_colors = np.vstack((cmap(np.linspace(0, 1, 256))))
    custom_cmap = ListedColormap(new_cmap_colors)

    X_hbo_brain = X[X.is_brain.values, 0]
    X_hbr_brain = X[X.is_brain.values, 1]

    X_hbo_scalp = X[~X.is_brain.values, 0]
    X_hbr_scalp = X[~X.is_brain.values, 1]

    positions = {
        'superior': [0, 0, 1],
        'left': [-1, 0, 0],
        'right': [1, 0, 0],
        'anterior': [0, 1, 0],
        'posterior': [0, -1, 0],
        'scale_bar': [0, 0, 1]
    }
    # FIXME provide clim if not provided
    #clim=(-X_hbo_brain.max(), X_hbo_brain.max())

    # Get the camera direction vector for the specified view_position
    camera_direction = positions.get(view_position, [0, 0, 1])

    if p0 is None:
        p0 = pv.Plotter(shape=(shape[0],shape[1]), window_size = [2000, 1500], off_screen=off_screen)

    p0.subplot(iax[0], iax[1])

    show_scalar_bar = False
    smooth_shading = True

    if flag_hbx in ['hbo_brain', 'hbr_brain']:
        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
    elif flag_hbx in ['hbo_scalp', 'hbr_scalp']:
        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
    else:
        raise ValueError(f"Invalid flag_hbx: {flag_hbx}")
    surf = pv.wrap(surf.mesh)

    # Calculate the centroid of the mesh
    centroid = np.mean(surf.points, axis=0)

    if flag_hbx == 'hbo_brain': # hbo brain 
        surf['brain'] = X_hbo_brain
        if clim is None:
           clim=(-X_hbo_brain.max(), X_hbo_brain.max())
        # FIXME propagate this surf change to other HbXs and Scalp
#        p0.add_mesh(surf, scalars=X_hbo_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=smooth_shading, interpolate_before_map=False )
        p0.add_mesh(surf, scalars='brain', cmap=custom_cmap, clim=clim,
                    show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), 
                    smooth_shading=smooth_shading, interpolate_before_map=False )
    elif flag_hbx == 'hbr_brain': # hbr brain
        surf['brain'] = X_hbr_brain
        if clim is None:
           clim=(-X_hbr_brain.max(), X_hbr_brain.max())
        p0.add_mesh(surf, scalars=X_hbr_brain, cmap=custom_cmap, clim=clim,
                    show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9),
                    smooth_shading=smooth_shading )
    elif flag_hbx == 'hbo_scalp': # hbo scalp
        surf['brain'] = X_hbo_scalp
        if clim is None:
            clim=(-X_hbo_scalp.max(), X_hbo_scalp.max())
        p0.add_mesh(surf, scalars=X_hbo_scalp, cmap=custom_cmap, clim=clim,
                    show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9),
                    smooth_shading=smooth_shading )
    elif flag_hbx == 'hbr_scalp': # hbr scalp
        surf['brain'] = X_hbr_scalp
        if clim is None:
            clim=(-X_hbr_scalp.max(), X_hbr_scalp.max())
        p0.add_mesh(surf, scalars=X_hbr_scalp, cmap=custom_cmap, clim=clim,
                    show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9),
                    smooth_shading=smooth_shading )

    # Determine an appropriate view up vector based on view_position
    if view_position == 'superior':
        view_up = [0, 1, 0]  # Use a lateral view-up vector for top-down view
    else:
        view_up = [0, 0, 1]  # Default view up vector

    p0.camera_position = [
        centroid + np.array(camera_direction) * 500,  # Camera position
        centroid,                                     # Focal point (centroid)
        view_up                                       # Updated view up direction
    ]

    if iax[0] == 1 and iax[1] == 1:
        p0.clear_actors()
        p0.add_scalar_bar(title=title_str, vertical=False, position_x=0.1, position_y=0.5,
                          height=0.1, width=0.8, fmt='%.1e',
                          label_font_size=24, title_font_size=32 )  # Add it separately
        # FIXME only add this if a movie... best to add it in calling function
        surf_label = p0.add_text('', position='upper_left', font_size=10 )
    else:
        surf_label = p0.add_text(view_position, position='lower_left', font_size=10)

    return p0, surf, surf_label



def surface_multi_view( X_ts: cdt.NDTimeSeries, head: TwoSurfaceHeadModel, clim=None,
                       view_type: str ='hbo_brain', title_str: str=None,
                       filename: str =None, SAVE: bool = False, SAVE_GIF: bool =False,
                       frame_indices = None, fps: int =6 ):
    """Generate a multi-view 3D visualization of brain or scalp data projected onto a head mesh, with optional saving as a static image or animated GIF over time.

    Args:
        X_ts : xarray.DataArray
            Functional image data. Must be 2D (vertex × chromo) or 3D (vertex × chromo × time).
            Requires 'vertex' and 'chromo' dimensions, and optionally 'time'.
        head : object
            A head surface mesh object with attributes `.brain` and `.scalp` containing
            `trimesh`-compatible geometry.
        clim : tuple of (float, float), optional
            Color limits for the visualization. If None, it will be automatically set based on the data.
        view_type : str, optional
            Type of signal and surface to display. Options include:
            - 'hbo_brain', 'hbr_brain'
            - 'hbo_scalp', 'hbr_scalp'
            Default is 'hbo_brain'.
        title_str : str, optional
            Title string for the scalar bar in the central panel (used when adding a scale bar).
        filename : str, optional
            Path and base name for saving the static PNG image or animated GIF without file extension. 
            Required if SAVE or SAVE_GIF is True.
        SAVE : bool, optional
            If True, saves a static image of the multi-view layout as `<filename>.png`.
        SAVE_GIF : bool, optional
            If True, generates an animated GIF of the changing signal over time, saving to `<filename>.gif`.
            Requires X_ts to have a 'time' dimension and `frame_indices` to be specified.
        frame_indices : tuple of (int, int, int), optional
            Tuple specifying the frame range for the GIF: (start_index, stop_index, step).
            Required if SAVE_GIF is True.
        fps : int, optional
            Frames per second for the animated GIF. Default is 6.

    Returns:
        None
            Displays or saves the multi-view rendering. Returns nothing.

    Initial Contributors:
    - David Boas | dboas@bu.edu | 2025
    - Alexander von Lühmann | vonluehmann@tu-berlin.de | 2025
    """

    subplot_shape = (2,3)

    if X_ts.ndim == 2: # assumed to be vertex and chromo
        foo_img = X_ts
    else: # assumed to have time
        # frame_indices must exist
        if frame_indices is None:
            print('No frame_indices provided. Please provide frame_indices=(idx_start,idx_stop,idx_step).')
            return
        foo_img = X_ts.isel(time=frame_indices[0]) 


    p0,_,surf0_label = plot_image_recon(foo_img, head, subplot_shape, (1,1), clim, view_type, 'scale_bar',
                            None, title_str, off_screen=SAVE )
    p0,surf1,_ = plot_image_recon(foo_img, head, subplot_shape, (0,0), clim, view_type, 'left', p0)
    p0,surf2,_ = plot_image_recon(foo_img, head, subplot_shape, (0,1), clim, view_type, 'superior', p0)
    p0,surf3,_ = plot_image_recon(foo_img, head, subplot_shape, (0,2), clim, view_type, 'right', p0)
    p0,surf4,_ = plot_image_recon(foo_img, head, subplot_shape, (1,0), clim, view_type, 'anterior', p0)
    p0,surf5,_ = plot_image_recon(foo_img, head, subplot_shape, (1,2), clim, view_type, 'posterior', p0)

    # FIXME: this is not working. It shows the p0 window
    # # update the text in surf0_label
    # if surf0_label is not None:
    #     if X_ts.ndim == 2:
    #         surf0_label.set_text('lower_left', f'Frame {0}', off_screen=SAVE)
    #     else:
    #         surf0_label.set_text('lower_left', f'Time {float(X_ts.time[0].values):0.1f} sec', off_screen=SAVE)
    # p0.render(off_screen=SAVE)

    if SAVE:
        if filename is None:
            print('No filename provided. You should provide a filename preferably with a path.')
        else:
            p0.screenshot( filename+'.png' )



    if SAVE_GIF:
        if filename is None:
            print('No filename provided. You should provide a filename preferably with a path.')
            return
        if X_ts.ndim == 2:
            print('X_ts is not a time series. Cannot create GIF.')
            return
        if frame_indices is None:
            print('No frame_indices provided. Please provide frame_indices=(idx_start,idx_stop,idx_step).')
            return

        p0.open_gif( filename+'.gif', fps=6 )

        # check if view_type contains 'hbo'
        if 'hbo' in view_type:
            HbX = 'HbO'
        else:
            HbX = 'HbR'

        foo_v = np.full([X_ts.vertex.size, 1], np.nan)
        for idx_frame in range(frame_indices[0],frame_indices[1],frame_indices[2]):

            foo_v[:,0] = X_ts.sel(chromo=HbX).isel(time=idx_frame).values

            surf1['brain'] = foo_v[X_ts.is_brain,0]
            surf2['brain'] = foo_v[X_ts.is_brain,0]
            surf3['brain'] = foo_v[X_ts.is_brain,0]
            surf4['brain'] = foo_v[X_ts.is_brain,0]
            surf5['brain'] = foo_v[X_ts.is_brain,0]
            surf0_label.set_text('upper_left', f'Time = {float(X_ts.time[idx_frame].values):0.1f} sec')

            p0.write_frame()

    p0.close()