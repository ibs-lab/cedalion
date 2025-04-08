import matplotlib.pyplot as p
import pyvista as pv
from matplotlib.colors import ListedColormap

import numpy as np

import cedalion.dataclasses as cdc 



def plot_image_recon( X, head, shape, iax, clim=(0,1), flag_hbx='hbo_brain', view_position='superior', p0 = None, title_str = None, off_screen= True ):

    cmap = p.get_cmap("jet", 256)
    new_cmap_colors = np.vstack((cmap(np.linspace(0, 1, 256))))
    custom_cmap = ListedColormap(new_cmap_colors)

    X_hbo_brain = X[X.is_brain.values, 0]
    X_hbr_brain = X[X.is_brain.values, 1]

    X_hbo_scalp = X[~X.is_brain.values, 0]
    X_hbr_scalp = X[~X.is_brain.values, 1]

    pos_names = ['superior', 'left', 'right', 'anterior', 'posterior','scale_bar']
    positions = [ 'xy',
        [(-400., 96., 130.),
        (96., 115., 165.),
        (0,0,1)],
        [(600, 96., 130.),
        (96., 115., 165.),
        (0,0,1)],
        [(100, 500, 200),
        (96., 115., 165.),
        (0,0,1)],
        [(100, -300, 300),
        (96., 115., 165.),
        (0,0,1)],
        [(100, -300, 300),
        (96., 115., 165.),
        (0,0,1)]
    ]
    # FIXME provide clim if not provided
    #clim=(-X_hbo_brain.max(), X_hbo_brain.max())

    # get index of pos_names that matches view_position
    idx = [i for i, s in enumerate(pos_names) if view_position in s]

    pos = positions[idx[0]]

    if p0 is None:
        p0 = pv.Plotter(shape=(shape[0],shape[1]), window_size = [2000, 1500], off_screen=off_screen)

    p0.subplot(iax[0], iax[1])

    show_scalar_bar = False
    smooth_shading = True

    if flag_hbx == 'hbo_brain': # hbo brain 
        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
        surf = pv.wrap(surf.mesh)
        surf['brain'] = X_hbo_brain
        if clim is None:
           clim=(-X_hbo_brain.max(), X_hbo_brain.max())
        # FIXME propagate this surf change to other HbXs and Scalp
#        p0.add_mesh(surf, scalars=X_hbo_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=smooth_shading, interpolate_before_map=False )
        p0.add_mesh(surf, scalars='brain', cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=smooth_shading, interpolate_before_map=False )
        p0.camera_position = pos

    elif flag_hbx == 'hbr_brain': # hbr brain
        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
        surf = pv.wrap(surf.mesh)   
        if clim is None:
           clim=(-X_hbr_brain.max(), X_hbr_brain.max())
        p0.add_mesh(surf, scalars=X_hbr_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=smooth_shading )
        p0.camera_position = pos

    elif flag_hbx == 'hbo_scalp': # hbo scalp
        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
        surf = pv.wrap(surf.mesh)
        if clim is None:
            clim=(-X_hbo_scalp.max(), X_hbo_scalp.max())
        p0.add_mesh(surf, scalars=X_hbo_scalp, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=smooth_shading )
        p0.camera_position = pos

    elif flag_hbx == 'hbr_scalp': # hbr scalp
        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
        surf = pv.wrap(surf.mesh)
        if clim is None:
            clim=(-X_hbr_scalp.max(), X_hbr_scalp.max())
        p0.add_mesh(surf, scalars=X_hbr_scalp, cmap=custom_cmap, clim=clim, show_scalar_bar=show_scalar_bar, nan_color=(0.9,0.9,0.9), smooth_shading=smooth_shading )
        p0.camera_position = pos

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



def surface_multi_view( X_ts, head, clim=None, view_type='hbo_brain', title_str=None, filename=None, SAVE=False, SAVE_GIF=False, frame_indices=None, fps=6 ):
    # X_ts - xarray with vertices and chromo. Can also have time. FIXME: add reltime?
    # head - surface mesh object
    # title_str - title for the plot
    # root_dir - root directory for saving
    #
    # clim = None
    # view_type = 'hbo_brain'
    #
    # FIXME:
    # frame_indices (idx_start, idx_stop, idx_step)
    # fps = 6

    subplot_shape = (2,3)

    if X_ts.ndim == 2: # assumed to be vertex and chromo
        foo_img = X_ts
    else: # assumed to have time
        foo_img = X_ts.isel(time=0) # FIXME use idx_start


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