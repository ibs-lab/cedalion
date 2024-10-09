#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:04:39 2024

@author: lauracarlton
"""

import numpy as np 
import xarray as xr
import pandas as pd
import scipy.sparse
from scipy.spatial import KDTree
import trimesh

import cedalion
import cedalion.dataclasses as cdc
import cedalion.imagereco.forward_model as cfm
from cedalion.geometry.registration import register_trans_rot_isoscale
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion.geometry.segmentation import surface_from_segmentation
from cedalion.imagereco.utils import map_segmentation_mask_to_surface

from .tissue_properties import get_tissue_properties
from tqdm import tqdm 

#%% GETTING THE SPATIAL BASIS 

def get_sensitivity_mask(sensitivity: xr.DataArray, threshold: float = -2, wavelength_idx: int = 0):
    """Generate a mask indicating vertices with sensitivity above a certain threshols
 
    Args:
        sensitivity (xr.DataArray): Sensitivity matrix for each vertex and
            wavelength.
        threshold (float): threshold for sensitivity.
        wavelength_idx (int): wavelength over which to compute mask 
            if multiple wavelengths are in sensitivity.
 
    Returns:
        xr.DataArray: mask containing True when vertex has sensitivity above given threshold
    """
   
    intensity = np.log10(sensitivity[:,:,wavelength_idx].sum('channel'))
    mask = intensity > threshold
    mask = mask.drop_vars('wavelength')
    
    return mask


def downsample_mesh(mesh: xr.DataArray, 
                    mask: xr.DataArray,
                    threshold: cedalion.Quantity = 5 * cedalion.units.mm):
    """Downsample the mesh to get seeds of spatial bases.

    Args:
        mesh (xr.DataArray): mesh of either the brain or scalp surface.
        mask (xr.DataArray): mask specifying which vertices have significant sensitivity.
        threshold (Quantity): distance between vertices in downsampled mesh.
       
    Returns:
        xr.DataArray: downsampled mesh
    
    Initial Contributors:
        - Yuanyuan Gao 
        - Laura Carlton | lcarlton@bu.edu | 2024

    """
    
    mesh_units = mesh.pint.units
    threshold = threshold.to(mesh_units)
    
    mesh = mesh.rename({'label':'vertex'}).pint.dequantify()
    mesh_masked = mesh[mask,:]
    mesh_new = []

    for vv in tqdm(mesh_masked):
        if len(mesh_new) == 0: 
            mesh_new.append(vv)
            tree = KDTree(mesh_new)  # Build KDTree for the first point
            continue
        
        # Query the nearest neighbor within the threshold
        distance, _ = tree.query(vv, distance_upper_bound=threshold.magnitude)

        
        # If no point is within the threshold, append the new point
        if distance == float('inf'):
            mesh_new.append(vv)
            tree = KDTree(mesh_new)  # Rebuild the KDTree with the new point

    
    mesh_new_xr = xr.DataArray(mesh_new,
                               dims = mesh.dims,
                               coords = {'vertex':np.arange(len(mesh_new))},
                               attrs = {'units': mesh_units }
        )
    
    mesh_new_xr = mesh_new_xr.pint.quantify()
    
    return mesh_new_xr




def get_kernel_matrix(mesh_downsampled: xr.DataArray, 
                      mesh: xr.DataArray, 
                      sigma: cedalion.Quantity = 5 * cedalion.units.mm):
    
    """Get the matrix containing the spatial bases.

    Args:
        mesh_downsampled (xr.DataArray): mesh of either the downsampled brain or scalp surface.
            This is used to define the centers of the spatial bases.
        mesh (xr.DataArray): the original fully sampeld mesh of the brain or scalp. 
        sigma (Quantity): standard deviation used for defining the Gaussian kernel.
       
    Returns:
        xr.DataArray: matrix containing the spatial bases
    
    Initial Contributors:
        - Yuanyuan Gao 
        - Laura Carlton | lcarlton@bu.edu | 2024

    """
    assert mesh.pint.units == mesh_downsampled.pint.units
    
    
    mesh_units = mesh.pint.units
    sigma = sigma.to(mesh_units)
    
    # Covariance matrix
    cov_matrix = (sigma.magnitude **2) * np.eye(3)
    inv_cov = np.linalg.inv(cov_matrix)  # Inverse of Cov_matrix
    det_cov = np.linalg.det(cov_matrix)  # Determinant of Cov_matrix
    denominator = np.sqrt((2 * np.pi) ** 3 * det_cov)  # Pre-calculate denominator

    mesh_downsampled = mesh_downsampled.pint.dequantify().values
    mesh = mesh.pint.dequantify().values
    
    diffs = mesh_downsampled[:, None, :] - mesh[None, :, :]

    # Efficient matrix multiplication using np.einsum to compute (x-mu)' * inv_cov * (x-mu) for all pairs
    exponents = -0.5 * np.einsum('ijk,kl,ijl->ij', diffs, inv_cov, diffs)

    # Compute the kernel matrix
    kernel_matrix = np.exp(exponents) / denominator
    
    kernel_matrix_xr = xr.DataArray(kernel_matrix, 
                                    dims = {"vertex", "kernel"},
                                    coords = {'vertex': np.arange(kernel_matrix.shape[1]),
                                              'kernel': np.arange(kernel_matrix.shape[0])}
                                    )
    
    return kernel_matrix_xr



def get_G_matrix(head: cfm.TwoSurfaceHeadModel, 
                 M: xr.DataArray,
                 threshold_brain: cedalion.Quantity = 5 * cedalion.units.mm, 
                 threshold_scalp: cedalion.Quantity = 20 * cedalion.units.mm, 
                 sigma_brain: cedalion.Quantity = 5 * cedalion.units.mm, 
                 sigma_scalp: cedalion.Quantity = 20 * cedalion.units.mm
                 ):
    
    """Get the G matrix which contains all the information of the spatial basis

    Args:
        head (cfm.TwoSurfaceHeadModel): Head model with brain and scalp surfaces.
        M (xr.DataArray): mask defining the sensitive vertices  
        threshold_brain (Quantity): distance between vertices in downsampled mesh for the brain.
        threshold_scalp (Quantity): distance between vertices in downsampled mesh for the scalp.
        sigma_brain (Quantity): standard deviation used for defining the Gaussian kernels of the brain.
        sigma_scalp (Quantity): standard deviation used for defining the Gaussian kernels of the scalp.
       
    Returns:
        xr.DataArray: matrix containing information of the spatial basis. Each column corresponds
            to the vertex representation of one kernel in the spatial basis. 
    
    Initial Contributors:
        - Yuanyuan Gao 
        - Laura Carlton | lcarlton@bu.edu | 2024

    """
    
    brain_downsampled = downsample_mesh(head.brain.vertices, M[M.is_brain], threshold_brain)
    scalp_downsampled = downsample_mesh(head.scalp.vertices, M[~M.is_brain], threshold_scalp)
    
    G_brain = get_kernel_matrix(brain_downsampled, head.brain.vertices, sigma_brain)
    G_scalp = get_kernel_matrix(scalp_downsampled, head.scalp.vertices, sigma_scalp)
    

    G = {'G_brain': G_brain, 
         'G_scalp': G_scalp
         }
    
    return G

#%% TRANSFORMING A 


# TODO -  H = A @ G
# use this for image recon 





