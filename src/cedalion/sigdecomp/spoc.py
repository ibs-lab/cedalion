"""Source Power Co-modulation (SPoC) algorithm."""

import numpy as np
from scipy.linalg import eigh
import cedalion.typing as cdt
from typing import Optional
import xarray as xr

class SPoC():
    """Implements the Source Power Co-modulation (SPoC_lambda) algorithm based on :cite:t:`BIBTEXLABEL`.

    Given a vector-valued time signal x(t) and a scalar target function z(t), 
    SPoC finds spatial filters W that maximize the covariance between
    the bandpower of the projected x signal, P(W.T @ x), and z. 
    Such a covariance defines the objective function of the problem, whose solution 
    can be formulated as the one for a generalized eigenvalue problem.
    The reconstructed sources s are given by the backward model s = W.T @ x.

    
    Assumptions:
        x(t) is of shape Nx x Nt, where Nx is the number of channels and Nt the 
        number of time points, and it is band-pass filtered in the frequency band 
        of interest. 
        z(e) is a standardize vector (zero mean and unit variance) of shape 1 x Ne, 
        where Ne < Nt is the number of "epochs". The latter represent labels 
        for intervals of the original time series. 
        Bandpower of the projected signal W.T @ x is then approximated by its 
        variance within epochs.
    
        
    Args:
        n_comp (int): Number of components the algorithm will find in decreasing 
        order of scores/eigenvalue. n_comp=1 returns the component of the highest eigenvalue. 
        If None, n_comp = Nx, the maximum possible number of components.
    """
    
    def __init__(self,
                 n_comp: Optional[int] = None):
        
        # Attributes will be initialized after calling the fit function
        self.W = None  # Filters
        self.scores = None  # Eigenvalues
        self.Nx = None  # Number of channels
        self.n_comp = n_comp

    def fit(self,
            x: cdt.NDTimeSeries,
            z: xr.DataArray) -> np.ndarray:
        """ Fit the model on the (x, z) dataset.

        Solve the generalized eigenvalue problem and store the trained spatial filters W
        as a local state of the class.
        
        Args:
            x (:class:`NDTimeSeries`, (channel, time)): Temporal signal 
            of shape Nx x Nt.
            z (:class:`DataArray`, (time)): Target (scalar) function
            of shape 1 x Ne, with Ne < Nt.
            n_comp (int): Number of components the algorithm will find in decreasing 
                order of scores/eigenvalue. n_comp=1 returns the component of the highest eigenvalue. 
                If None, n_comp = Nx, the maximum possible number of components. 
        
        Returns:
            scores: Array of Nx eigenvalues. The latter also coincide with 
                the corresponding covariances between P(W.T @ x) and z. 
        """
        
        # Store for transformation later
        self.Nx = len(x.channel)

        # Catch incompatible lengths
        Ne = len(z.time)
        Nt = len(x.time)
        if Nt <= Ne:
            raise ValueError("x should have more time points than z")
        if self.n_comp and self.n_comp > self.Nx:
            raise ValueError(f"Number of components {self.n_comp} should be smaller than number of x channels {self.Nx}")
        
        # Standardize z
        z = standardize(z, dim='time')
        
        # Split signal into epochs and build matrices for the eigenvalue problem
        x_epochs = x.groupby_bins(group='time', bins=Ne)
        Cxxe = np.stack([np.cov(xe) for _, xe in x_epochs])  # Epoch-wise cov. matrix
        Cxx = Cxxe.mean(axis=0)
        Cxxz = (Cxxe * z.values.reshape(-1, 1, 1)).mean(axis=0)
        
        # Restrict to number of components
        subset = [self.Nx - 1 - self.n_comp, self.Nx - 1] if self.n_comp else None
        # Solve generalized eigenvalue problem
        self.scores, W = eigh(Cxxz, Cxx, eigvals_only=False, subset_by_index=subset)
        # Bring to decreasing order
        self.scores = self.scores[::-1]
        W = W[:, ::-1]
        # Update state
        self.W = xr.DataArray(W, 
                              coords={'channel': x.channel, 
                                      'component': [f'S{i}' for i in range(len(self.scores))]})
        
        return self.scores
    
    def transform(self,
                  x: cdt.NDTimeSeries, 
                  get_bandpower: bool = True,
                  Ne: Optional[int] = None) -> xr.DataArray | tuple[xr.DataArray, 
                                                                    xr.DataArray]:
        """ Apply backward model to x to build reconstructed sources.

        Get reconstructed sources s by projecting x along the spatial filtes.
        If only_component = False, also estimate epoch-wise bandpower of the
        components via the per-epoch variance. 

        Args:
            x (:class:`NDTimeSeries`, (channel, time)): Temporal signal
                of shape Nx x Nt.
            only_component: Wether to return only the reconstructed sources
                or also the epoch-wise bandpower.
            Ne: Number of epochs along which to estimate the bandpower.

        Returns:
            s: Reconstructed sources (W.T @ x).
            s_power: standardized epoch-wise bandpower of s (Var(W.T @ x)).
        """
        
        # Catch icompatible number of channels
        if len(x.channel) != self.Nx:
            raise ValueError(f"x should have {self.Nx} number of channels, but {len(x.channel)} was found!")

        # Build reconstructed sources from projection
        s = standardize(self.W.T @ x, dim='time')
        
        if get_bandpower:
            epochs = np.linspace(s.time[0], s.time[-1], Ne)
            # Split into epochs, estimate bandpower, and standardize
            s_power = s.groupby_bins(group='time', bins=Ne).var()
            s_power = s_power.rename({'time_bins': 'time'}).assign_coords({'time': epochs})
            s_power = standardize(s_power, dim='time')

            return s, s_power
        
        else:
            return s 

def standardize(x: cdt.NDTimeSeriesSchema, dim: str = 'time'):
    """Standardize x along dimension dim.
    """

    x_standard = (x - x.mean(dim))/x.std(dim)

    return x_standard