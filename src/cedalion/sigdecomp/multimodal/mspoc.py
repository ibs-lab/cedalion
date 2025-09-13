"""Implements the mSPoC algorithm for multimodal data."""

import numpy as np
from scipy.linalg import eigh
from sklearn.cross_decomposition import CCA
import xarray as xr


from cedalion.sigdecomp.multimodal.utils_multimodal_models import validate_dimension_labels, validate_time_shifts

class mSPoC():
    """Implements the multimodal Source Power Co-modulation (mSPoC) algorithm based on :cite:t:`Dahne2013`.

    Given two vector-valued time series X(t), and Y(t), mSPoC finds component pairs Sx = Wx.T @ X,
    and Sy = Wy.T @ Y, such that the covariance between the temporally-embedded bandpower of Sx and the time course of Sy
    is maximized. The solution to that optimization problem is captured by the spatial (Wx, Wy), and temporal (Wt) filters.

    
    X(t) must be of shape Ntx x Nx, where Nx is the number of channels and Nt the number of time points, 
    and it is band-pass filtered in the frequency band of interest. Bandpower of Sx is then approximated 
    by its variance within epochs. The latter are defined by the time points of Y(t), which must have a 
    greater sampling rate. Y(t) is of shape Nty x Ny, and Nty < Ntx. Both signals are mean-free and 
    temporally aligned.
    
    Args:
        N_components (int): Number of component pairs the algorithm will find.
        time_shifts (list): List of time shifts to consider in the temporal embedding.
        N_restarts (int): Number of times the algorithm is repeated.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance value used for convergence criterion when comparing correlations of consecutive runs.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.
        shift_source (bool): Whether to shift the reconstructed sources by the optimal time lag found during training. Defaults to True.
    """

    def __init__(self,
                 N_components: int = None,
                 time_shifts = None,
                 N_restarts: int = 2,
                 max_iter: int = 200, 
                 tol: float = 1e-5, 
                 scale: bool = True,
                 shift_source : bool = True):

        # General parameters
        self.N_components = N_components
        self.N_restarts = N_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale
        self.Ntx = None
        self.Nty = None
        self.Nx = None
        self.Ny = None
        self.N_features = None

        # Labels for sample and feature dimensions
        self.sample_name = None
        self.featureX_name = None
        self.featureY_name = None

        # Labels for latent space dimension
        self.latent_featureX_name = self.__class__.__name__ + '_X'
        self.latent_featureY_name = self.__class__.__name__ + '_Y'
        self.latent_featureT_name = self.__class__.__name__ + '_T'

        # Normalization parameters
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        # Filters
        self.Wx = None
        self.Wy = None
        self.Wt = None

        # Temporal embedding parameters
        self.time_shifts = time_shifts
        self.N_shifts = len(time_shifts)
        self.optimal_shift = None
        self.shift_source = shift_source

    def fit(self, 
            X: xr.DataArray, 
            Y: xr.DataArray, 
            featureX_name : str = 'channel', 
            featureY_name : str = 'channel'):     
        """Train mSPoC model on the X, Y dataset

        Implement the pseudo-code of Algorithm 1 of :cite:t:`Dahne2013` 
        for a single component pair. After training, the filter attributes
        Wx, Wy, and Wt are updated.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            featureX_name (str): Name of the feature dimension for X.
            featureY_name (str): Name of the feature dimension for Y.            
        """

        # Labels for sample and feature directions
        self.sample_name = 'time'
        self.featureX_name = featureX_name
        self.featureY_name = featureY_name

        # Validates input data and returns them ordered as (sample_name, feature_name)
        X, Y = self.validate_inputs_fit(X, Y)
        e_len = self.Ntx//self.Nty  # Epoch length
        time = Y.time.data

        # Check right format for time shifts and return them in the right order and including zero lag
        self.time_shifts = validate_time_shifts(T=X.time[-1], time_shifts=self.time_shifts)
        self.N_shifts = len(self.time_shifts)

        # Number of deflation steps before looking for the N_components
        N_loops = np.min([self.N_components + 5, self.N_features - 1])

        # Initialize filters
        Wx = np.zeros([self.Nx, N_loops])
        Wy = np.zeros([self.Ny, N_loops])
        Wt = np.zeros([self.N_shifts, N_loops])
        corr = np.zeros(N_loops)

        # Initialize deflation matrices
        Bx = np.eye(self.Nx)
        By = np.eye(self.Ny)

        # Store full datasets for deflation
        X_full = X.copy()
        Y_full = Y.copy()
        # From this point on we work with numpy arrays
        X = X.data
        Y = Y.data

        for i in range(N_loops):

            # Split signal into epochs
            X_epochs = self.split_epochs(X, self.Nty, e_len)
            # Epoch-wise covariance matrix, its temporal embedding and mean
            Cxxe = np.stack([np.cov(x_e.T) for x_e in X_epochs])
            # Catch D=1 case
            if len(X)==1:
                Cxxe = Cxxe.reshape(-1, 1, 1)
            tCxxe = temporal_embedding(Cxxe, self.time_shifts, time)
            # Cxx = Cxxe.mean(axis=0)
            Cxx = (X.T @ X) / (self.Ntx - 1)
            # Run one-unit algorithm
            corr[i], wx, wy, wt = self.one_unit_algorithm(Y, Cxx, Cxxe, tCxxe, time)
            # Project into original space
            Wx[:, i] = (Bx @ wx).squeeze()
            Wy[:, i] = (By @ wy).squeeze()
            Wt[:, i] = wt.squeeze()

            # Deflate data and update proyection matrices
            X, Bx = self.deflate_data(X_full.data, Wx[:, :i+1])
            Y, By = self.deflate_data(Y_full.data, Wy[:, :i+1])
        
        # Sort eigenvalues and eigenvectors in descending order and select number of components
        idx = np.argsort(corr)[::-1][:self.N_components]
        Wx = Wx[:, idx]
        Wy = Wy[:, idx]
        Wt = Wt[:, idx]

        self.Wx, self.Wy, self.Wt = self.convert_filters_to_DataArray(Wx, Wy, Wt, X_full, Y_full)
        
        # Estimate optimal time lag
        self.estimate_optimal_shift()
    
    def transform(self, 
                  X: xr.DataArray, 
                  Y: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
        """Get reconstructed sources of the X and Y dataset.

        The X component is constructed by computing the bandpower of the X projection 
        along Wx, and then applying a liner temporal filter using Wt. 
        The Y component is constructed as the linear projection of Y along Wy.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
        Return:
            tuple: A tuple (Sx, Sy) where:
                Sx (DataArray): Reconstructed source of modality X.
                Sy (DataArray): Reconstructed source of modality Y.

        """

        # Check that X and Y have the same labels and number of features as the training data
        X, Y = self.validate_inputs_transform(X, Y)
        Ntx = X.shape[0]
        Nty = Y.shape[0]
        e_len = Ntx//Nty
        time = Y.time.data

        # Split signal into epochs
        X_epochs = self.split_epochs(X.data, Nty, e_len)
        # Epoch-wise covariance matrix
        Cxxe = np.stack([np.cov(x_e.T) for x_e in X_epochs])
        # Epoch-wise bandpower with temporal embedding
        phi_x = self.get_bandpower(self.Wx.data, Cxxe)
        Phi_x = temporal_embedding(phi_x, self.time_shifts, time)
        Phi_x = Phi_x.T
        
        # Reconstructed sources
        Sx = np.stack([p @ w for p, w in zip(Phi_x, self.Wt.data.T)]).T
        Sx = xr.DataArray(data=Sx, 
                          coords={'time': time, 
                                  self.latent_featureX_name: self.Wx[self.latent_featureX_name]})
        Sy = Y @ self.Wy

        # Shift Sx by optimal shift found during training
        if self.shift_source:
            # Shift with zero padding
            Sx = self.shift_by_optimal(Sx)
            # Truncate to remove zero padding 
            minimal_shift = np.min(self.optimal_shift)
            Sx = Sx.loc[Sx.time <= Sx.time[-1].data - minimal_shift]
            Sy = Sy.loc[Sy.time <= Sy.time[-1].data - minimal_shift]

        return Sx, Sy


    def one_unit_algorithm(self, Y, Cxx, Cxxe, tCxxe, time):
        """ Run the one-unit algorithm of mSPoC to compute one single set of filters.
        """
        
        # Store new dimension
        Nx = len(Cxx)
        # To keep track of best model
        corr_best = 0.0
        wx_best = wy_best = wt_best = 0.0
        for i in range(self.N_restarts):
            
            # Initialize random filter
            wx = np.random.normal(0, 1, [Nx, 1])
            # Epoch-wise bandpower with temporal embedding
            phi_x = self.get_bandpower(wx, Cxxe)
            Phi_x = temporal_embedding(phi_x, self.time_shifts, time=time)
            Phi_x = Phi_x.squeeze().T
            # For convergence condition
            converged = False
            corr_old = 0.0
            # print('Restart', i+1)
            for i in range(self.max_iter):
                # Apply CCA to get wt and wy
                wt, wy = self.apply_cca(Phi_x, Y)
                # Apply temporal filter to tCxxe
                hCxxe = (wt.reshape(-1, 1, 1, 1) * tCxxe).sum(axis=0)
                # Reconstructed y-source
                sy = Y @ wy
                # Build matrix for SPoC algorithm 
                hsy = (sy.reshape(-1, 1, 1) * hCxxe).mean(axis=0)
                # Solve generalized eigenvalue problem (SPoC)
                subset = [Nx - 1, Nx - 1]  # Get most positive eigenvalue
                _, wx = eigh(hsy, Cxx, eigvals_only=False, subset_by_index=subset)
                # Epoch-wise bandpower with temporal embedding
                phi_x = self.get_bandpower(wx, Cxxe)
                Phi_x = temporal_embedding(phi_x, self.time_shifts, time)
                Phi_x = Phi_x.squeeze().T
                # Compute correlation
                sx = Phi_x @ wt
                corr = np.corrcoef(sx[:, 0], sy[:, 0])[0, 1]
                # Check for convergence
                if np.abs(corr - corr_old) < self.tol:
                    converged = True
                    break
                else:
                    corr_old = corr

            # Check for convergence
            if converged and (corr > corr_best):  # Save filters for best model
                wx_best = wx
                wy_best = wy
                wt_best = wt
                corr_best = corr
            
        if not corr_best:
            print(f"mSPoC did not converged in any of the {self.N_restarts} restarts!"
                  " (Reached maximum number of iterations)."
                  " Returning last computed filters and correlation.")
            wx_best = wx
            wy_best = wy
            wt_best = wt
            corr_best = corr

        return corr_best, wx_best, wy_best, wt_best
    
    def deflate_data(self, x, w):
        """ Deflate data by removing from x the contribution of the projection on w.
        """

        # Builld orthonormal basis
        B = get_orthonormal_matrix(w)
        # Project to space orthonormal to w
        x = x @ B
        
        return x, B

    def get_bandpower(self, W, C):
        """Compute bandpower with temporal embedding.

        It estimates the bandpower of a signal by computing the variance within epochs.
        """

        # Epoch-wise power
        phi_x = np.stack([[w.T.dot(c).dot(w) for c in C] for w in W.T]).T
        return phi_x
    
    @staticmethod
    def apply_cca(a, b):
        """Initialize and fit 1-component CCA to the a, b pair.
        """
        
        cca = CCA(n_components=1).fit(a, b)
        wa = cca.x_weights_
        wb = cca.y_weights_

        return wa, wb
        
    @staticmethod
    def split_epochs(x, Ne, e_len):
        """Split a signal x into Ne epochs of length e_len.
        """

        return np.stack([x[i*e_len:(i+1)*e_len] for i in range(Ne)])
        
    def validate_inputs_fit(self,
                        X : xr.DataArray,
                        Y : xr.DataArray
                        ) -> tuple[xr.DataArray, xr.DataArray]:
        """Validates input data of fit function and returns it with the correct dimensions and labels.

        This method ensures that the input data to the fit function, X and Y, i.e. those used for training,
        have the expected dimension labels and sizes and returns them with the dimensions 
        ordered as (sample_name, feature_name). It also initializes the number of samples, and features.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
        Returns:
            tuple: A tuple (X, Y) where:
                X (DataArray): Input data for modality X ordered as (sample_name, featureX_name).
                Y (DataArray): Input data for modality Y ordered as (sample_name, featureY_name).
        """

        # Check thath X and Y have correct dimension labels
        validate_dimension_labels(X, Y, 
                                  sample_name=self.sample_name,
                                  featureX_name=self.featureX_name,
                                  featureY_name=self.featureY_name)
        
        # Bring dimensions to a pre-stablished order
        X = X.transpose(self.sample_name, self.featureX_name)
        Y = Y.transpose(self.sample_name, self.featureY_name)

        # Read dimensions sizes
        self.Ntx, self.Nx = X.shape
        self.Nty, self.Ny = Y.shape
        self.N_features = min(self.Nx, self.Ny)

        # Validate dimension sizes
        if self.Ntx <= self.Nty:
            raise ValueError(f"Number of X samples ({self.Ntx}) must be bigger than mumber of Y samples ({self.Nty})")
        if self.Nty < self.N_features:
            raise ValueError(f"Number of samples {self.Nty} should be bigger than number of features {self.N_features}")
        if self.N_components:
            if self.N_components > self.N_features:
                raise ValueError(f"Number of components {self.N_components} should be smaller than number of features {self.N_features}")
        else:
            self.N_components = self.N_features  # Catch None and 0 case

        return X, Y
    
    def validate_inputs_transform(self,
                        X : xr.DataArray,
                        Y : xr.DataArray
                        ) -> tuple[xr.DataArray, xr.DataArray]:
        """Validates that the to-be-transformed data have the expected dimension labels and sizes.

        This method ensures that X and Y have the same dimension labels and number of features
        than the ones used during training.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.
        Returns:
            tuple: A tuple (X, Y) where:
                X (DataArray): Input data for modality X ordered as (sample_name, featureX_name).
                Y (DataArray): Input data for modality Y ordered as (sample_name, featureY_name).
        """

        # Check thath X and Y have correct dimension labels
        validate_dimension_labels(X, Y, 
                                  sample_name=self.sample_name,
                                  featureX_name=self.featureX_name,
                                  featureY_name=self.featureY_name)
        
        # Bring dimensions to a pre-stablished order
        X = X.transpose(self.sample_name, self.featureX_name)
        Y = Y.transpose(self.sample_name, self.featureY_name)

        # Read dimensions sizes
        Ntx, Nx = X.shape
        Nty, Ny = Y.shape

        # Validate feature sizes
        if Ntx <= Nty:
            raise ValueError(f"Number of X samples ({Ntx}) must be bigger than mumber of Y samples ({Nty})")
        if Nx != self.Nx:
            raise ValueError(f"X should have {self.Nx} number of features, but {Nx} was found!")
        if Ny != self.Ny:
            raise ValueError(f"Y should have {self.Ny} number of features, but {Ny} was found!")

        return X, Y

    
    def convert_filters_to_DataArray(self, 
                                    Wx : np.ndarray,
                                    Wy : np.ndarray,
                                    Wt : np.ndarray,
                                    X : xr.DataArray,
                                    Y : xr.DataArray
                                    ) -> list[xr.DataArray, xr.DataArray, xr.DataArray]:
        """Convert filters Wx, Wy, Wt in numpy array format to DataArray with right dimensions and coordinates.
        
        Args:
            Wx (ndarray): Filter matrix for modality X with shape (Nx, N_components).
            Wy (ndarray): Filter matrix for modality Y with shape (Ny, N_components).
            Wt (ndarray): Filter matrix for time embedding with shape (N_shifts, N_components).
            X_features (DataArray): DataArray containing the features of modality X.
            Y_features (DataArray): DataArray containing the features of modality Y.

        Returns:
            tuple[DataArray, DataArray, DataArray]: A tuple containing the DataArray versions of 
            Wx, Wy, and Wt.
        """

        component_name_x = [f'Sx{i+1}' for i in range(self.N_components)]
        component_name_y = [f'Sy{i+1}' for i in range(self.N_components)]
        componentt_name = [f'St{i+1}' for i in range(self.N_components)]
        
        coords_x = {d: X[d] for d in X.dims if d != self.sample_name}
        coords_x[self.latent_featureX_name] = component_name_x
        Wx_xr = xr.DataArray(Wx, coords=coords_x)

        Wy_xr = xr.DataArray(Wy)
        coords_y = {d: Y[d] for d in Y.dims if d != self.sample_name}
        coords_y[self.latent_featureY_name] = component_name_y
        Wy_xr = xr.DataArray(Wy, coords=coords_y)

        Wt_xr = xr.DataArray(Wt, coords={'time_embedding': self.time_shifts,
                                         self.latent_featureT_name: componentt_name})
        
        return Wx_xr, Wy_xr, Wt_xr


    def estimate_optimal_shift(self):
        """Find optimal time shifts for X by looking for the largest Wt component.
        """

        wt_max_ndx = np.argmax(self.Wt.data, axis=0)
        self.optimal_shift = self.time_shifts[wt_max_ndx]

    def shift_by_optimal(self, 
                         X : xr.DataArray) -> xr.DataArray:
        """Shift X by optimal time shift using zero padding.
        """

        ti = X.time[0].data
        Nt = len(X.time)
        X_shifted = xr.zeros_like(X)
        for i in range(self.N_components):
            start = X.time.searchsorted(ti + self.optimal_shift[i])
            X_shifted[:Nt - start, i] = X[start:, i].data

        return X_shifted
    
def temporal_embedding(X : np.ndarray, 
                       time_shifts : np.ndarray, 
                       time : np.ndarray) -> np.ndarray:
    """Construct a time-embedded version of a matrix X.

    Args:
        X (ndarray): Matrix to embed in time.
        time_shifts (ndarray): Array of time shifts to consider.
        time (ndarray): Array of time points.
    Returns:
        ndarray: Time-embedded version of X.
    """

    # Convert to DataArray
    dimensions = ['time']
    if X.ndim > 1:
        dimensions += [f'aux{i+1}' for i in range(X.ndim - 1)]
    X = xr.DataArray(data=X, 
                     dims=dimensions, 
                     coords={'time': time})

    # Read and initial and final time
    ti = X.time[0].data
    Nt = len(X.time)
    X_emb_list = []
    for shift in time_shifts:
        X_emb = xr.zeros_like(X)
        start = X_emb.time.searchsorted(shift + ti)
        X_emb[start:] = X[:Nt - start].data
        # X_emb = X_emb.assign_coords({feature_name: new_features})
        X_emb_list.append(X_emb)
    
    # Stack over new dimension
    X_emb = np.stack(X_emb_list)
    
    return X_emb

def get_orthonormal_matrix(W : np.ndarray) -> np.ndarray:
    """Generate an orthonormal basis for an N-dimensional space
    where the columns of W are some of the basis vectors.
    
    Parameters:
        W (np.ndarray): A N x Nc array representing the given vectors.
    
    Returns:
        basis (np.ndarray): An N x N - Nc orthonormal basis matrix, 
        where the columns of W are not present.
    """
    N, Nc = W.shape
    # Normalize w
    W = np.stack([w / np.linalg.norm(w) for w in W.T]).T
    # Generate random vectors to form the rest of the basis
    random_vectors = np.random.randn(N, N - Nc)
    # Combine w with the random vectors
    B = np.column_stack([W, random_vectors])
    # Perform QR decomposition to orthogonalize
    B, _ = np.linalg.qr(B)

    # Ensure W is in the basis (numerical precision can cause slight misalignment)
    for w in W.T:
        # Find the column most aligned with w
        similarity = np.abs(B.T @ w)
        w_idx = np.argmax(similarity)
        
        # Remove w from B
        B = np.delete(B, w_idx, 1)
    
    return B
