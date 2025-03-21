"""Module for temporally embedded CCA-like models. The temporally embedded technique is based on :cite:t:`biesmann_temporal_2010`"""

import numpy as np
import xarray as xr

from utils_multimodal_models import validate_dimension_labels, validate_time_shifts, validate_l1_reg, validate_l2_reg

from cca_models import MultimodalSourceDecomposition, estimate_filters

class MultimodalSourceDecompositionWithTemporalEmbedding(MultimodalSourceDecomposition):
    """Class for decomposing multimodal data (X and Y) into latent sources using linear filters, using temporal embedding.

    This main class is inherited by other source decomposition methods, such as tCCA. 
    It implements methods to validate input dimensions, apply normalization, and transform 
    data from two modalities using filters learned during training. It assumes modality Y is delayed 
    with respect to modality X.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features
            between modalities.
        time_shifts (np.ndarray): Array with time shifts to be used for temporal embedding.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for convergence. 
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.
        shift_source (bool): Whether to shift the reconstructed sources by the optimal time shift found during training. 
            Defaults to True.
    """

    def __init__(self, 
                 N_components : int = None, 
                 time_shifts : np.ndarray = None,
                 max_iter : int = 100, 
                 tol : float = 1e-6, 
                 scale : bool = True,
                 shift_source = True):
        
        super(MultimodalSourceDecompositionWithTemporalEmbedding, self).__init__(
            N_components=N_components,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
        )

        # Sample name must be always time
        self.sample_name = 'time'

        # Time embedding attributes
        self.time_shifts = time_shifts
        self.N_shifts = len(time_shifts)
        self.optimal_shift = None
        self.shift_source = shift_source
    
    def estimate_optimal_shift(self, 
                               X_emb : xr.DataArray, 
                               Y : xr.DataArray):
        """Find optimal time shifts for X, one per component.
        
        It finds the optimal time shift for each component by looking for the time-shifted
        X that produces the biggest correlation between reconstructed sources sx and sy.

        Args:
            X_emb (DataArray): Time-embedded version of X with dimensions (time_shift, sample_name, featureX_name).
            Y (DataArray): Input data for modality Y with dimensions (sample_name, featureY_name).
        """
        
        # Normalize 
        X_emb, Y = self.normalization_transform(X_emb, Y)

        # Poject into each set of weights separately
        sx_list = [X_emb[i] @ self.Wx[i] for i in range(self.N_shifts)]
        sy = Y @ self.Wy
        
        # Iterate over components
        self.optimal_shift = np.zeros(self.N_components)
        for i in range(self.N_components):
            # Calculate correlations
            corrxy = [np.corrcoef(s[:, i], sy[:, i])[0, 1] for s in sx_list]

            # Find optimal lag
            max_corr_ndx = np.argmax(np.abs(corrxy))
            self.optimal_shift[i] = self.time_shifts[max_corr_ndx]
        
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

    def transform(self, 
                  X : xr.DataArray, 
                  Y : xr.DataArray
                  ) -> tuple[xr.DataArray, xr.DataArray]:
        """Apply the linear transformation on the input data using learnt filters.

        This method validates the dimension labels and sizes of the input data to ensure
        consistency with the training data, perform temporal embedding on X,
        applies normalization using the stored parametersand, and then projects the normalized data 
        onto a lower-dimensional space using the learned filters Wx and Wy. 
        It retrieves the transformed arrays, a.k.a reconstructed sources.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).

        Returns:
            tuple: A tuple (X_new, Y_new) where:
                X_new (DataArray): Transformed data for modality X.
                Y_new (DataArray): Transformed data for modality Y.
        """

        # Check dimension labels to be consistent with the ones used during training
        validate_dimension_labels(X, Y, 
                                  sample_name=self.sample_name,
                                  featureX_name=self.featureX_name,
                                  featureY_name=self.featureY_name)

        # Check that X and Y have the same labels and number of features as the training data
        X, Y = self.validate_inputs_transform(X, Y)

        # Construct a time-embedded version of X by concatenating time-shifted versions of it using times in time_shifts
        X = temporal_embedding(X, self.time_shifts)

        # Normalize
        X, Y = self.normalization_transform(X, Y)

        # Project
        X_new = X @ self.Wx
        Y_new = Y @ self.Wy
        
        # Shift X by optimal shift found during training
        if self.shift_source:
            X_new = self.shift_by_optimal(X_new)
            # Truncate to remove zero padding 
            minimal_shift = np.min(self.optimal_shift)
            X_new = X_new.loc[X_new.time <= X_new.time[-1].data - minimal_shift]
            Y_new = Y_new.loc[Y_new.time <= Y_new.time[-1].data - minimal_shift]

        return X_new, Y_new


class ElasticNetTCCA(MultimodalSourceDecompositionWithTemporalEmbedding):
    """Perform Elastic Net Canonical Correlation Analysis (CCA) between X_emb and Y.
         
    Apply temporally embedded CCA (tCCA) with L1 + L2 regularization, a.k.a elastic net. The algorithm finds sparse (L1) 
    and normalized (L2) vectors Wx, and Wy as the solution to the following constrained optimization problem:

    maximize    Wx^T Cxy Wy
    subject to  Wx^T Cx Wx = 1,  Wy^T Cy Wy = 1, 
                ||Wx||_1 <= c1x,  ||Wy||_1 <= c1y, 
                ||Wx||^2_2 <= c2x,  ||Wy||^2_2 <= c2y
    
    where Cx, Cy, and Cxy are the individual and cross-covariance matrices between X_emb and Y datasets, 
    and the last four constraints correspond to the standard L1-norm and L2-norm penalization terms. 
    c1x and c1y controls sparsity while c2x and c2y controls the magnitude of the vectors. PLS algorithms 
    are also captured by this algorithm by sending Cx and Cy to the identity matrices. 

    The temporally embedded matrix X_emb is constructed by concatenating time-shifted versions of the original X.
    Y is assumed to be the delayed signals with respect to X so time shifts are always positive.

    For the one-unit algorithm (sparse) SVD decomposition is performed on the whitened cross-covariance matrix 
    K = Cx^(-1/2) Cxy Cy^(-1/2) (reduced to K = Cxy for PLS), using the following standard alternating 
    power method (based on :cite:t:`parkhomenko_sparse_2009`):

    - Update u:
        1. u <- K * v
        2. u <- u / ||u||
        3. If L1:
            u <- SoftThresholding(u, lambda_u/2)
            u <- u / ||u||
    - Update v:
        1. v <- K^T * u
        2. v <- v / ||v||
        3. If L1:
            v <- SoftThresholding(v, lambda_v/2)
            v <- v / ||v||

    The resulting u and v are the leading left and right singular vectors of K which are nothing but individual components of the
    filters Wx and Wy. The softthresholding function bring some components to zero. If L2 regularization is used, prior to 
    computing K, Cx and Cy are shifted by Cx <- Cx + alpha_x I and Cy <- Cy + alpha_y I. 
    
    Multiple components are obtained via a deflation method, subtracting from K its 1-rank approximation on each iteration. 
    The returned vectors Wx and Wy are ordered in desceding order w.r.t. the singular values, which coincide with the canonical 
    correlations.
    
    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between modalities.
        l1_reg (float or list of floats): list containing lambda_u and lambda_v (see above). If a single float is provided,
        then lambda_u = lambda_v.
        l2_reg (float or list of floats): list containing alpha_x and alpha_y (see above). If a single float is provided,
        then alpha_x = alpha_y.
        time_shifts (np.ndarray): Array with time shifts to be used for temporal embedding.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.
        pls (bool): Whether to perform PLS regression. Defaults to False.
        shift_source (bool): Whether to shift the reconstructed sources by the optimal time shift found during training.

    Attributes:
        Ntx (int): Number of samples of modality X during training.
        Nty (int): Number of samples of modality Y during training.
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for X modality.
        latent_featureY_name (str): Label of transformed latent space dimension for Y modality.
        Wx (DataArray): Linear filters for dataset X with dimensions (featureX_name, latent_featureX_name)
        Wy (DataArray): Linear filters for dataset Y with dimensions (featureY_name, latent_featureY_name).
        optimal_shift (np.ndarray): Optimal time shift for each component.
        
    """
    def __init__(self, 
                 N_components: int = None, 
                 l1_reg: float | list[float, float] = 0,
                 l2_reg: float | list[float, float] = 0, 
                 time_shifts = None,
                 max_iter: int = 100, 
                 tol: float = 1e-6, 
                 scale: bool = True,
                 pls : bool = False,
                 shift_source = True):

        super(ElasticNetTCCA, self).__init__(
            N_components=N_components,
            time_shifts=time_shifts,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            shift_source=shift_source
        )

        # CCA or PLS
        self.pls = pls

        # Validate regularization parameters
        self.l1_reg = validate_l1_reg(l1_reg)
        self.l2_reg = validate_l2_reg(l2_reg)

    def fit(self, 
            X : xr.DataArray, 
            Y : xr.DataArray, 
            featureX_name : str = 'channel', 
            featureY_name : str = 'channel'):
        """Find the canonical vectors Wx, and Wy for the datasets X and Y.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            featureX_name (str, optional): Label for X-feature dimension, set to 'channel' by default.
            featureY_name (str, optional): Label for Y-feature dimension, set to 'channel' by default.
        """

        # Labels for feature directions
        self.featureX_name = featureX_name
        self.featureY_name = featureY_name

        # Validates input data and returns it ordered as (sample_name, feature_name)
        X, Y = self.validate_inputs_fit(X, Y)

        # Check right format for time shifts and return them in the right order and including zero shift
        self.time_shifts = validate_time_shifts(T=X.time[-1], time_shifts=self.time_shifts)
        self.N_shifts = len(self.time_shifts)

        # Construct a time-embedded version of X by concatenating time-shifted versions.
        X = temporal_embedding(X, self.time_shifts)
        
        # Center and diagonal-scale data
        X, Y = self.normalization_fit(X, Y)
        
        # Concatenate over the feature dimension
        X_concat = xr.concat(X, dim=self.featureX_name)

        Wx, Wy = estimate_filters(X_concat.data, 
                                  Y.data,
                                  N_components=self.N_components,
                                  l1_reg=self.l1_reg,
                                  l2_reg=self.l2_reg,
                                  pls=self.pls)

        # Split Wx into N_shifts components
        Wx = np.array([Wx[i*self.Nx:(i+1)*self.Nx] for i in range(self.N_shifts)])
        
        # Convert back to xarray with the right dimensions and coordinates
        self.Wx, self.Wy = self.convert_filters_to_DataArray(Wx, Wy, X, Y)

        # Estimate optimal time lag
        self.estimate_optimal_shift(X, Y)


class StructuredSparseTCCA(MultimodalSourceDecompositionWithTemporalEmbedding):

    """Perform structured sparse Canonical Correlation Analysis (ssCCA) between two datasets X_emb and Y.
         
    The sstCCA algorithm is a temporally embedded extension of :cite:t:`chen_structure-constrained_2013`, and it assumes the underlying 
    X and Y features are linked through a graph structure. It finds sparse (L1) vectors Wx, and Wy as the solution to the 
    following constrained optimization problem:

    maximize    Wx^T Cxy Wy
    subject to  Wx^T Cx Wx = 1,  Wy^T Cy Wy = 1, 
                ||Wx||_1 <= c1x,  ||Wy||_1 <= c1y, 
                Wx^T Lx Wx <= c2x,  Wy^T Ly Wy <= c2y
    
    where Cx, Cy, and Cxy are the individual and cross-covariance matrices between X_emb and Y datasets.
    The second constraint is the standard L1-norm penalization term, while the last constraint incorporates 
    local information of the spatial distribution of the features trough the Laplacian matrices Lx and Ly.
    These terms encaurage filter components that are linked on the graphical structure to have similar values, 
    making them to vary smoothly across the graph. The c1x and c1y controls sparsity while c2x and c2y controls
    the relative importante of the graph structure.

    The temporally embedded matrix X_emb is constructed by concatenating time-shifted versions of the original X.
    Y is assumed to be the delayed signals with respect to X so time shifts are always positive.

    For the one-unit algorithm, first Cx and Cy are shifted by Cx <- Cx + alpha_x Lx and Cy <- Cy + alpha_y Ly, and then
    SVD decomposition is performed on the whitened cross-covariance matrix K = Cx^(-1/2) Cxy Cy^(-1/2), 
    using the following standard alternating power method (based on :cite:t:`parkhomenko_sparse_2009`):

    - Update u:
        1. u <- K * v
        2. u <- u / ||u||
        3. If L1:
            u <- SoftThresholding(u, lambda_u/2)
            u <- u / ||u||
    - Update v:
        1. v <- K^T * u
        2. v <- v / ||v||
        3. If L1:
            v <- SoftThresholding(v, lambda_v/2)
            v <- v / ||v||

    The resulting u and v are the leading left and right singular vectors of K which are nothing but individual components of the
    filters Wx and Wy. The softthresholding function bring some components to zero.
    
    Multiple components are obtained via a deflation method, subtracting from K its 1-rank approximation on each iteration. 
    The returned vectors Wx and Wy are ordered in desceding order w.r.t. the singular values, which coincide with the canonical 
    correlations.
    
    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between modalities.
        Lx (ndarray): Laplacian matrix for modality X.
        Ly (ndarray): Laplacian matrix for modality Y.
        time_shifts (np.ndarray): Array with time shifts to be used for temporal embedding.
        l1_reg (float or list of floats): list containing lambda_u and lambda_v (see above). If a single float is provided,
        then lambda_u = lambda_v.
        l2_reg (float or list of floats): list containing alpha_x and alpha_y (see above). If a single float is provided,
        then alpha_x = alpha_y.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.
        pls (bool): Whether to perform PLS regression. Defaults to False.
        shift_source (bool): Whether to shift the reconstructed sources by the optimal time shift found during training.

    Attributes:
        Ntx (int): Number of samples of modality X during training.
        Nty (int): Number of samples of modality Y during training.
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for X modality.
        latent_featureY_name (str): Label of transformed latent space dimension for Y modality.
        Wx (DataArray): Linear filters for dataset X with dimensions (featureX_name, latent_featureX_name)
        Wy (DataArray): Linear filters for dataset Y with dimensions (featureY_name, latent_featureY_name).
        optimal_shift (np.ndarray): Optimal time shift for each component.
    """               


    def __init__(self, 
                 N_components : int = None, 
                 Lx = None,
                 Ly = None,
                 time_shifts = None,
                 l1_reg : float | list[float, float] = 0,
                 l2_reg : float | list[float, float] = 0,
                 max_iter : int = 100, 
                 tol : float = 1e-6, 
                 scale : bool = True,
                 shift_source : bool = True,
                 pls : bool = False):
        
        super(StructuredSparseTCCA, self).__init__(
            N_components=N_components,
            time_shifts=time_shifts,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            shift_source=shift_source
        )

        # CCA or PLS
        self.pls = pls

        # Validate regularization parameters
        self.l1_reg = validate_l1_reg(l1_reg)
        self.l2_reg = validate_l2_reg(l2_reg)
        self.Lx = Lx
        self.Ly = Ly

    def fit(self, 
            X : xr.DataArray, 
            Y : xr.DataArray, 
            featureX_name : str = 'channel', 
            featureY_name : str = 'channel'):
        """Find the canonical vectors Wx, and Wy for the datasets X and Y.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            featureX_name (str, optional): Label for X-feature dimension, set to 'channel' by default.
            featureY_name (str, optional): Label for Y-feature dimension, set to 'channel' by default.
        """

        # Labels for sample and feature directions
        self.featureX_name = featureX_name
        self.featureY_name = featureY_name
    
        # Validates input data and returns it ordered as (sample_name, feature_name)
        X, Y = self.validate_inputs_fit(X, Y)

        # Check right format for time shifts and return them in the right order and including tau=0
        self.time_shifts = validate_time_shifts(T=X.time[-1], time_shifts=self.time_shifts)
        self.N_shifts = len(self.time_shifts)

        # Construct a time-embedded version of X by concatenating time-shifted versions of it using times in time_shifts
        X = temporal_embedding(X, self.time_shifts)

        # Copy Laplace matrix over diagonals
        new_Lx = np.zeros([self.Nx*self.N_shifts, self.Nx*self.N_shifts])
        for i in range(self.N_shifts):
            new_Lx[i*self.Nx:(i+1)*self.Nx,i*self.Nx:(i+1)*self.Nx] = self.Lx
        
        # Center and diagonal-scale data
        X, Y = self.normalization_fit(X, Y)

        # Concatenate over the feature dimension
        X_concat = xr.concat(X, dim=self.featureX_name)

        Wx, Wy = estimate_filters(X_concat.data, 
                                  Y.data, 
                                  N_components=self.N_components, 
                                  l1_reg=self.l1_reg, 
                                  l2_reg=self.l2_reg,
                                  Lx=new_Lx,
                                  Ly=self.Ly,
                                  pls=self.pls)

        # Split Wx into N_shifts components
        Wx = np.array([Wx[i*self.Nx:(i+1)*self.Nx] for i in range(self.N_shifts)])
        
        # Convert back to xarray with the right dimensions and coordinates
        self.Wx, self.Wy = self.convert_filters_to_DataArray(Wx, Wy, X, Y)

        # Estimate optimal time lag
        self.estimate_optimal_shift(X, Y)


class tCCA(ElasticNetTCCA):
    """Perform tCCA between two datasets X and Y.

    This algorithm is a particular case of the one implemented in the ElasticNetTCCA class. See there for 
    a detailed explanation of the algorithm.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between modalities.
        time_shifts (np.ndarray): Array with time shifts to be used for temporal embedding.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.
        shift_source (bool): Whether to shift the reconstructed sources by the optimal time shift found during training.

    Attributes:
        Ntx (int): Number of samples of modality X during training.
        Nty (int): Number of samples of modality Y during training.
        Nx (int): Number of features of modality X.
        Ny (int): Number of features of modality Y.
        x_mean (DataArray): Mean of modality X during training.
        y_mean (DataArray): Mean of modality Y during training.
        x_std (DataArray): Standard deviation of modality X during training.
        y_std (DataArray): Standard deviation of modality Y during training.
        latent_featureX_name (str): Label of transformed latent space dimension for X modality.
        latent_featureY_name (str): Label of transformed latent space dimension for Y modality.
        Wx (DataArray): Linear filters for dataset X with dimensions (featureX_name, latent_featureX_name)
        Wy (DataArray): Linear filters for dataset Y with dimensions (featureY_name, latent_featureY_name).
        optimal_shift (np.ndarray): Optimal time shift for each component.
    """

    def __init__(self, 
                N_components : int = None,
                time_shifts : np.ndarray = None,
                max_iter : int = 100, 
                tol : float = 1e-6, 
                scale : bool = True,
                shift_source = True):
        
        super(tCCA, self).__init__(
            N_components=N_components,
            time_shifts=time_shifts,
            l1_reg=0,
            l2_reg=0,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            pls=False,
            shift_source=shift_source
        )


def temporal_embedding(X : xr.DataArray, 
                       time_shifts : np.ndarray
                       ) -> xr.DataArray:
    """Construct a time-embedded version of a matrix X. 
    
    If X has shape T x N, the embedding matrix X_emb has shape P x T x N, where P is the number 
    of time shifts in time_shifts. The embedding matrix is built by concatenating time-shift versions
    of the original matrix using the shifts inside time_shifts. Zero padding is used at the beggining
    of each time-shifted copy to preserve the original length of time direction.

    Args:
        X (DataArray): Input data with at least a time dimension.
        time_shifts (np.ndarray): Array with time shifts to be used for temporal embedding.
    Returns:
        DataArray: Time-embedded version of X with dimensions (time_shift, time, ...).
    """

    # Bring dimensions to a pre-stablished order
    X = X.transpose('time', ...)
    
    # Read and initial and final time
    ti = X.time[0].data
    Nt = len(X.time)

    # Build temporal embeddings one by one and append to list
    X_emb_list = []
    for shift in time_shifts:
        X_emb = xr.zeros_like(X)
        start = X_emb.time.searchsorted(shift + ti)
        X_emb[start:] = X[:Nt - start].data
        X_emb_list.append(X_emb)

    # Concatenate over feature dimension (and assure dimensions are again in the right order)
    X_emb_xr = xr.concat(X_emb_list, dim='time_shift').assign_coords({'time_shift': time_shifts})
    X_emb_xr = X_emb_xr.transpose('time_shift', ...)
    
    return X_emb_xr
