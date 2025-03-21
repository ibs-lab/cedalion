"""Module for CCA-like models"""

import numpy as np
import xarray as xr

from utils_multimodal_models import validate_dimension_labels, validate_dimension_sizes, standardize, validate_l1_reg, validate_l2_reg

class MultimodalSourceDecomposition():
    """Class for decomposing multimodal data, X and Y, into latent sources using linear filters.

    This main class is inherited by other source decomposition methods, such as ElasticNetCCA, ssCCA, and PLS. 
    It implements methods to validate input dimensions, apply normalization, and transform data from 
    two modalities using filters learned during training.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features
            between modalities.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.
    
    """

    def __init__(self, 
                 N_components : int = None, 
                 max_iter : int = 100, 
                 tol : float = 1e-6, 
                 scale : bool = True):

        # General parameters
        self.N_components = N_components
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

        # Normalization parameters
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None

        # Filters
        self.Wx = None
        self.Wy = None

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
        self.N_components = validate_dimension_sizes(self.Ntx, self.Nty, self.N_features, self.N_components)

        return X, Y


    def validate_inputs_transform(self,
                        X : xr.DataArray,
                        Y : xr.DataArray
                        ) -> tuple[xr.DataArray, xr.DataArray]:
        """Validates that the to-be-transformed data have the expected dimension labels and sizes.

        This method ensures that X and Y have the same dimension labels and number of features
        than the ones used during training. The number of time points, however, can be different.

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
        _, Nx = X.shape
        _, Ny = Y.shape

        # Validate feature sizes
        if Nx != self.Nx:
            raise ValueError(f"X should have {self.Nx} number of features, but {Nx} was found!")
        if Ny != self.Ny:
            raise ValueError(f"Y should have {self.Ny} number of features, but {Ny} was found!")

        return X, Y


    def normalization_fit(self, 
                          X : xr.DataArray, 
                          Y : xr.DataArray
                          ) -> tuple[xr.DataArray, xr.DataArray]:
        """Normalize input data and save normalization parameters (mean and std) for later.

        This method centers and scales the data for both modalities along the sample
        dimension. It computes the mean and standard deviation for X and Y using the
        provided standardization function, updating the corresponding class attributes.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.

        Returns:
            tuple: A tuple (X, Y) of standardized data arrays for modalities X and Y.
        """
        # Center and diagonal-scale data
        X, self.x_mean, self.x_std = standardize(X, dim=self.sample_name, scale=self.scale)
        Y, self.y_mean, self.y_std = standardize(Y, dim=self.sample_name, scale=self.scale)

        return X, Y
    
    def normalization_transform(self, 
                          X : xr.DataArray, 
                          Y : xr.DataArray
                          ) -> tuple[xr.DataArray, xr.DataArray]:
        """Applies normalization input data using trained parameters.

        This method standardizes the input data arrays X and Y using the normalization
        parameters (mean and standard deviation) computed during the fitting process.

        Args:
            X (DataArray): Input data for modality X.
            Y (DataArray): Input data for modality Y.

        Returns:
            tuple: A tuple (X, Y) of normalized data arrays.
        """
        # Center and diagonal-scale data
        X = (X - self.x_mean) / self.x_std
        Y = (Y - self.y_mean) / self.y_std

        return X, Y
    
    def convert_filters_to_DataArray(self, 
                                    Wx : np.ndarray,
                                    Wy : np.ndarray,
                                    X : xr.DataArray,
                                    Y : xr.DataArray
                                    ) -> list[xr.DataArray, xr.DataArray]:
        """Convert filters Wx and Wy in numpy array format to DataArray with right dimensions and coordinates.
        
        Args:
            Wx (ndarray): Filter matrix for modality X with shape (Nx, N_components).
            Wy (ndarray): Filter matrix for modality Y with shape (Ny, N_components).
            X_features (DataArray): DataArray containing the features of modality X.
            Y_features (DataArray): DataArray containing the features of modality Y.

        Returns:
            tuple[DataArray, DataArray]: A tuple (Wx_xr, Wy_xr) containing the DataArray versions of 
            Wx and Wy respectively.
        """

        component_name_x = [f'Sx{i+1}' for i in range(self.N_components)]
        component_name_y = [f'Sy{i+1}' for i in range(self.N_components)]
        
        coords_x = {d: X[d] for d in X.dims if d != self.sample_name}
        coords_x[self.latent_featureX_name] = component_name_x
        Wx_xr = xr.DataArray(Wx, coords=coords_x)

        Wy_xr = xr.DataArray(Wy)
        coords_y = {d: Y[d] for d in Y.dims if d != self.sample_name}
        coords_y[self.latent_featureY_name] = component_name_y
        Wy_xr = xr.DataArray(Wy, coords=coords_y)

        return Wx_xr, Wy_xr


    def transform(self, 
                  X : xr.DataArray, 
                  Y : xr.DataArray
                  ) -> tuple[xr.DataArray, xr.DataArray]:
        """Apply the linear transformation on the input data using learnt filters.

        This method validates the dimension labels and sizes of the input data to ensure
        consistency with the training data, applies normalization using the stored parameters,
        and then projects the normalized data onto a lower-dimensional space using the learned
        filters Wx and Wy. It retrieves the transformed arrays, a.k.a reconstructed sources.

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

        # Check that X and Y have the same labels and number of features as the training data
        X, Y = self.validate_inputs_transform(X, Y)
        
        # Normalize
        X, Y = self.normalization_transform(X, Y)

        # Project
        X_new = X @ self.Wx
        Y_new = Y @ self.Wy

        return X_new, Y_new


class ElasticNetCCA(MultimodalSourceDecomposition):

    """Perform Elastic Net Canonical Correlation Analysis (CCA) between two datasets X and Y.
         
    Apply CCA with L1 + L2 regularization, a.k.a elastic net. The algorithm finds sparse (L1) 
    and normalized (L2) vectors Wx, and Wy as the solution to the following constrained optimization problem:

    maximize    Wx^T Cxy Wy
    subject to  Wx^T Cx Wx = 1,  Wy^T Cy Wy = 1, 
                ||Wx||_1 <= c1x,  ||Wy||_1 <= c1y, 
                ||Wx||^2_2 <= c2x,  ||Wy||^2_2 <= c2y
    
    where Cx, Cy, and Cxy are the individual and cross-covariance matrices between X and Y datasets, 
    and the last four constraints correspond to the standard L1-norm and L2-norm penalization terms. 
    c1x and c1y controls sparsity while c2x and c2y controls the magnitude of the vectors. PLS algorithms 
    are also captured by this algorithm by sending Cx and Cy to the identity matrices.

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
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.
        pls (bool): Whether to perform PLS regression. Defaults to False.

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
        
    """               


    def __init__(self, 
                 N_components : int = None, 
                 l1_reg : float | list[float, float] = 0,
                 l2_reg : float | list[float, float] = 0,
                 max_iter : int = 100, 
                 tol : float = 1e-6, 
                 scale : bool = True,
                 pls : bool = False):
        
        super(ElasticNetCCA, self).__init__(
            N_components=N_components,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
        )

        # CCA or PLS
        self.pls = pls
        # Validate regularization parameters
        self.l1_reg = validate_l1_reg(l1_reg)
        self.l2_reg = validate_l2_reg(l2_reg)

    def fit(self, 
            X : xr.DataArray, 
            Y :  xr.DataArray, 
            sample_name : str = 'time', 
            featureX_name : str = 'channel', 
            featureY_name : str = 'channel'):
        """Find the canonical vectors Wx, and Wy for the datasets X and Y.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            sample_name (str, optional): Label for sample dimension, set to 'time' by default.
            featureX_name (str, optional): Label for X-feature dimension, set to 'channel' by default.
            featureY_name (str, optional): Label for Y-feature dimension, set to 'channel' by default.
        """

        # Labels for sample and feature directions
        self.sample_name = sample_name
        self.featureX_name = featureX_name
        self.featureY_name = featureY_name

        # Validates input data and returns them ordered as (sample_name, feature_name)
        X, Y = self.validate_inputs_fit(X, Y)
        
        # Center and diagonal-scale data
        X, Y = self.normalization_fit(X, Y)

        # Run main algorithm for filter estimation
        Wx, Wy = estimate_filters(X.data, 
                                  Y.data, 
                                  N_components=self.N_components, 
                                  l1_reg=self.l1_reg, 
                                  l2_reg=self.l2_reg,
                                  pls=self.pls)

        # Convert back to xarray with the right dimensions and coordinates
        self.Wx, self.Wy = self.convert_filters_to_DataArray(Wx, Wy, X, Y)


class StructuredSparseCCA(MultimodalSourceDecomposition):

    """Perform structured sparse Canonical Correlation Analysis (ssCCA) between two datasets X and Y.
         
    The ssCCA algorithm is based on :cite:t:`chen_structure-constrained_2013` and it assumes the underlying X and Y features 
    are linked through a graph structure. It finds sparse (L1) vectors Wx, and Wy as the solution to the 
    following constrained optimization problem:

    maximize    Wx^T Cxy Wy
    subject to  Wx^T Cx Wx = 1,  Wy^T Cy Wy = 1, 
                ||Wx||_1 <= c1x,  ||Wy||_1 <= c1y, 
                Wx^T Lx Wx <= c2x,  Wy^T Ly Wy <= c2y
    
    where Cx, Cy, and Cxy are the individual and cross-covariance matrices between X and Y datasets.
    The second constraint is the standard L1-norm penalization term, while the last constraint incorporates 
    local information of the spatial distribution of the features trough the Laplacian matrices Lx and Ly.
    These terms encaurage filter components that are linked on the graphical structure to have similar values, 
    making them to vary smoothly across the graph. The c1x and c1y controls sparsity while c2x and c2y controls
    the relative importante of the graph structure.

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
        l1_reg (float or list of floats): list containing lambda_u and lambda_v (see above). If a single float is provided,
        then lambda_u = lambda_v.
        l2_reg (float or list of floats): list containing alpha_x and alpha_y (see above). If a single float is provided,
        then alpha_x = alpha_y.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.
        pls (bool): Whether to perform PLS regression. Defaults to False.

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
    """               


    def __init__(self, 
                 N_components : int = None, 
                 Lx : np.ndarray = None,
                 Ly : np.ndarray = None,
                 l1_reg : float | list[float, float] = 0,
                 l2_reg : float | list[float, float] = 0,
                 max_iter : int = 100, 
                 tol : float = 1e-6, 
                 scale : bool = True,
                 pls : bool = False):
        
        super(StructuredSparseCCA, self).__init__(
            N_components=N_components,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
        )

        # Validate regularization parameters
        self.l1_reg = validate_l1_reg(l1_reg)
        self.l2_reg = validate_l2_reg(l2_reg)
        self.Lx = Lx
        self.Ly = Ly
        self.pls = pls

    def fit(self, 
            X : xr.DataArray, 
            Y :  xr.DataArray, 
            sample_name : str = 'time', 
            featureX_name : str = 'channel', 
            featureY_name : str = 'channel'):
        """Find the canonical vectors Wx, and Wy for the datasets X and Y.

        Args:
            X (DataArray): Input data for modality X. Expected to have dimensions
                (sample_name, featureX_name).
            Y (DataArray): Input data for modality Y. Expected to have dimensions
                (sample_name, featureY_name).
            sample_name (str, optional): Label for sample dimension, set to 'time' by default.
            featureX_name (str, optional): Label for X-feature dimension, set to 'channel' by default.
            featureY_name (str, optional): Label for Y-feature dimension, set to 'channel' by default.
        """

        # Labels for sample and feature directions
        self.sample_name = sample_name
        self.featureX_name = featureX_name
        self.featureY_name = featureY_name

        # Validates input data and returns it ordered as (sample_name, feature_name)
        X, Y = self.validate_inputs_fit(X, Y)
        
        # Center and diagonal-scale data
        X, Y = self.normalization_fit(X, Y)

        Wx, Wy = estimate_filters(X.data, 
                                  Y.data, 
                                  N_components=self.N_components, 
                                  l1_reg=self.l1_reg, 
                                  l2_reg=self.l2_reg,
                                  Lx=self.Lx,
                                  Ly=self.Ly,
                                  pls=self.pls)

        # Convert back to xarray with the right dimensions and coordinates
        self.Wx, self.Wy = self.convert_filters_to_DataArray(Wx, Wy, X, Y)


class RidgeCCA(ElasticNetCCA):
    """Perform CCA between two datasets X and Y with L2 regularization, a.k.a ridge CCA. 
    
    This algorithm is a particular case of the one implemented in the ElasticNetCCA class. See there for 
    a detailed explanation of the algorithm.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between modalities.
        l2_reg (float or list of floats): list containing alpha_x and alpha_y (see above). If a single float is provided,
        then alpha_x = alpha_y.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.

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
    """
    
    def __init__(self, 
                N_components : int = None,
                l2_reg : float | list[float, float] = 0, 
                max_iter : int = 100, 
                tol : float = 1e-6, 
                scale : bool = True):
        
        super(RidgeCCA, self).__init__(
            N_components=N_components,
            l1_reg=0,
            l2_reg=l2_reg,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            pls=False
        )


class SparseCCA(ElasticNetCCA):
    """Perform Sparse CCA between two datasets X and Y with L1 regularization, a.k.a sparse CCA, based on :cite:t:`parkhomenko_sparse_2009`.
    
    This algorithm is a particular case of the one implemented in the ElasticNetCCA class. See there for 
    a detailed explanation of the algorithm.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between modalities.
        l1_reg (float or list of floats): list containing lambda_u and lambda_v (see above). If a single float is provided,
        then lambda_u = lambda_v.
        max_iter (int): Maximum number of iterations for the algorithm. 
        tol (float): Tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.

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
    """
    
    def __init__(self, 
                N_components : int = None, 
                l1_reg : float | list[float, float] = 0,
                max_iter : int = 100, 
                tol : float = 1e-6, 
                scale : bool = True):

        super(SparseCCA, self).__init__(
            N_components=N_components,
            l1_reg=l1_reg,
            l2_reg=0,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            pls=False
        )


class CCA(ElasticNetCCA):
    """Perform CCA between two datasets X and Y.

    This algorithm is a particular case of the one implemented in the ElasticNetCCA class. See there for 
    a detailed explanation of the algorithm.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between modalities.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.

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
    """

    def __init__(self, 
                N_components: int = None,
                max_iter: int = 100, 
                tol: float = 1e-6, 
                scale: bool = True):
        
        super(CCA, self).__init__(
            N_components=N_components,
            l1_reg=0,
            l2_reg=0,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            pls=False
        )


class SparsePLS(ElasticNetCCA):
    """Perform Partial Least Squares (PLS) between two datasets X and Y with L1 regularization, a.k.a sparse PLS, 
    based on a combination from :cite:t:`parkhomenko_sparse_2009` and :cite:t:`witten_penalized_2009`.

    In Witten's paper, the algorithm is presented as a particular case of their Penalized Matrix Decomposition (PMD) method, 
    called PMD(L1, L1) or Sparse CCA. However, the latter name is misleading since in this problem we use identity matrices 
    for the L2-norm constraints, rather than correlation matrices. That difference makes the method truly a SparsePLS one. 
    Here, Witten's method is modified by adding normalization on each iteration and dividing L1 parameters by 2.
    
    This algorithm is a particular case of the one implemented in the ElasticNetCCA class. See there for 
    a detailed explanation of the algorithm.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between modalities.
        l1_reg (float or list of floats): list containing lambda_u and lambda_v (see above). 
        If a single float is provided, then lambda_u = lambda_v.
        max_iter (int): Maximum number of iterations for the algorithm.
        tol (float): Tolerance for convergence.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.

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
    """
    
    def __init__(self, 
                N_components: int = None, 
                l1_reg: float | list[float, float] = 0,
                max_iter: int = 100, 
                tol: float = 1e-6, 
                scale: bool = True):
    

        super(SparsePLS).__init__(
            N_components=N_components,
            l1_reg=l1_reg,
            l2_reg=0,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
            pls=True
        )


class PLS(SparsePLS):
     """Perform PLS between two datasets X and Y. This algorithm is a particular case of the one implemented in the SparsePLS class
    when no penalty is imposed. See there for a detailed explanation of the algorithm.

    Args:
        N_components (int, optional): Number of components to extract. If None,
            the number of components is set to the minimum number of features between modalities.
        max_iter (int): Maximum number of iterations for the algorithm. Defaults to 100.
        tol (float): Tolerance for convergence. Defaults to 1e-6.
        scale (bool): Whether to scale the data during normalization to unit variance. Defaults to True.

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
    """
     def __init__(self, 
                N_components: int = None,
                max_iter: int = 100, 
                tol: float = 1e-6, 
                scale: bool = True):

        super().__init__(
            N_components=N_components,
            l1_reg=0,
            max_iter=max_iter,
            tol=tol,
            scale=scale,
        )


def estimate_filters(X : np.ndarray, 
                     Y : np.ndarray,
                     N_components : int,
                     l1_reg : list[float, float] = 0,
                     l2_reg : list[float, float] = 0,
                     Lx : np.ndarray = None,
                     Ly : np.ndarray = None,
                     pls : bool = False
                     ) -> tuple[np.ndarray, np.ndarray]:
        """"Estimate the canonical vectors Wx, and Wy for the datasets X and Y.

        Main function that estimates the canonical vectors Wx, and Wy for the datasets X and Y using an 
        Elastic Net CCA algorithm with the option of incorporating Laplace matrices, Lx and Ly that 
        transforms the algorithm into a structured sparse CCA.
        It assumes X and Y have shapes (samples, features).

        Args:
            X (ndarray): Input data for modality X with shape (Nt, Nx).
            Y (ndarray): Input data for modality Y with shape (Nt, Ny).
            N_components (int): Number of components to extract.
            l1_reg (list of floats): list containing lambda_u and lambda_v.
            l2_reg (list of floats): list containing alpha_x and alpha_y.
            Lx (ndarray, optional): Laplacian matrix for modality X. Defaults to None.
            Ly (ndarray, optional): Laplacian matrix for modality Y. Defaults to None.
            pls (bool, optional): Whether to perform PLS regression. Defaults to False.
            
        Returns:
            tuple: A tuple (Wx, Wy) with the canonical vectors for X and Y, ordered by descending singular values.
        """

        Nt, Nx = X.shape
        Nt, Ny = Y.shape

        # Covariance matrices
        Cxy = (X.T @ Y) / (Nt - 1)
        
        # CCA or PLS
        if pls:
            K = Cxy
        else:
            Cxx = (X.T @ X) / (Nt - 1)
            Cyy = (Y.T @ Y) / (Nt - 1)

            # Add L2 penalty
            if l2_reg[0]:
                Lx = Lx if Lx is not None else np.eye(Nx)
                Cxx += l2_reg[0] * Lx
            if l2_reg[1]:
                Ly = Ly if Ly is not None else np.eye(Ny)
                Cyy += l2_reg[1] * Ly
            
            # Get inverse square root from eigenvalue decomposition
            Cxx_inv_sqrt = inv_sqrt_cov(Cxx)
            Cyy_inv_sqrt = inv_sqrt_cov(Cyy)
        
            # Matrix to peform SVD in the optimization problem 
            K = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
        
        # Perform SVD + deflation via rank-1 approximation subtraction
        Wx, _, Wy = get_singular_vectors(K, N_components, l1_reg)
        
        # Project back to original unwhitened space
        if not pls:
            Wx = Cxx_inv_sqrt @ Wx
            Wy = Cyy_inv_sqrt @ Wy

        return Wx, Wy

def inv_sqrt_cov(C : np.ndarray, 
                 eps : float = 1e-10
                 ) -> np.ndarray:
    """Compute the inverse square root of a covariance matrix C.
        
    Given a (symmetric) covariance matrix C, it computes C^{-1/2} = U Lambda^{-1/2} U^T
    using eigen-decomposition and clipping the inverse diagonal entries to eps to avoid
    division by zero and instabilities.

    Args:
        C (ndarray): Convariance matrix. Expected to be square and symmetric.
        eps (float, optional): Small value to avoid division by zero during inversion.

    Returns:
        Inverse square root of C of the same size as input matrix.
    """
    
    # Perform eigenvalue decomposition
    eigvals, U = np.linalg.eigh(C)
    eigvals_reg = np.clip(eigvals, eps, None)  # Avoid zero values

    # Inverse square root
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals_reg))
    C_inv_sqrt = U @ D_inv_sqrt @ U.T
    
    return C_inv_sqrt

def get_singular_vectors(X : np.ndarray, 
                         N_components : int, 
                         l1_reg : float | list[float, float] = 0,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts the top singular vectors from X using an iterative power method.

    The function iteratively extracts one sparse singular component at a time. On each iteration,
    it computes the leading singular pair using an alternating power method, subtracts the rank-1 
    approximation from X, and stores the component. Sparsity is enforced via L1 regularization 
    when l1_reg > 0.

    Args:
        X (ndarray): Input matrix of shape (M, N) from which singular vectors are extracted.
        N_components (int): Number of singular components to extract.
        l1_reg (float or list of floats, optional): Regularization parameter for L1 sparsity. If scalar, the 
            same value is applied for both u and v. Defaults to 0 (no sparsity).

    Returns:
        tuple:
            U (ndarray): Matrix of left singular vectors with shape (m, N_components).
            S (ndarray): Array of singular values in the diagonal with length N_components.
            V (ndarray): Matrix of right singular vectors with shape (n, N_components).
    """

    # Store multicomponents
    U = np.zeros([X.shape[0], N_components])
    V = np.zeros([X.shape[1], N_components])
    S = np.zeros(N_components)

    X_new = X.copy()
    for k in range(N_components):
        # Apply one-unit algorithm
        u, s, v = leading_singular_pair_power_method(X_new, l1_reg)
        # Substract rank-1 approximation
        X_rank1 = s * u @ v.T
        X_new -= X_rank1
        # Store component
        U[:, k] = u[:, 0]
        V[:, k] = v[:, 0]
        S[k] = s 

    return U, S, V

def leading_singular_pair_power_method(X : np.ndarray, 
                                       l1_reg : float | list[float, float] = 0, 
                                       max_iter : int = 1000, 
                                       tol : float = 1e-6
                                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the leading (sparse) singular vector pair and value (u, sigma, v) of a matrix X
    using an alternating power method.

    The method alternates between updating the left singular vector (u) and the right singular vector (v)
    until convergence following the rules:
    
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

    Sparsity is enforced via soft-thresholding if the corresponding regularization parameters
    (lambda_u and lambda_v) encoded in l1_reg are set to a nonzero value. 

    Args:
        X (ndarray): Input matrix of shape (m, n).
        l1_reg (int, float, or list, optional): L1 regularization parameter(s) for sparsity. If a scalar, 
            the same value is applied to both u and v; if a list of two values, the first is used for u and 
            the second for v. Defaults to 0 (no sparsity).
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        tol (float, optional): Convergence tolerance. Defaults to 1e-6.

    Returns:
        tuple:
            u (np.ndarray): Leading left singular vector of shape (m, 1).
            sigma (float): Leading singular value.
            v (np.ndarray): Leading right singular vector of shape (n, 1).
    """

    m, n = X.shape

    # Validate regularization parameter
    if isinstance(l1_reg, float) or isinstance(l1_reg, int):
        # Assume same parameters for both datasets
        l1_reg = [l1_reg, l1_reg]
    elif isinstance(l1_reg, list) and len(l1_reg)==2:
        l1_reg = l1_reg
    else:
        raise ValueError(f"Wrong format for l1_reg. Got {l1_reg}, but it should be a scalar, or a 2D list.")
    
    # Split parameters
    lambda_u, lambda_v = l1_reg

    # Random initialization
    v = np.random.rand(n)
    v /= np.linalg.norm(v)
    u = np.random.rand(m)
    u /= np.linalg.norm(u)

    for i in range(max_iter):

        u_old = u.copy()
        v_old = v.copy()

        # Update u = X v / ||X v||
        u = X @ v
        u /= np.linalg.norm(u)
        # Apply sparsity contraint
        if lambda_u:
            u = soft_threshold(u, lambda_u/2)
            u /= np.linalg.norm(u)
        
        # Update v = X^T u / ||X^T u||
        v = X.T @ u
        v /= np.linalg.norm(v)

        # Apply sparsity contraint
        if lambda_v:
            v = soft_threshold(v, lambda_v/2)
            v /= np.linalg.norm(v)
        
        # Check convergence: if the change in v is small, break.
        if np.linalg.norm(v - v_old) < tol and np.linalg.norm(u - u_old) < tol :
            # print(f'Converged at iteration {i+1}')
            break

    # After convergence, compute the singular value s = ||X v||
    Xv = X @ v
    s = np.linalg.norm(Xv)

    return u.reshape(-1, 1), s, v.reshape(-1, 1)

def soft_threshold(x : np.ndarray, 
                   thresh : float
                   ) -> np.ndarray:
    """Apply soft thresholding elementwise to an array X.
    """
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)
