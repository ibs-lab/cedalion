import numpy as np
from scipy.linalg import eigh
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from typing import Optional
import xarray as xr

class SPoC():
    """ Implements the Source Power Co-modulation (SPoC_lambda) algorithm based on :cite:t:`BIBTEXLABEL`.

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


class mSPoC():
    """ Implements the multimodal Source Power Co-modulation (mSPoC) algorithm based on :cite:t:`BIBTEXLABEL`.

    Given two vector-valued time series x(t), and y(t), mSPoC finds component pairs Sx = Wx.T @ x,
    and Sy = Wy.T @, such that the covariance between the temporally-embedded bandpower of Sx and the time course of Sy
    is maximized. The solution to that optimization problem is captured by the spatial (Wx, Wy), and temporal (Wt) filters.

    Assumptions:
        x(t) is of shape Nx x Nt, where Nx is the number of channels and Nt the 
        number of time points, and it is band-pass filtered in the frequency band 
        of interest. Bandpower of Sx is then approximated by its variance within epochs. 
        y(t) is of shape Ny x Ne, and Ne < Nt. Both signals are mean-free and temporally aligned.    
    
    Args:
        n_lags (int): Number of time lags for temporal embedding.
        n_components (int): Number of components/filter/eigenvalues the algorithm will find.
    """

    def __init__(self, 
                 n_lags: Optional[int] = 0, 
                 n_comp: Optional[int] = None):

        self.n_lags = n_lags + 1  # Includes 0 
        self.n_comp = n_comp

        # Time lags, including t_0=0 (no lag)
        self.t_lags = np.arange(0, self.n_lags)

        # Spatial and temporal filters (initialized after calling the fit function)
        self.wx = None
        self.wy = None
        self.wt = None

        # Number of channels
        self.Nx = None
        self.Ny = None

    def fit(self, 
            x: cdt.NDTimeSeries, 
            y: cdt.NDTimeSeries, 
            N_restarts: int = 2, 
            max_iter: int = 200, 
            tol: float = 1e-5) -> float:
        """Train mSPoC model on the x, y dataset

        Implement the pseudo-code of Algorithm 1 of :cite:t:`BIBTEXLABEL` 
        for a single component pair. After training, the filter attributes
        wx, wy, and wt are updated.

        Args:
            x (:class:`NDTimeSeries`, (channel, time)): Time series of the 
                first modality
            y (:class:`NDTimeSeries`, (channel, time)): Time series of the 
                second modality.
            N_restarts (int): Number of times the algorithm is repeated.
            max_ter (int): Maximum number of iterations.
            tol (float): Tolerance value used for convergence criterion 
                when comparing correlations of consecutive runs.
        Returns:
            corr_best (float): Best correlation achieved among all repetitions
            
        """

        self.Nx = len(x.channel)
        self.Ny = len(y.channel)
        Nt = len(x.time)
        Ne = len(y.time)
        e_len = Nt//Ne
        N_features_max = np.min([self.Nx, self.Ny]) 
        if self.n_comp and self.n_comp > N_features_max:
            raise ValueError(f"Number of components {self.n_comp} should be smaller than number of features {N_features_max}")
        if self.n_comp is None:
            self.n_comp = N_features_max

        self.wx = np.zeros([self.Nx, N_features_max])
        self.wy = np.zeros([self.Ny, N_features_max])
        self.wt = np.zeros([self.n_lags, N_features_max])
        corr = np.zeros(N_features_max)

        # Initialize deflation matrices
        Bx = np.eye(self.Nx)
        By = np.eye(self.Ny)

        # # Whiten data (required for deflation procedure)
        x_wh = PCA(whiten=True).fit_transform(x.data.T).T
        y_wh = PCA(whiten=True).fit_transform(y.data.T).T
        # x = x.data
        # y = y.data
    
        y = y_wh.copy()
        x = x_wh.copy()

        for i in range(N_features_max):

            # Remove first epochs so its dimensions match bandpower Phi_x later
            y = y[:, self.n_lags - 1:]

            # Split signal into epochs
            x_epochs = self.split_epochs(x, Ne, e_len)
            
            # Epoch-wise covariance matrix, its temporal embedding and mean
            Cxxe = np.stack([np.cov(x_e) for x_e in x_epochs])
            # Catch D=1 case
            if len(x)==1:
                Cxxe = Cxxe.reshape(-1, 1, 1)
            tCxxe = self.temporal_embedding(Cxxe)
            Cxx = Cxxe.mean(axis=0)

            # Run one-unit algorithm
            corr[i], wx, wy, wt = self.one_unit_algorithm(y, Cxx, Cxxe, tCxxe, N_restarts, max_iter, tol)
            
            # Project into original space
            self.wx[:, i] = (Bx @ wx).squeeze()
            self.wy[:, i] = (By @ wy).squeeze()
            self.wt[:, i] = wt.squeeze()

            # Deflate data and update proyection matrices
            x, Bx = self.deflate_data(x_wh, self.wx[:, :i+1])
            y, By = self.deflate_data(y_wh, self.wy[:, :i+1])
        
        # Sort eigenvalues and eigenvectors in descending order and select number of components
        idx = np.argsort(corr)[::-1][:self.n_comp]
        corr = corr[idx]
        self.wx = self.wx[:, idx]
        self.wy = self.wy[:, idx]
        self.wt = self.wt[:, idx]
            
        return corr
    
    def transform(self, 
                  x: cdt.NDTimeSeries, 
                  y: cdt.NDTimeSeries) -> tuple:
        """ Get reconstructed sources of the x and y dataset.

        The x component is constructed by computing the bandpower of the x projection 
        along wx, and then applying a liner temporal filter using wt. 
        The y component is constructed as the linear projection of y along wy.

        Args:
            x (:class:`NDTimeSeries`, (channel, time)): Time series of the 
                first modality
            y (:class:`NDTimeSeries`, (channel, time)): Time series of the 
                second modality.

        Return:
            sx (ndarray): (Bandpower of) reconstructed x source.
            sy (ndarray): Reconstructed y source.

        """

        # Catch icompatible number of channels
        if len(x.channel) != self.Nx:
            raise ValueError(f"x should have {self.Nx} number of channels, but {len(x.channel)} was found!")
        if len(y.channel) != self.Ny:
            raise ValueError(f"y should have {self.Ny} number of channels, but {len(y.channel)} was found!")

        Nt = len(x.time)
        Ne = len(y.time)
        e_len = Nt//Ne

        # Whiten data (required for deflation procedure)
        x = PCA(whiten=True).fit_transform(x.data.T).T
        y = PCA(whiten=True).fit_transform(y.data.T).T

        # Remove first epochs so its dimensions match bandpower Phi_x later
        y = y[:, self.n_lags - 1:]

        # Split signal into epochs
        x_epochs = self.split_epochs(x, Ne, e_len)
        # Epoch-wise covariance matrix
        Cxxe = np.stack([np.cov(x_e) for x_e in x_epochs])
        # Epoch-wise bandpower with temporal embedding
        Phi_x = self.get_bandpower_temp(self.wx, Cxxe)
        # Catch 1D case
        if self.n_comp == 1:
            Phi_x = np.expand_dims(Phi_x, 0)
        # Reconstructed sources
        sx = np.stack([p.T @ w for p, w in zip(Phi_x, self.wt.T)])
        sy = self.wy.T.dot(y)
        # Normalize
        sx = standardize(sx, dim=1)
        sy = standardize(sy, dim=1)

        return sx, sy


    def one_unit_algorithm(self, y, Cxx, Cxxe, tCxxe, N_restarts, max_iter, tol):
        """ Run the one-unit algorithm of mSPoC to compute one single set of filters.
        """
        
        # Store new dimension
        Nx = len(Cxx)
        # To keep track of best model
        corr_best = 0.0
        wx_best = wy_best = wt_best = 0.0 
        for i in range(N_restarts):
            
            # Initialize random filter
            wx = np.random.normal(0, 1, [Nx, 1])
            # Epoch-wise bandpower with temporal embedding
            Phi_x = self.get_bandpower_temp(wx, Cxxe)
            # For convergence condition
            converged = False
            corr_old = 0.0

            for i in range(max_iter):
                # Apply CCA to get wt and wy
                wt, wy = self.apply_cca(Phi_x.T, y.T)
                # Apply temporal filter to tCxxe
                hCxxe = (wt.reshape(1, -1, 1, 1) * tCxxe).sum(axis=1)
                # Reconstructed y-source
                sy = wy.T.dot(y)
                # Build matrix for SPoC algorithm 
                hsy = (sy.reshape(-1, 1, 1) * hCxxe).mean(axis=0)
                # Solve generalized eigenvalue problem (SPoC)
                subset = [Nx - 1, Nx - 1]  # Get most positive eigenvalue
                _, wx = eigh(hsy, Cxx, eigvals_only=False, subset_by_index=subset)
                # Epoch-wise bandpower with temporal embedding
                Phi_x = self.get_bandpower_temp(wx, Cxxe)
                # Compute correlation
                sx = wt.T.dot(Phi_x).squeeze()
                corr = np.corrcoef(sx, sy)[0, 1]
                # Check for convergence
                if np.abs(corr - corr_old) < tol:
                    converged = True
                    break
                else:
                    corr_old = corr

            # Check for convergence
            if converged:
                if corr > corr_best:  # Save filters for best model
                    wx_best = wx
                    wy_best = wy
                    wt_best = wt
                    corr_best = corr
            else:
                print("mSPoC did not converged! (reached maximum number of iterations)")

        return corr_best, wx_best, wy_best, wt_best
    
    def deflate_data(self, x, w):
        """ Deflate data by removing from x the contribution of the projection on w.
        """

        # Builld orthonormal basis
        B = get_orthonormal_matrix(w)
        # Project to space orthonormal to w
        x = B.T @ x

        return x, B
    
    def temporal_embedding(self, v):
        """Build temporal embedding of v
        
        Stack copies of v shifted by the time lags. It assumes time direction 
        corresponds to index 0.
        """

        v_embedding = np.stack([
            v[e - self.t_lags] for e in range(self.n_lags - 1, v.shape[0])
            ])

        return v_embedding


    def get_bandpower_temp(self, W, C):
        """Compute bandpower with temporal embedding

        TODO: Add description.
        """

        # Epoch-wise power
        phi_x = np.stack([[w.T.dot(c).dot(w) for c in C] for w in W.T]).T
        # Temporal embedding
        Phi_x = self.temporal_embedding(phi_x).T
        
        # Catch 1D filter case
        if Phi_x.shape[0]==1:
            Phi_x = Phi_x[0]
        return Phi_x
    
    @staticmethod
    def apply_cca(a, b):
        """Initialize and fit CCA to the a, b pair

        TODO: Add description.
        """

        cca = CCA(n_components=1).fit(a, b)
        wa = cca.x_weights_
        wb = cca.y_weights_

        return wa, wb
        
    @staticmethod
    def split_epochs(x, Ne, e_len):
        """Split a signal x into Ne epochs of length e_len (index)
        TODO: Add description.
        """

        return np.stack([x[:, i*e_len:(i+1)*e_len] for i in range(Ne)])
        

def standardize(x: cdt.NDTimeSeriesSchema, dim: str = 'time'):
    """Standardize x along dimension dim.
    """

    x_standard = (x - x.mean(dim))/x.std(dim)

    return x_standard

def get_orthonormal_matrix(W):
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