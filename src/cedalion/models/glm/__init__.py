"""Tools for describing fNIRS data with general linear models."""

import cedalion.models.glm.design_matrix

from .basis_functions import TemporalBasisFunction, GaussianKernels, Gamma, DiracDelta
#from .design_matrix import hrf_regressors
from .solve import fit, predict, predict_with_uncertainty
