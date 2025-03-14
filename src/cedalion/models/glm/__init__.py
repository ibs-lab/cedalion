"""Tools for describing fNIRS data with general linear models."""

from .basis_functions import TemporalBasisFunction, GaussianKernels, Gamma, DiracDelta
from .design_matrix import make_design_matrix
from .solve import fit, predict
