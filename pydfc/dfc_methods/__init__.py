"""The :mod:`pydfc.dfc_methods` contains dFC methods objects."""

from .adaptive_exponential_window import ADAPTIVE_EXPONENTIAL_WINDOW
from .base_dfc_method import BaseDFCMethod
from .cap import CAP
from .changepoint_reset_window import CHANGEPOINT_RESET_WINDOW
from .continuous_hmm import HMM_CONT
from .copula_tail_dependence import COPULA_TAIL_DEPENDENCE
from .derivative_weighted_window import DERIVATIVE_WEIGHTED_WINDOW
from .discrete_hmm import HMM_DISC
from .edge_coactivation import EDGE_COACTIVATION
from .event_synchronization import EVENT_SYNCHRONIZATION
from .exponential_window import EXPONENTIAL_WINDOW
from .graph_diffusion_coactivation import GRAPH_DIFFUSION_COACTIVATION
from .kalman_covariance import KALMAN_COVARIANCE
from .lagged_max_correlation import LAGGED_MAX_CORRELATION
from .multiscale_window import MULTISCALE_WINDOW
from .oja_subspace_connectivity import OJA_SUBSPACE_CONNECTIVITY
from .phase_locking_window import PHASE_LOCKING_WINDOW
from .precision_shrinkage_window import PRECISION_SHRINKAGE_WINDOW
from .random_fourier_dependence import RANDOM_FOURIER_DEPENDENCE
from .recurrence_kernel_dependence import RECURRENCE_KERNEL_DEPENDENCE
from .sliding_window import SLIDING_WINDOW
from .sliding_window_clustr import SLIDING_WINDOW_CLUSTR
from .time_freq import TIME_FREQ
from .windowless import WINDOWLESS

__all__ = [
    "BaseDFCMethod",
    "CAP",
    "SLIDING_WINDOW_CLUSTR",
    "HMM_CONT",
    "HMM_DISC",
    "EXPONENTIAL_WINDOW",
    "ADAPTIVE_EXPONENTIAL_WINDOW",
    "MULTISCALE_WINDOW",
    "EDGE_COACTIVATION",
    "PHASE_LOCKING_WINDOW",
    "DERIVATIVE_WEIGHTED_WINDOW",
    "CHANGEPOINT_RESET_WINDOW",
    "KALMAN_COVARIANCE",
    "LAGGED_MAX_CORRELATION",
    "PRECISION_SHRINKAGE_WINDOW",
    "RECURRENCE_KERNEL_DEPENDENCE",
    "RANDOM_FOURIER_DEPENDENCE",
    "EVENT_SYNCHRONIZATION",
    "COPULA_TAIL_DEPENDENCE",
    "OJA_SUBSPACE_CONNECTIVITY",
    "GRAPH_DIFFUSION_COACTIVATION",
    "SLIDING_WINDOW",
    "TIME_FREQ",
    "WINDOWLESS",
]
