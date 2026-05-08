"""The :mod:`pydfc.dfc_methods` contains dFC methods objects."""

from .adaptive_exponential_window import ADAPTIVE_EXPONENTIAL_WINDOW
from .agglomerative_states import AGGLOMERATIVE_STATES
from .base_dfc_method import BaseDFCMethod
from .bayesian_gaussian_mixture_states import BAYESIAN_GAUSSIAN_MIXTURE_STATES
from .birch_states import BIRCH_STATES
from .cap import CAP
from .changepoint_reset_window import CHANGEPOINT_RESET_WINDOW
from .continuous_hmm import HMM_CONT
from .copula_tail_dependence import COPULA_TAIL_DEPENDENCE
from .derivative_weighted_window import DERIVATIVE_WEIGHTED_WINDOW
from .discrete_hmm import HMM_DISC
from .edge_coactivation import EDGE_COACTIVATION
from .event_synchronization import EVENT_SYNCHRONIZATION
from .exponential_window import EXPONENTIAL_WINDOW
from .gaussian_mixture_states import GAUSSIAN_MIXTURE_STATES
from .graph_diffusion_coactivation import GRAPH_DIFFUSION_COACTIVATION
from .kalman_covariance import KALMAN_COVARIANCE
from .lagged_kmeans_states import LAGGED_KMEANS_STATES
from .lagged_max_correlation import LAGGED_MAX_CORRELATION
from .markov_smoothed_gmm_states import MARKOV_SMOOTHED_GMM_STATES
from .markov_smoothed_kmeans_states import MARKOV_SMOOTHED_KMEANS_STATES
from .minibatch_kmeans_states import MINIBATCH_KMEANS_STATES
from .multiscale_window import MULTISCALE_WINDOW
from .oja_subspace_connectivity import OJA_SUBSPACE_CONNECTIVITY
from .phase_locking_window import PHASE_LOCKING_WINDOW
from .pooled_kmeans_states import POOLED_KMEANS_STATES
from .precision_shrinkage_window import PRECISION_SHRINKAGE_WINDOW
from .random_fourier_dependence import RANDOM_FOURIER_DEPENDENCE
from .recurrence_kernel_dependence import RECURRENCE_KERNEL_DEPENDENCE
from .sliding_window import SLIDING_WINDOW
from .sliding_window_clustr import SLIDING_WINDOW_CLUSTR
from .spectral_states import SPECTRAL_STATES
from .time_freq import TIME_FREQ
from .windowless import WINDOWLESS

__all__ = [
    "BaseDFCMethod",
    "AGGLOMERATIVE_STATES",
    "BAYESIAN_GAUSSIAN_MIXTURE_STATES",
    "BIRCH_STATES",
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
    "GAUSSIAN_MIXTURE_STATES",
    "LAGGED_MAX_CORRELATION",
    "LAGGED_KMEANS_STATES",
    "MARKOV_SMOOTHED_GMM_STATES",
    "MARKOV_SMOOTHED_KMEANS_STATES",
    "MINIBATCH_KMEANS_STATES",
    "POOLED_KMEANS_STATES",
    "PRECISION_SHRINKAGE_WINDOW",
    "RECURRENCE_KERNEL_DEPENDENCE",
    "RANDOM_FOURIER_DEPENDENCE",
    "SPECTRAL_STATES",
    "EVENT_SYNCHRONIZATION",
    "COPULA_TAIL_DEPENDENCE",
    "OJA_SUBSPACE_CONNECTIVITY",
    "GRAPH_DIFFUSION_COACTIVATION",
    "SLIDING_WINDOW",
    "TIME_FREQ",
    "WINDOWLESS",
]
