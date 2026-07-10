"""The :mod:`pydfc.dfc_methods` contains dFC methods objects."""

from .base_dfc_method import BaseDFCMethod
from .cap import CAP
from .continuous_hmm import HMM_CONT
from .discrete_hmm import HMM_DISC
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
    "SLIDING_WINDOW",
    "TIME_FREQ",
    "WINDOWLESS",
]
