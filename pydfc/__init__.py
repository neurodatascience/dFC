"""The pydFC toolbox.

Submodules
---------

dfc_methods                --- implementation of dFC methods
multi_analysis             --- multi analysis class implementing
                               multiple dFC methods simultaneously
time_series                --- time series class
dfc                        --- dfc class
data_loader                --- load data
multi_analysis_utils       --- utility functions for multi analysis
                               to implement multiple dFC methods
                               simultaneously
dfc_utils                  --- functions used for dFC analysis
comparison                 --- functions used for dFC results comparison

"""

from . import dfc_methods
from .dfc import DFC
from .multi_analysis import MultiAnalysis
from .time_series import TIME_SERIES

__all__ = [
    "MultiAnalysis",
    "TIME_SERIES",
    "DFC",
    "data_loader",
    "multi_analysis_utils",
    "dfc_methods",
    "dfc_utils",
    "comparison",
]
