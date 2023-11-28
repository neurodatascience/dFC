"""The pydFC toolbox.

Submodules
---------

dfc_methods                --- implementation of dFC methods
multi_analysis             --- multi analysis class implementing 
                               multiple dFC methods simultaneously
time_series                --- time series class
dfc                        --- dfc class
data_loader                --- load data
dfc_utils                  --- functions used for dFC analysis
comparison                 --- functions used for dFC results comparison

"""

from . import dfc_methods
from .multi_analysis import MultiAnalysis
from .time_series import TIME_SERIES
from .dfc import DFC

__all__ = ['MultiAnalysis', 
           'TIME_SERIES', 
           'DFC', 
           'data_loader',
           'dfc_methods',
           'dfc_utils',
           'comparison'
           ]
