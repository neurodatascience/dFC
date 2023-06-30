"""The dFC multi_analysis toolbox.

Submodules
---------

dfc_methods                --- implementation of dFC methods
multi_analysis             --- implementation of dFC methods
time_series                --- implementation of dFC methods
dfc                        --- implementation of dFC methods
data_loader                --- implementation of dFC methods
dfc_utils                  --- implementation of dFC methods
comparison                 --- implementation of dFC methods

"""

from git_codes.multi_analysis_dfc import dfc_methods
from .multi_analysis import MultiAnalysis
from .time_series import TIME_SERIES
from .dfc import DFC
from .data_loader import DATA_LOADER


__all__ = ['MultiAnalysis', 
           'TIME_SERIES', 
           'DFC', 
           'DATA_LOADER',
           'dfc_methods',
           'dfc_utils',
           'comparison'
           ]