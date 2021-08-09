
"""
Notes:
- Something to be aware of when grouping tests inside classes is that each test has 
    a unique instance of the class. Having each test share the same class instance 
    would be very detrimental to test isolation and would promote poor test practices.

todo:
- 
"""

from dFC_funcs import *
import numpy as np
import time
import hdf5storage
import scipy.io as sio

################################# UNIT TESTs ######################################
class Test_TIME_SERIES_CLASS:

    data = np.random.randn(333, 1200)
    data2 = np.random.randn(333, 1200)

    def test_n_regions(self):
        self.time_series = TIME_SERIES(data=self.data, Fs=2)
        assert self.time_series.n_regions==333, \
            "n_regions property failed."

    def test_n_time(self):
        self.time_series = TIME_SERIES(data=self.data, Fs=2)
        assert self.time_series.n_time==1200, \
            "n_time property failed."

    def test_append_ts(self):
        self.time_series = TIME_SERIES(data=self.data, Fs=2)
        self.time_series.append_ts(new_time_series=np.random.randn(333, 1200))
        assert self.time_series.n_time==2400, \
            "append_ts failed."
        assert self.time_series.n_regions==333, \
            "append_ts failed."

    def test_node_lst_after_append(self):
        self.time_series = TIME_SERIES(data=self.data, Fs=2)
        nodes_idx = np.random.choice(333, size = 50, replace=False)
        # nodes_idx.sort()
        self.time_series.select_nodes(nodes_idx=nodes_idx)
        self.time_series.append_ts(new_time_series=np.random.randn(333, 1200))

        assert self.time_series.n_regions==50, \
            "keeping node selection after append_ts failed."

        assert np.all(self.time_series.nodes_lst==np.array(nodes_idx)[:, np.newaxis]), \
            "keeping node selection after append_ts failed."

        self.time_series.select_nodes(nodes_idx=None)
        assert self.time_series.n_regions==333, \
            "resetting node selection after select_nodes(None) failed."

class Test_DFCM_CLASS:

    FCPs = np.random.rand(12, 333, 333)
    FCP_idx = np.random.randint(low=0, high=12, size=(27,))
    TR_array = np.arange(43, 1200, 44)

    FCPs2 = np.random.rand(12, 333, 333)
    FCP_idx2 = np.random.randint(low=0, high=12, size=(27,))
    TR_array2 = np.arange(43, 1200, 44)

    def test_one(self):
        self.dFCM = DFCM()
        self.dFCM.add_FCP(FCPs=self.FCPs, FCP_idx=self.FCP_idx, TR_array=self.TR_array)
        assert self.dFCM.TR_array[2] == 131

    def test_two(self):
        self.dFCM = DFCM()
        self.dFCM.add_FCP(FCPs=self.FCPs, FCP_idx=self.FCP_idx, TR_array=self.TR_array)
        assert self.dFCM.n_time == 27

################################# INTEGRATION TESTs ######################################



################################# MAIN ######################################

# if __name__ == '__main__':
#     TestNumber.test_two(x=2)