
"""
Notes:
- Something to be aware of when grouping tests inside classes is that each test has 
    a unique instance of the class. Having each test share the same class instance 
    would be very detrimental to test isolation and would promote poor test practices.

todo:
- carefully test TS truncate
"""

from functions.dFC_funcs import *
import numpy as np
import time
import hdf5storage
import scipy.io as sio

################################# FUNCTIONS ######################################

def check_symmetric(A, rtol=1e-05, atol=1e-08):

    for t in range(A.shape[0]):
        if not np.allclose(A[t,:,:], A[t,:,:].T, rtol=rtol, atol=atol):
            return False
    return True

################################# UNIT TESTs ######################################
class Test_TIME_SERIES_CLASS:

    data = np.random.randn(333, 1200)
    data2 = np.random.randn(333, 1200)

    def test_n_regions(self):
        self.time_series = TIME_SERIES(data=self.data, subj_id='1', Fs=2)
        assert self.time_series.n_regions==333, \
            "n_regions property failed."

    def test_n_time(self):
        self.time_series = TIME_SERIES(data=self.data, subj_id='1', Fs=2)
        assert self.time_series.n_time==1200, \
            "n_time property failed."

    def test_append_ts(self):
        self.time_series = TIME_SERIES(data=self.data, subj_id='1', Fs=2)
        self.time_series.append_ts(new_time_series=np.random.randn(333, 1200), subj_id='2')
        assert self.time_series.n_time==2400, \
            "append_ts failed."
        assert self.time_series.n_regions==333, \
            "append_ts failed."

    def test_node_lst_after_append(self):
        self.time_series = TIME_SERIES(data=self.data, subj_id='1', Fs=2)
        nodes_idx = np.random.choice(333, size = 50, replace=False)
        # nodes_idx.sort()
        self.time_series.select_nodes(nodes_idx=nodes_idx)
        self.time_series.append_ts(new_time_series=np.random.randn(333, 1200), subj_id='2')

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
        self.dFCM.add_FCP(FCPs=self.FCPs, FCP_idx=self.FCP_idx, subj_id_array=[1]*len(self.FCP_idx), TR_array=self.TR_array)
        assert self.dFCM.TR_array[2] == 131

    def test_two(self):
        self.dFCM = DFCM()
        self.dFCM.add_FCP(FCPs=self.FCPs, FCP_idx=self.FCP_idx, subj_id_array=[1]*len(self.FCP_idx), TR_array=self.TR_array)
        assert self.dFCM.n_time == 27

class Test_methods:

    Fs = 2
    data = np.random.randn(50, 300)
    data2 = np.random.randn(50, 300)
    time_BOLD = 1/Fs + np.arange(0, data.shape[1]/Fs, 1/Fs)

    n_states = 2
    n_subj_clstrs = 4
    n_hid_states = 3
    n_overlap = 0.5
    W_sw = 44 # in seconds, 44, choose even Ws!?
    n_jobs = None
    verbose=0

    params = {'W': int(W_sw*Fs), 'n_overlap': n_overlap, \
    'n_states': n_states, 'n_subj_clstrs': n_subj_clstrs, 'n_hid_states': n_hid_states, \
    'n_jobs': n_jobs, 'verbose': verbose, 'backend': 'loky' \
            }

    hmm_cont = HMM_CONT(**params)
    windowless = WINDOWLESS(**params)

    sw_pc = SLIDING_WINDOW(sw_method='pear_corr', **params)
    sw_mi = SLIDING_WINDOW(sw_method='MI', **params)
    # sw_gLasso = SLIDING_WINDOW(sw_method='GraphLasso', **params)

    time_freq_cwt = TIME_FREQ(method='CWT_mag', **params)
    time_freq_cwt_r = TIME_FREQ(method='CWT_phase_r', **params)
    time_freq_wtc = TIME_FREQ(method='WTC', **params)

    swc_pc = SLIDING_WINDOW_CLUSTR(sw_method='pear_corr', **params)
    swc_mi = SLIDING_WINDOW_CLUSTR(sw_method='MI', **params)
    # swc_gLasso = SLIDING_WINDOW_CLUSTR(sw_method='GraphLasso', **params)

    hmm_disc_pc = HMM_DISC(sw_method='pear_corr', **params)
    hmm_disc_mi = HMM_DISC(sw_method='MI', **params)
    # hmm_disc_gLasso = HMM_DISC(sw_method='GraphLasso', **params)

    time_series = TIME_SERIES(data=data, Fs=Fs, subj_id='1', time_array=time_BOLD, TS_name='BOLD Simulation')

    def test_one(self):

        MEASURES = [
            self.hmm_cont, \
            self.windowless, \
            self.sw_pc, \
            self.sw_mi, \
            # self.sw_gLasso, \
            self.time_freq_cwt, \
            self.time_freq_cwt_r, \
            self.time_freq_wtc, \
            self.swc_pc, \
            self.swc_mi, \
            # self.swc_gLasso, \
            self.swc_mi, \
            self.hmm_disc_pc,\
            # self.hmm_disc_gLasso, \
            self.hmm_disc_mi \
                    ]

        for measure in MEASURES:
            if measure.is_state_based:
                measure.estimate_FCS(time_series=self.time_series)
            dFCM = measure.estimate_dFCM(time_series=self.time_series)

            assert not dFCM is None, \
                measure.measure_name + " has None dFCM."

            assert dFCM.n_regions==self.time_series.n_regions, \
                measure.measure_name + " has n_regions mismatch."
################################# INTEGRATION TESTs ######################################



################################# MAIN ######################################

# if __name__ == '__main__':
#     TestNumber.test_two(x=2)