"""
Notes:
- Something to be aware of when grouping tests inside classes is that each test has
    a unique instance of the class. Having each test share the same class instance
    would be very detrimental to test isolation and would promote poor test practices.

todo:
- carefully test TS truncate
"""

import time

import hdf5storage
import numpy as np
import scipy.io as sio
from functions.dFC_funcs import *

################################# FUNCTIONS ######################################


def check_symmetric(A, rtol=1e-05, atol=1e-08):

    for t in range(A.shape[0]):
        if not np.allclose(A[t, :, :], A[t, :, :].T, rtol=rtol, atol=atol):
            return False
    return True


def sign_maintenance(C, C_z):
    sign_check = True
    for t in range(C.shape[0]):
        slice_C = C[t, :, :]
        slice_C_z = C_z[t, :, :]
        slice_C = slice_C[np.where(~np.eye(slice_C.shape[0], dtype=bool))]
        slice_C_z = slice_C_z[np.where(~np.eye(slice_C_z.shape[0], dtype=bool))]
        if not np.all(
            np.sign(slice_C[slice_C_z != 0]) == np.sign(slice_C_z[slice_C_z != 0])
        ):
            sign_check = False
    return sign_check


################################# UNIT TESTs ######################################
class Test_dFC_mat_normalize:

    def test_constant(self):
        C = 2 * np.ones((120, 333, 333))
        C_z = dFC_mat_normalize(C, global_normalization=False, threshold=0.0)
        assert np.all(C_z == 0), "constant matrix normalization failed."
        C_z = dFC_mat_normalize(C, global_normalization=True, threshold=0.0)
        assert np.all(C_z == 0), "constant matrix normalization failed."

    def test_random_C(self):
        C = -1 + 2 * np.random.randn(120, 333, 333)
        for i in range(C.shape[1]):
            C[:, i, i] = 1
        C_z = dFC_mat_normalize(C, global_normalization=False, threshold=0.0)

        assert sign_maintenance(
            C=C, C_z=C_z
        ), "random matrix normalization sign maintenance failed."

        assert np.all(C_z <= 1) and np.all(
            C_z >= -1
        ), "random matrix normalization failed."

    def test_cov_C(self):
        TS = np.random.randn(333, 200)
        C = list()
        for l in range(100):
            C.append(np.cov(TS[:, l : l + 100], bias=False))
        C = np.array(C)
        assert check_symmetric(C), "Covariance matrix Not symmetric."

        C_z = dFC_mat_normalize(C, global_normalization=False, threshold=0.0)

        assert np.all(C_z <= 1) and np.all(
            C_z >= -1
        ), "Covariance matrix normalization failed."
        assert sign_maintenance(
            C=C, C_z=C_z
        ), "Covariance matrix normalization sign maintenance failed."
        assert check_symmetric(
            C_z
        ), "Covariance matrix normalization failed. Not symmetric."

    # def test_threshold(self):
    #     C = -1 + 2 * np.random.randn(120, 333, 333)
    #     for i in range(C.shape[1]):
    #         C[:,i,i] = 1
    #     C_z = dFC_mat_normalize(C, global_normalization=False, threshold=0.0)
    #     assert time_series.n_regions==333, \
    #         "n_regions property failed."


class Test_TIME_SERIES_CLASS:

    data = np.random.randn(333, 1200)
    time_array = np.random.randn(1200)
    time_array = np.sort(time_array)
    data2 = np.random.randn(333, 1200)
    time_array2 = np.random.randn(1200)
    time_array2 = np.sort(time_array2)

    def test_n_regions(self):
        time_series = TIME_SERIES(data=self.data, subj_id="1", Fs=2)
        assert time_series.n_regions == 333, "n_regions property failed."

    def test_get_subj_ts(self):

        ########## normal append_ts ##############
        time_series = TIME_SERIES(
            data=self.data, subj_id="1", Fs=2, time_array=self.time_array2
        )
        time_series.append_ts(
            new_time_series=self.data2, subj_id="2", time_array=self.time_array2
        )

        assert time_series.n_time_ == 2400, "n_time_ property failed."
        assert time_series.n_time == 2400, "n_time property failed."
        assert time_series.n_regions_ == 333, "n_regions_ property failed."
        assert time_series.n_regions == 333, "n_regions property failed."

        assert time_series.interval_ == list(
            range(time_series.n_time_)
        ), "interval property failed."
        assert time_series.interval.tolist()[0] == list(
            range(time_series.n_time_)
        ), "interval property failed."

        assert np.all(
            time_series.get_subj_ts("1").data_ == self.data
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("1").data == self.data
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("2").data_ == self.data2
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("2").data == self.data2
        ), "get_subj_ts method failed."

        assert (
            np.unique(time_series.get_subj_ts("1").subj_id_array) == "1"
        ), "get_subj_ts method failed."
        assert (
            np.unique(time_series.get_subj_ts("2").subj_id_array) == "2"
        ), "get_subj_ts method failed."

        ########## append_ts after node selection ##############
        time_series = TIME_SERIES(
            data=self.data, subj_id="1", Fs=2, time_array=self.time_array2
        )
        nodes_idx = np.random.choice(range(time_series.n_regions), size=50, replace=False)
        nodes_idx.sort()
        time_series.select_nodes(nodes_idx=nodes_idx)
        time_series.append_ts(
            new_time_series=self.data2, subj_id="2", time_array=self.time_array2
        )

        assert time_series.n_regions_ == 333, "n_regions_ property failed."
        assert time_series.n_regions == 50, "n_regions property failed."

        assert np.all(
            time_series.get_subj_ts("1").data_ == self.data[nodes_idx, :]
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("1").data == self.data[nodes_idx, :]
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("1").get_subj_ts("1").get_subj_ts("1").data
            == self.data[nodes_idx, :]
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("2").data_ == self.data2[nodes_idx, :]
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("2").data == self.data2[nodes_idx, :]
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("2").get_subj_ts("2").get_subj_ts("2").data
            == self.data2[nodes_idx, :]
        ), "get_subj_ts method failed."

        ########## append_ts after truncate ##############
        time_series = TIME_SERIES(
            data=self.data, subj_id="1", Fs=2, time_array=self.time_array2
        )
        time_series.truncate(start_point=10, end_point=100)
        time_series.append_ts(
            new_time_series=self.data2, subj_id="2", time_array=self.time_array2
        )

        assert time_series.n_regions == 333, "n_regions_ property failed."
        assert time_series.n_time_ == 2400, "n_time_ property failed."
        assert time_series.n_time == 2400, "n_time property failed."

        assert np.all(
            time_series.get_subj_ts("1").data_ == self.data
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("1").data == self.data
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("1").get_subj_ts("1").get_subj_ts("1").data
            == self.data
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("2").data_ == self.data2
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("2").data == self.data2
        ), "get_subj_ts method failed."
        assert np.all(
            time_series.get_subj_ts("2").get_subj_ts("2").get_subj_ts("2").data
            == self.data2
        ), "get_subj_ts method failed."

    def test_n_time(self):
        time_series = TIME_SERIES(data=self.data, subj_id="1", Fs=2)
        assert time_series.n_time_ == 1200, "n_time property failed."
        assert time_series.n_time == 1200, "n_time property failed."
        time_series.truncate(start_point=10, end_point=100)
        assert time_series.n_time_ == 1200, "TRUNCATE: n_time property failed."
        assert time_series.n_time == 91, "TRUNCATE: n_time property failed."
        time_series.truncate()
        assert time_series.n_time_ == 1200, "TRUNCATE_RESET: n_time property failed."
        assert time_series.n_time == 1200, "TRUNCATE_RESET: n_time property failed."

    def test_append_ts(self):
        time_series = TIME_SERIES(data=self.data, subj_id="1", Fs=2)
        time_series.append_ts(new_time_series=np.random.randn(333, 1200), subj_id="2")
        assert time_series.n_time == 2400, "append_ts failed."
        assert time_series.n_regions == 333, "append_ts failed."

    def test_node_lst_after_append(self):
        time_series = TIME_SERIES(data=self.data, subj_id="1", Fs=2)
        nodes_idx = np.random.choice(333, size=50, replace=False)
        # nodes_idx.sort()
        time_series.select_nodes(nodes_idx=nodes_idx)
        time_series.append_ts(new_time_series=np.random.randn(333, 1200), subj_id="2")

        assert (
            time_series.n_regions == 50
        ), "keeping node selection after append_ts failed."

        assert np.all(
            time_series.nodes_lst == np.array(nodes_idx)[:, np.newaxis]
        ), "keeping node selection after append_ts failed."

        time_series.select_nodes(nodes_idx=None)
        assert (
            time_series.n_regions == 333
        ), "resetting node selection after select_nodes(None) failed."


class Test_DFCM_CLASS:

    FCPs = np.random.rand(12, 333, 333)
    FCP_idx = np.random.randint(low=0, high=12, size=(27,))
    TR_array = np.arange(43, 1200, 44)

    FCPs2 = np.random.rand(12, 333, 333)
    FCP_idx2 = np.random.randint(low=0, high=12, size=(27,))
    TR_array2 = np.arange(43, 1200, 44)

    def test_get_dFC_mat(self):
        dFCM = DFCM()
        dFCM.add_FCP(
            FCPs=self.FCPs,
            FCP_idx=self.FCP_idx,
            subj_id_array=[1] * len(self.FCP_idx),
            TR_array=self.TR_array,
        )

        assert np.all(
            dFCM.get_dFC_mat(TRs=[131]) == self.FCPs[self.FCP_idx[2]]
        ), "get_dFC_mat method failed."

        idx = np.random.randint(low=0, high=27, size=(10,))
        assert np.all(
            dFCM.get_dFC_mat(TRs=self.TR_array[idx]) == self.FCPs[self.FCP_idx[idx]]
        ), "get_dFC_mat method failed."

    def test_add_FCP_multiple(self):
        ########## add_FCP multiple ##############
        dFCM = DFCM()
        dFCM.add_FCP(
            FCPs=self.FCPs,
            FCP_idx=self.FCP_idx,
            subj_id_array=[1] * len(self.FCP_idx),
            TR_array=self.TR_array,
        )
        assert dFCM.TR_array[2] == 131, "add_FCP method failed."
        assert dFCM.n_time == 27, "add_FCP method failed."

    def test_add_FCP_single(self):
        ########## add_FCP one-by-one ##############
        # for SW class

        dFCM = DFCM()
        for i in range(12):
            dFCM.add_FCP(
                FCPs=self.FCPs[i], subj_id_array="1", TR_array=self.TR_array[i : i + 1]
            )

        assert dFCM.TR_array[2] == 131, "add_FCP method failed."
        assert dFCM.n_time == 12, "add_FCP method failed."

        idx = np.random.randint(low=0, high=12, size=(10,))
        assert np.all(
            dFCM.get_dFC_mat(TRs=self.TR_array[idx]) == self.FCPs[idx]
        ), "get_dFC_mat method failed."

    def test_two(self):
        dFCM = DFCM()
        dFCM.add_FCP(
            FCPs=self.FCPs,
            FCP_idx=self.FCP_idx,
            subj_id_array=[1] * len(self.FCP_idx),
            TR_array=self.TR_array,
        )
        assert dFCM.n_time == 27


# class Test_methods:

#     Fs = 2
#     data = np.random.randn(50, 300)
#     data2 = np.random.randn(50, 300)
#     time_BOLD = 1/Fs + np.arange(0, data.shape[1]/Fs, 1/Fs)

#     n_states = 2
#     n_subj_clstrs = 4
#     n_hid_states = 3
#     n_overlap = 0.5
#     W_sw = 44 # in seconds, 44, choose even Ws!?
#     n_jobs = None
#     verbose=0

#     params = {'W': int(W_sw*Fs), 'n_overlap': n_overlap, \
#     'n_states': n_states, 'n_subj_clstrs': n_subj_clstrs, 'n_hid_states': n_hid_states, \
#     'n_jobs': n_jobs, 'verbose': verbose, 'backend': 'loky' \
#             }

#     hmm_cont = HMM_CONT(**params)
#     windowless = WINDOWLESS(**params)

#     sw_pc = SLIDING_WINDOW(sw_method='pear_corr', **params)
#     sw_mi = SLIDING_WINDOW(sw_method='MI', **params)
#     sw_gLasso = SLIDING_WINDOW(sw_method='GraphLasso', **params)

#     time_freq_cwt = TIME_FREQ(method='CWT_mag', **params)
#     time_freq_cwt_r = TIME_FREQ(method='CWT_phase_r', **params)
#     time_freq_wtc = TIME_FREQ(method='WTC', **params)

#     swc_pc = SLIDING_WINDOW_CLUSTR(sw_method='pear_corr', **params)
#     swc_mi = SLIDING_WINDOW_CLUSTR(sw_method='MI', **params)
#     # swc_gLasso = SLIDING_WINDOW_CLUSTR(sw_method='GraphLasso', **params)

#     hmm_disc_pc = HMM_DISC(sw_method='pear_corr', **params)
#     hmm_disc_mi = HMM_DISC(sw_method='MI', **params)
#     # hmm_disc_gLasso = HMM_DISC(sw_method='GraphLasso', **params)

#     time_series = TIME_SERIES(data=data, Fs=Fs, subj_id='1', time_array=time_BOLD, TS_name='BOLD Simulation')

#     def test_one(self):

#         MEASURES = [
#             self.hmm_cont, \
#             self.windowless, \
#             self.sw_pc, \
#             self.sw_mi, \
#             self.sw_gLasso, \
#             self.time_freq_cwt, \
#             self.time_freq_cwt_r, \
#             self.time_freq_wtc, \
#             self.swc_pc, \
#             # self.swc_mi, \
#             # self.swc_gLasso, \
#             self.hmm_disc_pc,\
#             # self.hmm_disc_mi \
#             # self.hmm_disc_gLasso, \
#                     ]

#         for measure in MEASURES:
#             if measure.is_state_based:
#                 measure.estimate_FCS(time_series=self.time_series)
#             dFCM = measure.estimate_dFCM(time_series=self.time_series)

#             assert not dFCM is None, \
#                 measure.measure_name + " has None dFCM."

#             assert dFCM.n_regions==self.time_series.n_regions, \
#                 measure.measure_name + " has n_regions mismatch."
################################# INTEGRATION TESTs ######################################


################################# MAIN ######################################

# if __name__ == '__main__':
#     TestNumber.test_two(x=2)
