"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import time

import numpy as np
from joblib import Parallel, delayed
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..dfc import DFC
from ..dfc_utils import KMeansCustom, SW_downsample, dFC_mat2vec, dFC_vec2mat
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod
from .sliding_window import SLIDING_WINDOW
from .time_freq import TIME_FREQ

########################### Sliding_Window + Clustering ############################

"""
- We used a tapered window as in Allen et al., created by convolving a rectangle (width = 22 TRs = 44s)
  with a Gaussian (σ = 3 TRs) and slid in steps of 1 TR, resulting in W= 126 windows (Allen et al., 2014).
- Kmeans Clustering is repeated 500 times to escape local minima (Allen et al., 2014)
- for clustering, we have a 2-level kmeans clustering. First, we cluster FCSs of each subject. Then, we
    cluster all clustering centers from all subjects. the final estimate_dFC is using the second kmeans
    model (Allen et al., 2014; Ou et al., 2015).

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.
"""


class SLIDING_WINDOW_CLUSTR(BaseDFCMethod):

    def __init__(self, **params):

        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "clstr_base_measure",
            "sw_method",
            "tapered_window",
            "clstr_distance",
            "coi_correction",
            "n_subj_clstrs",
            "W",
            "window_std",
            "n_overlap",
            "n_states",
            "normalization",
            "n_jobs_swc",
            "verbose",
            "backend_swc",
            "n_jobs_sw",
            "backend_sw",
            "n_jobs_tf",
            "backend_tf",
            "num_subj",
            "num_select_nodes",
            "num_time_point",
            "Fs_ratio",
            "noise_ratio",
            "num_realization",
            "session",
        ]
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None

        self.params["measure_name"] = "Clustering"
        self.params["is_state_based"] = True

        if self.params["clstr_distance"] is None:
            # Default clustering distance is euclidean
            self.params["clstr_distance"] = "euclidean"

        assert (
            self.params["clstr_distance"] == "euclidean"
            or self.params["clstr_distance"] == "manhattan"
        ), "Clustering distance not recognized. It must be either \
                euclidean or manhattan."

        assert (
            self.params["clstr_base_measure"] in self.base_methods_name_lst
        ), "Base method not recognized."

    @property
    def measure_name(self):
        return self.params["measure_name"]  # + '_' + self.base_method

    def dFC_mat2vec(self, C_t):
        return dFC_mat2vec(C_t)

    def dFC_vec2mat(self, F, N):
        return dFC_vec2mat(F=F, N=N)

    def clusters_lst2idx(self, clusters):
        Z = np.zeros((self.F.shape[0],))
        for i, cluster in enumerate(clusters):
            for sample in cluster:
                Z[sample] = i
        return Z.astype(int)

    def set_mean_activity(self, time_series):
        """
        Calculate the mean activity of regions at each state across all subjects.
        For this method, you have to run this separately after estimating FCSs to save computation time.
        """

        SUBJECTs = time_series.subj_id_lst
        TS_data = None
        Z = None
        for subject in SUBJECTs:
            subj_TS = time_series.get_subj_ts(subjs_id=subject)
            # build base measure inside worker (safer for multiprocessing)
            if self.params["clstr_base_measure"] == "Time-Freq":
                base_dFC = TIME_FREQ(**self.params)
            elif self.params["clstr_base_measure"] == "SlidingWindow":
                base_dFC = SLIDING_WINDOW(**self.params)
            else:
                raise ValueError("Base method not recognized.")

            sw_mat = base_dFC.estimate_dFC(
                time_series=subj_TS
            ).get_dFC_mat()  # shape: (n_win, n, n)
            subj_Z = self.kmeans_.predict(
                self.dFC_mat2vec(sw_mat).astype(np.float32)
            )  # shape: (n_win,)

            new_TS_data = SW_downsample(
                data=subj_TS.data.T,
                Fs=time_series.Fs,
                W=self.params["W"],
                n_overlap=self.params["n_overlap"],
                tapered_window=self.params["tapered_window"],
            ).T
            if TS_data is None:
                TS_data = new_TS_data
                Z = subj_Z
            else:
                TS_data = np.concatenate((TS_data, new_TS_data), axis=1)
                Z = np.concatenate((Z, subj_Z), axis=0)

        assert (
            TS_data.shape[1] == Z.shape[0]
        ), "Mismatch in time points between TS data and state assignments."
        mean_act = list()
        for i in np.unique(Z):
            ids = np.array([int(state == i) for state in Z])
            mean_act.append(np.average(TS_data, weights=ids, axis=1))
        self.mean_act = np.array(mean_act)

    def cluster_FC(self, FCS_raw, n_clusters, n_regions):

        F = self.dFC_mat2vec(FCS_raw).astype(np.float32, copy=False)

        if self.params["clstr_distance"] == "manhattan":
            ########### Manhattan Clustering ##############
            kmeans_ = KMeansCustom(
                n_clusters=n_clusters,
                n_init=500,
                init="k-means++",
                metric="manhattan",
            ).fit(F)
            kmeans_.cluster_centers_ = kmeans_.cluster_centers_.astype(np.float32)
            F_cent = kmeans_.cluster_centers_
        else:
            ########### Euclidean Clustering ##############
            kmeans_ = KMeans(n_clusters=n_clusters, n_init=500, init="k-means++").fit(F)
            kmeans_.cluster_centers_ = kmeans_.cluster_centers_.astype(np.float32)
            F_cent = kmeans_.cluster_centers_

        FCS_ = self.dFC_vec2mat(F_cent, N=n_regions)
        return FCS_, kmeans_

    def estimate_FCS(self, time_series):

        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."
        time_series = self.manipulate_time_series4FCS(time_series)

        tic = time.time()

        SUBJECTs = time_series.subj_id_lst

        # --- worker ---
        def _process_one_subject(ts_sub):

            # build base measure inside worker (safer for multiprocessing)
            if self.params["clstr_base_measure"] == "Time-Freq":
                base_dFC = TIME_FREQ(**self.params)
            elif self.params["clstr_base_measure"] == "SlidingWindow":
                base_dFC = SLIDING_WINDOW(**self.params)
            else:
                raise ValueError("Base method not recognized.")

            dFC_raw = base_dFC.estimate_dFC(time_series=ts_sub)

            # subject-level clusters: do NOT mutate self.params in parallel
            n_subj_clstrs = self.params["n_subj_clstrs"]
            if dFC_raw.n_time < n_subj_clstrs:
                # keep behavior: cap clusters to available samples
                n_subj_clstrs = dFC_raw.n_time

            sw_mat = dFC_raw.get_dFC_mat().astype(
                np.float32, copy=False
            )  # shape: (n_win, n, n)

            FCS_sub, _ = self.cluster_FC(
                FCS_raw=sw_mat,
                n_clusters=n_subj_clstrs,
                n_regions=dFC_raw.n_regions,
            )

            # return what we need for level-2 clustering + later labeling
            return FCS_sub

        # choose parallelism
        n_jobs = self.params["n_jobs_swc"] if self.params["n_jobs_swc"] is not None else 1
        backend = (
            self.params["backend_swc"]
            if self.params["backend_swc"] is not None
            else "threading"
        )
        # if SW is already parallelized, do not parallelize over subjects
        if self.params["n_jobs_sw"] is not None:
            if self.params["n_jobs_sw"] != 1:
                n_jobs = 1

        if n_jobs == 1:
            # no parallelism
            FCS_1st_level_list = []
            for subj in SUBJECTs:
                res = _process_one_subject(time_series.get_subj_ts(subjs_id=subj))
                FCS_1st_level_list.append(res)
        else:
            # parallelism
            FCS_1st_level_list = Parallel(n_jobs=n_jobs, backend=backend)(
                delayed(_process_one_subject)(time_series.get_subj_ts(subjs_id=subj))
                for subj in SUBJECTs
            )

        # unpack + concatenate once (much faster than repeated concatenate)
        # NOTE: results may come back in any order; but that's OK since we just need all FCSs

        FCS_1st_level = np.concatenate(FCS_1st_level_list, axis=0)

        assert (
            FCS_1st_level.shape[1] == FCS_1st_level.shape[2]
        ), "FCS matrices must be square."
        assert (
            FCS_1st_level.shape[1] == time_series.n_regions
        ), "FCS matrices size must match number of regions."

        # 2nd-level clustering (unchanged)
        self.FCS_, self.kmeans_ = self.cluster_FC(
            FCS_raw=FCS_1st_level,
            n_clusters=self.params["n_states"],
            n_regions=time_series.n_regions,
        )

        self.set_FCS_fit_time(time.time() - tic)
        return self

    def estimate_dFC(self, time_series):

        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."

        assert (
            len(time_series.subj_id_lst) == 1
        ), "this function takes only one subject as input."

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        base_dFC = None
        if self.params["clstr_base_measure"] == "Time-Freq":
            base_dFC = TIME_FREQ(**self.params)
        if self.params["clstr_base_measure"] == "SlidingWindow":
            base_dFC = SLIDING_WINDOW(**self.params)

        dFC_raw = base_dFC.estimate_dFC(time_series=time_series)

        F = self.dFC_mat2vec(dFC_raw.get_dFC_mat(TRs=dFC_raw.TR_array))

        # The code below is similar for both clustering methods,
        # but is kept this way for clarity.
        if self.params["clstr_distance"] == "manhattan":
            ########### Manhattan Clustering ##############
            Z = self.kmeans_.predict(F.astype(np.float32))
            # get distances from the cluster centers for each sample
            distances = self.kmeans_.transform(
                F.astype(np.float32)
            )  # shape: (n_samples, n_clusters)
            # normalize distances to semi probabilities
            rel = -distances
            rel = rel - rel.min(axis=1, keepdims=True)  # shift min to 0
            rel = rel / rel.sum(axis=1, keepdims=True)  # normalize
            Z_proba = rel  # shape: (n_samples, n_clusters) = (n_time, n_states)
        else:
            ########### Euclidean Clustering ##############
            Z = self.kmeans_.predict(F.astype(np.float32))
            # get distances from the cluster centers for each sample
            distances = self.kmeans_.transform(
                F.astype(np.float32)
            )  # shape: (n_samples, n_clusters)
            # normalize distances to semi probabilities
            rel = -distances
            rel = rel - rel.min(axis=1, keepdims=True)  # shift min to 0
            rel = rel / rel.sum(axis=1, keepdims=True)  # normalize
            Z_proba = rel  # shape: (n_samples, n_clusters) = (n_time, n_states)

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(
            FCSs=self.FCS_,
            FCS_idx=Z,
            FCS_proba=Z_proba,
            TS_info=time_series.info_dict,
            TR_array=dFC_raw.TR_array,
        )

        return dFC


################################################################################
