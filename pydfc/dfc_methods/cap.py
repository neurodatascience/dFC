"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import time

import numpy as np
from sklearn.cluster import KMeans

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod

################################## CAP ##################################

"""
by : web link

Reference: ##

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
"""


class CAP(BaseDFCMethod):

    def __init__(self, **params):
        self.logs_ = ""
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "n_states",
            "n_subj_clstrs",
            "normalization",
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

        self.params["measure_name"] = "CAP"
        self.params["is_state_based"] = True

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def act_vec2FCS(self, act_vecs):
        FCS_ = list()
        for act_vec in act_vecs:
            FCS_.append(np.multiply(act_vec[:, np.newaxis], act_vec[np.newaxis, :]))
        return np.array(FCS_)

    def cluster_act_vec(self, act_vecs, n_clusters):

        kmeans_ = KMeans(n_clusters=n_clusters, n_init=500).fit(act_vecs)
        act_centroids = kmeans_.cluster_centers_

        return act_centroids, kmeans_

    def estimate_FCS(self, time_series):

        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # start timing
        tic = time.time()

        # 2-level clustering
        SUBJECTs = time_series.subj_id_lst
        act_center_1st_level = None
        for subject in SUBJECTs:

            act_vecs = time_series.get_subj_ts(subjs_id=subject).data.T

            # test
            if act_vecs.shape[0] < self.params["n_subj_clstrs"]:
                print(
                    "Number of subject-level clusters cannot be more than time samples! n_subj_clstrs was changed to "
                    + str(act_vecs.shape[0])
                )
                self.params["n_subj_clstrs"] = act_vecs.shape[0]

            act_centroids, _ = self.cluster_act_vec(
                act_vecs=act_vecs, n_clusters=self.params["n_subj_clstrs"]
            )
            if act_center_1st_level is None:
                act_center_1st_level = act_centroids
            else:
                act_center_1st_level = np.concatenate(
                    (act_center_1st_level, act_centroids), axis=0
                )

        group_act_centroids, self.kmeans_ = self.cluster_act_vec(
            act_vecs=act_center_1st_level, n_clusters=self.params["n_states"]
        )
        self.FCS_ = self.act_vec2FCS(group_act_centroids)
        self.Z = self.kmeans_.predict(time_series.data.T)

        # mean activation of states
        self.set_mean_activity(time_series)

        # record time
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

        act_vecs = time_series.data.T

        Z = self.kmeans_.predict(act_vecs)

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=self.FCS_, FCS_idx=Z, TS_info=time_series.info_dict)
        return dFC


################################################################################
