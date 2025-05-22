"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import time
from copy import deepcopy

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod
from .sliding_window_clustr import SLIDING_WINDOW_CLUSTR

################################# HMM Discrete #################################

"""

Parameters
    ----------
    Z : numpy.ndarray
        state time course
    M : int
        (num of observations/n_state) of 16
    N : int
        (num of hidden states) of 24
    self.FCC :
        dFC estimated by Clustering which is then used to fit Discrete HMM
    self.FCS_ :
        collection FCS pattern coded in numbers for Discrete HMM

todo:
- two-level hierarchical clustering ?
- find a better name for FCC
"""
# from HMM_discrete import *
from hmmlearn import hmm


class HMM_DISC(BaseDFCMethod):

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.swc = None
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "clstr_base_measure",
            "clstr_distance",
            "sw_method",
            "tapered_window",
            "dhmm_obs_state_ratio",
            "coi_correction",
            "hmm_iter",
            "n_jobs",
            "verbose",
            "backend",
            "n_subj_clstrs",
            "W",
            "window_std",
            "n_overlap",
            "n_states",
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

        self.params["measure_name"] = "DiscreteHMM"
        self.params["is_state_based"] = True

        assert (
            self.params["clstr_base_measure"] in self.base_methods_name_lst
        ), "Base measure not recognized."

    @property
    def measure_name(self):
        return self.params["measure_name"]  # + '_' + self.base_method

    def estimate_FCS(self, time_series):

        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # start timing
        tic = time.time()

        # change n_states of swc to n_observations which is dhmm_obs_state_ratio*n_states
        params = deepcopy(self.params)
        params["n_states"] = int(
            self.params["dhmm_obs_state_ratio"] * self.params["n_states"]
        )

        self.swc = SLIDING_WINDOW_CLUSTR(**params)
        self.swc.estimate_FCS(time_series=time_series)

        SUBJECTs = time_series.subj_id_lst
        SWC_dFC = None
        Obs_seq = None
        for subject in SUBJECTs:
            new_dFC = self.swc.estimate_dFC(
                time_series=time_series.get_subj_ts(subjs_id=subject)
            )
            new_dFC_mat = new_dFC.get_dFC_mat()
            new_Obs = new_dFC.FCS_idx_array.reshape(-1, 1)
            if SWC_dFC is None:
                SWC_dFC = new_dFC_mat
                Obs_seq = new_Obs
            else:
                SWC_dFC = np.concatenate((SWC_dFC, new_dFC_mat), axis=0)
                Obs_seq = np.concatenate((Obs_seq, new_Obs), axis=0)

        Models, Scores = [], []
        for i in range(self.params["hmm_iter"]):
            model = hmm.CategoricalHMM(n_components=self.params["n_states"])
            model.fit(Obs_seq)
            score = model.score(Obs_seq)
            Models.append(model)
            Scores.append(score)

        self.hmm_model = Models[np.argmax(Scores)]
        self.Z = self.hmm_model.predict(Obs_seq)
        self.TPM = self.hmm_model.transmat_
        self.EPM = self.hmm_model.emissionprob_

        if len(np.unique(self.Z)) < self.params["n_states"]:
            self.logs_ += "Less number of states were fitted than n_states. \n"

        self.FCS_ = np.zeros(
            (self.params["n_states"], time_series.n_regions, time_series.n_regions)
        )
        for i in range(self.params["n_states"]):
            ids = np.array([int(state == i) for state in self.Z])
            if np.any(ids > 0):
                self.FCS_[i, :, :] = np.average(SWC_dFC, weights=ids, axis=0)

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

        FCC = self.swc.estimate_dFC(time_series=time_series)
        Obs_seq = FCC.FCS_idx_array.reshape(-1, 1)

        Z = self.hmm_model.predict(Obs_seq)

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(
            FCSs=self.FCS_,
            FCS_idx=Z,
            TS_info=time_series.info_dict,
            TR_array=FCC.TR_array,
        )

        return dFC


###################################################################################
