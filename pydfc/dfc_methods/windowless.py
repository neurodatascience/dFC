"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import time

import numpy as np
from ksvd import ApproximateKSVD
from sklearn.linear_model import orthogonal_mp_gram

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod

################################## Windowless ##################################

"""
by : https://github.com/nel215/ksvd

Reference: Rubinstein, R., Zibulevsky, M. and Elad, M., Efficient Implementation
of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit Technical
Report - CS Technion, April 2008

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
"""


class WINDOWLESS(BaseDFCMethod):

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

        self.params["measure_name"] = "Windowless"
        self.params["is_state_based"] = True

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def estimate_FCS(self, time_series):

        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # start timing
        tic = time.time()

        # time_series ~ gamma.dot(dictionary)
        self.aksvd = ApproximateKSVD(
            n_components=self.params["n_states"], transform_n_nonzero_coefs=1
        )
        self.dictionary = self.aksvd.fit(time_series.data.T).components_
        self.gamma = self.aksvd.transform(time_series.data.T)

        self.FCS_ = np.zeros(
            [self.params["n_states"], time_series.n_regions, time_series.n_regions]
        )
        for i in range(self.params["n_states"]):
            self.FCS_[i, :, :] = np.multiply(
                np.expand_dims(self.dictionary[i, :], axis=0).T,
                np.expand_dims(self.dictionary[i, :], axis=0),
            )

        self.Z = list()
        for i in range(self.gamma.shape[0]):
            self.Z.append(np.argwhere(self.gamma[i, :] != 0)[0, 0])

        # mean activation of states
        self.set_mean_activity(time_series)

        # record time
        self.set_FCS_fit_time(time.time() - tic)

        return self

    def transform_proba(self, D, X, n_nonzero_coefs=None):
        """
        returns the probability of each state for each time point
        D: dictionary, shape = (n_states, n_regions)
        X: time series data, shape = (n_time, n_regions)
        n_nonzero_coefs: number of non-zero coefficients to use in orthogonal matching pursuit
        Returns:
        Z_proba: shape = (n_time, n_states)
        """
        gram = D.dot(D.T)  # shape: (n_features, n_features) = (n_states, n_states)
        Xy = D.dot(X.T)  # shape: (n_features, n_targets) = (n_states, n_time)

        if n_nonzero_coefs is None:
            n_nonzero_coefs = D.shape[0]

        gamma = orthogonal_mp_gram(gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T

        Z_proba = np.abs(gamma) / np.abs(gamma).sum(axis=1, keepdims=True)

        return Z_proba

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

        gamma = self.aksvd.transform(time_series.data.T)  # shape: (n_time, n_states)

        Z = list()
        for i in range(time_series.n_time):
            Z.append(np.argwhere(gamma[i, :] != 0)[0, 0])

        # get probability for each state for each time point
        Z_proba = self.transform_proba(
            D=self.dictionary,
            X=time_series.data.T,
            n_nonzero_coefs=self.params["n_states"],
        )  # shape: (n_targets, n_features) = (n_time, n_states)

        assert Z_proba.shape[0] == time_series.n_time, (
            "Z_proba shape does not match time_series.n_time. "
            f"Z_proba shape: {Z_proba.shape}, time_series.n_time: {time_series.n_time}"
        )

        assert Z_proba.shape[1] == self.params["n_states"], (
            "Z_proba shape does not match n_states. "
            f"Z_proba shape: {Z_proba.shape}, n_states: {self.params['n_states']}"
        )

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(
            FCSs=self.FCS_, FCS_idx=Z, FCS_proba=Z_proba, TS_info=time_series.info_dict
        )
        return dFC


##########################################################################################
