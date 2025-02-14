"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import time

import numpy as np
from scipy import signal
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod

################################# Sliding-Window #################################

"""

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
"""


class SLIDING_WINDOW(BaseDFCMethod):

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.graphical_lasso_alpha_ = None

        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "sw_method",
            "tapered_window",
            "window_std",
            "W",
            "n_overlap",
            "normalization",
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

        self.params["measure_name"] = "SlidingWindow"
        self.params["is_state_based"] = False

        assert (
            self.params["sw_method"] in self.sw_methods_name_lst
        ), "sw_method not recognized."

    @property
    def measure_name(self):
        return self.params["measure_name"]  # + '_' + self.sw_method

    def shan_entropy(self, c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized * np.log2(c_normalized))
        return H

    def calc_MI(self, X, Y):

        bins = 20

        c_XY = np.histogram2d(X, Y, bins)[0]
        c_X = np.histogram(X, bins)[0]
        c_Y = np.histogram(Y, bins)[0]

        H_X = self.shan_entropy(c_X)
        H_Y = self.shan_entropy(c_Y)
        H_XY = self.shan_entropy(c_XY)

        MI = H_X + H_Y - H_XY
        return MI

    def FC(self, time_series):
        # Graphical Lasso
        if self.params['sw_method']=='GraphLasso':
            model = GraphicalLassoCV()
            model.fit(time_series.T)
            C = model.covariance_
        # Mutual information
        elif self.params['sw_method']=='MI':
            C = np.zeros((time_series.shape[0], time_series.shape[0]))
            
            for i in range(time_series.shape[0]):
                for j in range(i, time_series.shape[0]):      
                    X = time_series[i, :]
                    Y = time_series[j, :]
                    C[j, i] = self.calc_MI(X, Y)
        # Pearson correlation
        else:
            C = np.corrcoef(time_series)
            C[np.isnan(C)] = 0

        return C

    def dFC(self, time_series, W=None, n_overlap=None, tapered_window=False, window_std=None):
        # W is in time samples

        L = time_series.shape[1]
        step = int((1 - n_overlap) * W)
        if step == 0:
            step = 1

        # find the L1 penalty for GraphLasso
        if self.params["sw_method"] == "GraphLasso":
            model = GraphicalLassoCV()
            model.fit(time_series.T)
            self.graphical_lasso_alpha_ = model.alpha_

        FCSs = list()
        TR_array = list()
        for l in range(0, L - W + 1, step):

            # Create rectangular window
            window = np.zeros((L))
            window[l : l + W] = 1

            # Taper the window
            if tapered_window:
                std = window_std if window_std is not None else 3 * W / 22
                window_taper = signal.windows.gaussian(W, std=std)
                #window = signal.convolve(window, window_taper, mode="same") / sum(window_taper)
                window[l:l+W] = window_taper
            
            window = np.repeat(np.expand_dims(window, axis=0), time_series.shape[0], axis=0)
            FCSs.append(self.FC(np.multiply(time_series, window)[:,l:l+W]))
            TR_array.append(int((l + (l + W)) / 2))
    
        return np.array(FCSs), np.array(TR_array)

    def estimate_FCS(self, time_series):

        return self

    def estimate_dFC(self, time_series):
        """
        we assume calc is applied on subjects separately
        """
        assert (
            len(time_series.subj_id_lst) == 1
        ), "this function takes only one subject as input."

        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        # W is converted from sec to samples
        FCSs, TR_array = self.dFC(
            time_series=time_series.data,
            W=int(self.params["W"] * time_series.Fs),
            n_overlap=self.params["n_overlap"],
            tapered_window=self.params["tapered_window"],
            window_std=self.params["window_std"]
        )

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.info_dict)

        return dFC


################################################################################
