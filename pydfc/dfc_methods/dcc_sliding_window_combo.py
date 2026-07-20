"""
DCC sliding-window dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class DCC_SLIDING_WINDOW_COMBO(BaseDFCMethod):
    """Sliding-window summaries of DCC conditional correlation trajectories."""

    MEASURE_NAME = "DccSlidingWindowcombo"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "W",
            "n_overlap",
            "garch_lambda",
            "dcc_a",
            "dcc_b",
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
            self.params[params_name] = params.get(params_name, None)
        self.params["measure_name"] = self.MEASURE_NAME
        self.params["is_state_based"] = False
        if self.params["W"] is None:
            self.params["W"] = 44
        if self.params["n_overlap"] is None:
            self.params["n_overlap"] = 0.5
        if self.params["garch_lambda"] is None:
            self.params["garch_lambda"] = 0.94
        if self.params["dcc_a"] is None:
            self.params["dcc_a"] = 0.05
        if self.params["dcc_b"] is None:
            self.params["dcc_b"] = 0.85

    @property
    def measure_name(self):
        return self.params["measure_name"]

    @staticmethod
    def _corr_from_cov(covariance):
        variance = np.diag(covariance)
        scale = np.sqrt(np.outer(variance, variance))
        corr = np.divide(covariance, scale, out=np.zeros_like(covariance), where=scale > 0)
        corr = 0.5 * (corr + corr.T)
        corr[np.diag_indices_from(corr)] = 1
        corr[np.isnan(corr)] = 0
        return np.clip(corr, -1.0, 1.0)

    @staticmethod
    def _nearest_spd(matrix):
        matrix = 0.5 * (matrix + matrix.T)
        vals, vecs = np.linalg.eigh(matrix)
        return vecs @ np.diag(np.maximum(vals, 1e-6)) @ vecs.T

    def _dcc_path(self, time_series):
        a = float(self.params["dcc_a"])
        b = float(self.params["dcc_b"])
        if not 0 <= a < 1 or not 0 <= b < 1 or a + b >= 1:
            raise ValueError("dcc_a and dcc_b must be non-negative and sum to less than 1.")
        lam = float(self.params["garch_lambda"])
        eps = time_series - time_series.mean(axis=1, keepdims=True)
        sigma2 = np.zeros_like(eps)
        sigma2[:, 0] = np.maximum(np.var(eps, axis=1), 1e-8)
        for tr in range(1, eps.shape[1]):
            sigma2[:, tr] = lam * sigma2[:, tr - 1] + (1.0 - lam) * eps[:, tr - 1] ** 2
        Z = eps / np.sqrt(np.maximum(sigma2, 1e-12))
        Q_bar = self._nearest_spd(np.corrcoef(Z))
        Q_bar[np.isnan(Q_bar)] = 0
        Q_bar[np.diag_indices_from(Q_bar)] = 1
        Q = Q_bar.copy()
        matrices = [self._corr_from_cov(Q)]
        for tr in range(1, Z.shape[1]):
            Q = (1.0 - a - b) * Q_bar + a * np.outer(Z[:, tr - 1], Z[:, tr - 1]) + b * Q
            matrices.append(self._corr_from_cov(Q))
        return np.array(matrices)

    def dFC(self, time_series, Fs):
        path = self._dcc_path(time_series)
        n_time = time_series.shape[1]
        W = min(max(int(round(float(self.params["W"]) * Fs)), 2), n_time)
        step = max(int(round((1.0 - float(self.params["n_overlap"])) * W)), 1)
        FCSs = []
        TR_array = []
        for start in range(0, n_time - W + 1, step):
            end = start + W
            FCSs.append(np.mean(path[start:end], axis=0))
            TR_array.append(int((start + end) / 2))
        return np.array(FCSs), np.array(TR_array)

    def estimate_FCS(self, time_series):
        return self

    def estimate_dFC(self, time_series):
        assert len(time_series.subj_id_lst) == 1, "this function takes only one subject as input."
        assert type(time_series) is TIME_SERIES, "time_series must be of TIME_SERIES class."
        time_series = self.manipulate_time_series4dFC(time_series)
        tic = time.time()
        FCSs, TR_array = self.dFC(time_series=time_series.data, Fs=time_series.Fs)
        self.set_dFC_assess_time(time.time() - tic)
        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.info_dict)
        return dFC
