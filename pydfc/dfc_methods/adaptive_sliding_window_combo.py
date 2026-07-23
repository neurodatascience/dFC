"""
Adaptive sliding-window dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class ADAPTIVE_SLIDING_WINDOW_COMBO(BaseDFCMethod):
    """Sliding-window correlations with adaptive within-window recency weights."""

    MEASURE_NAME = "AdaptiveSlidingWindowcombo"

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
            "tapered_window",
            "window_std",
            "alpha_min",
            "alpha_max",
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
        if self.params["tapered_window"] is None:
            self.params["tapered_window"] = True
        if self.params["alpha_min"] is None:
            self.params["alpha_min"] = 0.02
        if self.params["alpha_max"] is None:
            self.params["alpha_max"] = 0.35

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

    def _weighted_corr(self, samples, weights):
        weights = np.maximum(np.asarray(weights, dtype=float), 0)
        if np.sum(weights) <= 0:
            weights = np.ones(samples.shape[1], dtype=float)
        weights = weights / np.sum(weights)
        centered = samples - np.sum(samples * weights[None, :], axis=1, keepdims=True)
        covariance = (centered * weights[None, :]) @ centered.T
        return self._corr_from_cov(covariance)

    def _alpha(self, diff_energy, baseline, tr):
        alpha_min = float(self.params["alpha_min"])
        alpha_max = float(self.params["alpha_max"])
        if not 0 <= alpha_min <= alpha_max <= 1:
            raise ValueError("alpha_min and alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1.")
        adapt = (diff_energy[tr] / baseline) / (1.0 + diff_energy[tr] / baseline)
        return alpha_min + (alpha_max - alpha_min) * adapt

    def dFC(self, time_series, Fs):
        n_time = time_series.shape[1]
        W = max(int(round(float(self.params["W"]) * Fs)), 2)
        W = min(W, n_time)
        step = max(int(round((1.0 - float(self.params["n_overlap"])) * W)), 1)
        diff_energy = np.zeros(n_time)
        diff_energy[1:] = np.mean(np.abs(np.diff(time_series, axis=1)), axis=0)
        baseline = np.median(diff_energy[1 : max(W, 2)]) + 1e-8
        FCSs = []
        TR_array = []
        for start in range(0, n_time - W + 1, step):
            end = start + W
            tr = int((start + end) / 2)
            alpha = self._alpha(diff_energy, baseline, min(tr, n_time - 1))
            age = W - 1 - np.arange(W)
            weights = alpha * np.power(1.0 - alpha, age)
            if self.params["tapered_window"]:
                taper = np.hanning(W)
                if np.any(taper):
                    weights = weights * taper
            FCSs.append(self._weighted_corr(time_series[:, start:end], weights))
            TR_array.append(tr)
            baseline = 0.98 * baseline + 0.02 * max(diff_energy[min(tr, n_time - 1)], 1e-8)
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
