"""
Adaptive multiscale window dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class ADAPTIVE_MULTISCALE_COMBO(BaseDFCMethod):
    """Multiscale recent-window correlations with adaptive exponential weights."""

    MEASURE_NAME = "AdaptiveMultiscalecombo"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "windows",
            "min_periods",
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
        if self.params["windows"] is None:
            self.params["windows"] = [15, 30, 60]
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10
        if self.params["alpha_min"] is None:
            self.params["alpha_min"] = 0.02
        if self.params["alpha_max"] is None:
            self.params["alpha_max"] = 0.35

    @property
    def measure_name(self):
        return self.params["measure_name"]

    @staticmethod
    def _samples(value, Fs, minimum=1):
        return max(int(round(value * Fs)), minimum)

    @staticmethod
    def _corr_from_cov(covariance):
        variance = np.diag(covariance)
        scale = np.sqrt(np.outer(variance, variance))
        corr = np.divide(
            covariance,
            scale,
            out=np.zeros_like(covariance, dtype=float),
            where=scale > 0,
        )
        corr = 0.5 * (corr + corr.T)
        corr[np.diag_indices_from(corr)] = 1
        corr[np.isnan(corr)] = 0
        return np.clip(corr, -1.0, 1.0)

    def _weighted_corr(self, samples, weights):
        weights = np.maximum(np.asarray(weights, dtype=float), 0)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights = np.ones(samples.shape[1], dtype=float)
            weights_sum = np.sum(weights)
        weights = weights / weights_sum
        mean = np.sum(samples * weights[None, :], axis=1, keepdims=True)
        centered = samples - mean
        covariance = (centered * weights[None, :]) @ centered.T
        return self._corr_from_cov(covariance)

    def _adaptive_alpha(self, diff_energy, baseline, tr):
        alpha_min = float(self.params["alpha_min"])
        alpha_max = float(self.params["alpha_max"])
        if not 0 <= alpha_min <= alpha_max <= 1:
            raise ValueError(
                "alpha_min and alpha_max must satisfy "
                "0 <= alpha_min <= alpha_max <= 1."
            )
        local_energy = diff_energy[tr] / baseline
        adapt = local_energy / (1.0 + local_energy)
        return alpha_min + (alpha_max - alpha_min) * adapt

    def dFC(self, time_series, Fs):
        windows = [self._samples(window, Fs, minimum=2) for window in self.params["windows"]]
        min_periods = int(self.params["min_periods"])
        min_start = max(min(min(windows), min_periods), 2)
        diff_energy = np.zeros(time_series.shape[1])
        diff_energy[1:] = np.mean(np.abs(np.diff(time_series, axis=1)), axis=0)
        baseline = np.median(diff_energy[1 : max(min_start, 2)]) + 1e-8
        FCSs = []
        TR_array = []

        for tr in range(min_start - 1, time_series.shape[1]):
            alpha = self._adaptive_alpha(diff_energy, baseline, tr)
            matrices = []
            scale_weights = []
            for window in windows:
                start = max(0, tr - window + 1)
                segment = time_series[:, start : tr + 1]
                if segment.shape[1] < 2:
                    continue
                age = segment.shape[1] - 1 - np.arange(segment.shape[1])
                weights = alpha * np.power(1.0 - alpha, age)
                matrices.append(self._weighted_corr(segment, weights))
                scale_weights.append(np.sqrt(segment.shape[1]))
            FCSs.append(np.average(np.array(matrices), axis=0, weights=scale_weights))
            TR_array.append(tr)
            baseline = 0.98 * baseline + 0.02 * max(diff_energy[tr], 1e-8)
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
