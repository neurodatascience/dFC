"""
Derivative-weighted window dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class DERIVATIVE_WEIGHTED_WINDOW(BaseDFCMethod):
    """Windowed correlation weighted toward high-amplitude temporal changes."""

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
        self.params["measure_name"] = "DerivativeWeightedWindow"
        self.params["is_state_based"] = False
        if self.params["W"] is None:
            self.params["W"] = 30

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _samples(self, value, Fs, minimum=1):
        return max(int(round(value * Fs)), minimum)

    def _corr_from_cov(self, covariance):
        variance = np.diag(covariance)
        scale = np.sqrt(np.outer(variance, variance))
        corr = np.divide(
            covariance,
            scale,
            out=np.zeros_like(covariance, dtype=float),
            where=scale > 0,
        )
        corr[np.diag_indices_from(corr)] = 1
        corr[np.isnan(corr)] = 0
        return corr

    def _weighted_corr(self, samples, weights):
        weights = np.asarray(weights, dtype=float)
        weights = np.maximum(weights, 0)
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            weights = np.ones(samples.shape[1], dtype=float)
            weights_sum = np.sum(weights)
        weights = weights / weights_sum
        mean = np.sum(samples * weights[None, :], axis=1, keepdims=True)
        centered = samples - mean
        covariance = (centered * weights[None, :]) @ centered.T
        return self._corr_from_cov(covariance)

    def dFC(self, time_series, Fs):
        window = self._samples(self.params["W"], Fs, minimum=2)
        derivative_energy = np.zeros(time_series.shape[1])
        derivative_energy[1:] = np.mean(np.abs(np.diff(time_series, axis=1)), axis=0)
        FCSs = []
        TR_array = []
        for tr in range(window - 1, time_series.shape[1]):
            start = tr - window + 1
            weights = derivative_energy[start : tr + 1]
            weights = weights + 0.1 * np.mean(weights + 1e-8)
            FCSs.append(self._weighted_corr(time_series[:, start : tr + 1], weights))
            TR_array.append(tr)
        return np.array(FCSs), np.array(TR_array)

    def estimate_FCS(self, time_series):
        return self

    def estimate_dFC(self, time_series):
        assert (
            len(time_series.subj_id_lst) == 1
        ), "this function takes only one subject as input."
        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."
        time_series = self.manipulate_time_series4dFC(time_series)
        tic = time.time()
        FCSs, TR_array = self.dFC(time_series=time_series.data, Fs=time_series.Fs)
        self.set_dFC_assess_time(time.time() - tic)
        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.info_dict)
        return dFC
