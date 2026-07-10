"""
Recurrence-kernel dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class RECURRENCE_KERNEL_DEPENDENCE(BaseDFCMethod):
    MEASURE_NAME = "RecurrenceKernelDependence"
    """State-dependent FC from samples whose whole-brain pattern recurs."""

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "min_periods",
            "kernel_width",
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
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10
        if self.params["kernel_width"] is None:
            self.params["kernel_width"] = 1.5

    @property
    def measure_name(self):
        return self.params["measure_name"]

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
        min_periods = int(self.params["min_periods"])
        FCSs = []
        TR_array = []
        for tr in range(min_periods - 1, time_series.shape[1]):
            history = time_series[:, : tr + 1]
            state = history[:, -1:]
            distances = np.mean((history - state) ** 2, axis=0)
            scale = np.median(distances) + 1e-8
            weights = np.exp(-distances / (self.params["kernel_width"] * scale))
            FCSs.append(self._weighted_corr(history, weights))
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
