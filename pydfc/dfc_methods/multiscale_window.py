"""
Multiscale window dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class MULTISCALE_WINDOW(BaseDFCMethod):
    """Average correlations across multiple recent temporal scales."""

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
        self.params["measure_name"] = "MultiscaleWindow"
        self.params["is_state_based"] = False
        if self.params["windows"] is None:
            self.params["windows"] = [15, 30, 60]

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

    def _corr(self, samples):
        corr = np.corrcoef(samples)
        corr[np.isnan(corr)] = 0
        corr[np.diag_indices_from(corr)] = 1
        return corr

    def dFC(self, time_series, Fs):
        windows = [
            self._samples(window, Fs, minimum=2) for window in self.params["windows"]
        ]
        min_periods = min(windows)
        FCSs = []
        TR_array = []
        for tr in range(min_periods - 1, time_series.shape[1]):
            matrices = []
            weights = []
            for window in windows:
                start = max(0, tr - window + 1)
                if tr - start + 1 >= 2:
                    matrices.append(self._corr(time_series[:, start : tr + 1]))
                    weights.append(np.sqrt(tr - start + 1))
            FCSs.append(np.average(np.array(matrices), axis=0, weights=weights))
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
