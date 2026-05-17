"""
Lagged maximum correlation dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class LAGGED_MAX_CORRELATION(BaseDFCMethod):
    MEASURE_NAME = "LaggedMaxCorrelation"
    """Windowed FC using the strongest short-lag pairwise correlation."""

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
            "max_lag",
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
            self.params["W"] = 30
        if self.params["max_lag"] is None:
            self.params["max_lag"] = 2

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _samples(self, value, Fs, minimum=1):
        return max(int(round(value * Fs)), minimum)

    def _corr(self, samples):
        corr = np.corrcoef(samples)
        corr[np.isnan(corr)] = 0
        corr[np.diag_indices_from(corr)] = 1
        return corr

    def _standardize(self, data):
        centered = data - np.mean(data, axis=1, keepdims=True)
        scale = np.std(centered, axis=1, keepdims=True)
        return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)

    def _lagged_corr(self, segment):
        max_lag = int(self.params["max_lag"])
        best = self._corr(segment)
        best_abs = np.abs(best)
        for lag in range(1, max_lag + 1):
            if segment.shape[1] <= lag + 1:
                break
            lead = self._standardize(segment[:, lag:])
            trail = self._standardize(segment[:, :-lag])
            corr = (lead @ trail.T) / max(lead.shape[1] - 1, 1)
            corr = 0.5 * (corr + corr.T)
            replace = np.abs(corr) > best_abs
            best[replace] = corr[replace]
            best_abs[replace] = np.abs(corr[replace])
        best[np.diag_indices_from(best)] = 1
        return best

    def dFC(self, time_series, Fs):
        window = self._samples(self.params["W"], Fs, minimum=2)
        FCSs = []
        TR_array = []
        for tr in range(window - 1, time_series.shape[1]):
            segment = time_series[:, tr - window + 1 : tr + 1]
            FCSs.append(self._lagged_corr(segment))
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
