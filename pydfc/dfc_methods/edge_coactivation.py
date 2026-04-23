"""
Edge co-activation dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class EDGE_COACTIVATION(BaseDFCMethod):
    """Smoothed edge co-activation from instantaneous z-scored products."""

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "half_life",
            "min_periods",
            "normalization",
            "num_select_nodes",
            "num_time_point",
            "Fs_ratio",
            "noise_ratio",
            "num_realization",
            "session",
            "alpha",
        ]
        self.params = {}
        for params_name in self.params_name_lst:
            self.params[params_name] = params.get(params_name, None)
        self.params["measure_name"] = "EdgeCoactivation"
        self.params["is_state_based"] = False
        if self.params["half_life"] is None:
            self.params["half_life"] = 30
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _alpha_from_half_life(self, Fs):
        if self.params["alpha"] is not None:
            return float(self.params["alpha"])
        half_life_samples = max(float(self.params["half_life"]) * Fs, 1.0)
        return 1.0 - np.exp(np.log(0.5) / half_life_samples)

    def dFC(self, time_series, Fs):
        alpha = self._alpha_from_half_life(Fs)
        min_periods = int(self.params["min_periods"])
        mean = np.zeros(time_series.shape[0])
        var = np.ones(time_series.shape[0])
        edge_state = np.eye(time_series.shape[0])
        FCSs = []
        TR_array = []
        for tr in range(time_series.shape[1]):
            sample = time_series[:, tr]
            delta = sample - mean
            mean = (1.0 - alpha) * mean + alpha * sample
            var = (1.0 - alpha) * var + alpha * delta**2
            z_sample = (sample - mean) / np.sqrt(var + 1e-8)
            edge_state = (1.0 - alpha) * edge_state + alpha * np.outer(z_sample, z_sample)
            matrix = edge_state.copy()
            matrix[np.diag_indices_from(matrix)] = 1
            if tr >= min_periods - 1:
                FCSs.append(matrix)
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
