"""
Multiscale time-frequency dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class MULTISCALE_TIME_FREQ_COMBO(BaseDFCMethod):
    """Time-frequency spectral-profile dependence averaged across window scales."""

    MEASURE_NAME = "MultiscaleTimeFreqcombo"

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
            "freq_bins",
            "min_periods",
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
        if self.params["freq_bins"] is None:
            self.params["freq_bins"] = 12
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10

    @property
    def measure_name(self):
        return self.params["measure_name"]

    @staticmethod
    def _samples(value, Fs, minimum=1):
        return max(int(round(float(value) * Fs)), minimum)

    @staticmethod
    def _dependence(features):
        features = features - np.mean(features, axis=1, keepdims=True)
        scale = np.linalg.norm(features, axis=1, keepdims=True)
        features = np.divide(features, scale, out=np.zeros_like(features), where=scale > 0)
        matrix = features @ features.T
        matrix = 0.5 * (matrix + matrix.T)
        matrix[np.diag_indices_from(matrix)] = 1
        matrix[np.isnan(matrix)] = 0
        return np.clip(matrix, -1.0, 1.0)

    def _profiles(self, time_series, tr, window):
        start = max(0, tr - window + 1)
        segment = time_series[:, start : tr + 1]
        segment = segment - segment.mean(axis=1, keepdims=True)
        if segment.shape[1] < window:
            segment = np.hstack([np.zeros((segment.shape[0], window - segment.shape[1])), segment])
        taper = np.hanning(window)
        if not np.any(taper):
            taper = np.ones(window)
        spectra = np.abs(np.fft.rfft(segment * taper[None, :], axis=1))[:, 1:]
        n_bins = min(int(self.params["freq_bins"]), spectra.shape[1])
        return np.log1p(spectra[:, :n_bins])

    def dFC(self, time_series, Fs):
        windows = [self._samples(window, Fs, minimum=4) for window in self.params["windows"]]
        min_periods = max(int(self.params["min_periods"]), min(windows))
        FCSs = []
        TR_array = []
        for tr in range(min_periods - 1, time_series.shape[1]):
            matrices = []
            weights = []
            for window in windows:
                matrices.append(self._dependence(self._profiles(time_series, tr, window)))
                weights.append(np.sqrt(min(window, tr + 1)))
            FCSs.append(np.average(np.array(matrices), axis=0, weights=weights))
            TR_array.append(tr)
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
