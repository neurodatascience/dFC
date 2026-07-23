"""
Sliding time-frequency dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class SLIDING_TIME_FREQ_COMBO(BaseDFCMethod):
    """Sliding-window dependence of local spectral profiles."""

    MEASURE_NAME = "SlidingTimeFreqcombo"

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
            "freq_bins",
            "tapered_window",
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
        if self.params["freq_bins"] is None:
            self.params["freq_bins"] = 12
        if self.params["tapered_window"] is None:
            self.params["tapered_window"] = True

    @property
    def measure_name(self):
        return self.params["measure_name"]

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

    def dFC(self, time_series, Fs):
        n_time = time_series.shape[1]
        W = min(max(int(round(float(self.params["W"]) * Fs)), 4), n_time)
        step = max(int(round((1.0 - float(self.params["n_overlap"])) * W)), 1)
        FCSs = []
        TR_array = []
        for start in range(0, n_time - W + 1, step):
            end = start + W
            segment = time_series[:, start:end]
            segment = segment - segment.mean(axis=1, keepdims=True)
            if self.params["tapered_window"]:
                taper = np.hanning(W)
                if np.any(taper):
                    segment = segment * taper[None, :]
            spectra = np.abs(np.fft.rfft(segment, axis=1))[:, 1:]
            n_bins = min(int(self.params["freq_bins"]), spectra.shape[1])
            FCSs.append(self._dependence(np.log1p(spectra[:, :n_bins])))
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
