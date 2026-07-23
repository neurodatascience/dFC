"""
Random Fourier sliding-window dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class RANDOM_FOURIER_SLIDING_WINDOW_COMBO(BaseDFCMethod):
    """Sliding-window dependence after nonlinear random Fourier projection."""

    MEASURE_NAME = "RandomFourierSlidingWindowcombo"

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
            "n_random_features",
            "random_state",
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
        if self.params["n_random_features"] is None:
            self.params["n_random_features"] = 32

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _projector(self):
        rng = np.random.default_rng(self.params["random_state"])
        n_features = max(int(self.params["n_random_features"]), 2)
        return rng.normal(size=n_features), rng.uniform(0, 2 * np.pi, size=n_features)

    @staticmethod
    def _dependence(phi):
        phi = phi.reshape(phi.shape[0], -1)
        phi = phi - np.mean(phi, axis=1, keepdims=True)
        norm = np.linalg.norm(phi, axis=1, keepdims=True)
        phi = np.divide(phi, norm, out=np.zeros_like(phi), where=norm > 0)
        matrix = phi @ phi.T
        matrix = 0.5 * (matrix + matrix.T)
        matrix[np.diag_indices_from(matrix)] = 1
        matrix[np.isnan(matrix)] = 0
        return np.clip(matrix, -1.0, 1.0)

    def dFC(self, time_series, Fs):
        n_time = time_series.shape[1]
        W = min(max(int(round(float(self.params["W"]) * Fs)), 2), n_time)
        step = max(int(round((1.0 - float(self.params["n_overlap"])) * W)), 1)
        omega, phase = self._projector()
        FCSs = []
        TR_array = []
        for start in range(0, n_time - W + 1, step):
            end = start + W
            segment = time_series[:, start:end]
            if self.params["tapered_window"]:
                taper = np.hanning(W)
                if np.any(taper):
                    segment = segment * taper[None, :]
            phi = np.sqrt(2.0 / len(omega)) * np.cos(segment[:, :, None] * omega + phase)
            FCSs.append(self._dependence(phi))
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
