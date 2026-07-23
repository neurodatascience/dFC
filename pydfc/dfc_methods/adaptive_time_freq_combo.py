"""
Adaptive time-frequency dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class ADAPTIVE_TIME_FREQ_COMBO(BaseDFCMethod):
    """Local spectral-profile dependence smoothed with adaptive forgetting."""

    MEASURE_NAME = "AdaptiveTimeFreqcombo"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "tf_window",
            "freq_bins",
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
        if self.params["tf_window"] is None:
            self.params["tf_window"] = 30
        if self.params["freq_bins"] is None:
            self.params["freq_bins"] = 12
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

    def _spectral_profiles(self, time_series, tr, window_samples):
        start = max(0, tr - window_samples + 1)
        segment = time_series[:, start : tr + 1]
        segment = segment - segment.mean(axis=1, keepdims=True)
        if segment.shape[1] < window_samples:
            segment = np.hstack([np.zeros((segment.shape[0], window_samples - segment.shape[1])), segment])
        taper = np.hanning(window_samples)
        if not np.any(taper):
            taper = np.ones(window_samples)
        spectra = np.abs(np.fft.rfft(segment * taper[None, :], axis=1))[:, 1:]
        n_bins = min(int(self.params["freq_bins"]), spectra.shape[1])
        return np.log1p(spectra[:, :n_bins])

    def _alpha(self, diff_energy, baseline, tr):
        alpha_min = float(self.params["alpha_min"])
        alpha_max = float(self.params["alpha_max"])
        if not 0 <= alpha_min <= alpha_max <= 1:
            raise ValueError("alpha_min and alpha_max must satisfy 0 <= alpha_min <= alpha_max <= 1.")
        ratio = diff_energy[tr] / baseline
        return alpha_min + (alpha_max - alpha_min) * (ratio / (1.0 + ratio))

    def dFC(self, time_series, Fs):
        min_periods = int(self.params["min_periods"])
        window_samples = self._samples(self.params["tf_window"], Fs, minimum=max(min_periods, 4))
        diff_energy = np.zeros(time_series.shape[1])
        diff_energy[1:] = np.mean(np.abs(np.diff(time_series, axis=1)), axis=0)
        baseline = np.median(diff_energy[1 : max(min_periods, 2)]) + 1e-8
        smoothed = np.eye(time_series.shape[0])
        FCSs = []
        TR_array = []
        for tr in range(time_series.shape[1]):
            profiles = self._spectral_profiles(time_series, tr, window_samples)
            alpha = self._alpha(diff_energy, baseline, tr)
            smoothed = (1.0 - alpha) * smoothed + alpha * self._dependence(profiles)
            smoothed = 0.5 * (smoothed + smoothed.T)
            smoothed[np.diag_indices_from(smoothed)] = 1
            if tr >= min_periods - 1:
                FCSs.append(smoothed.copy())
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
