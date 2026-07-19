"""
Random Fourier time-frequency dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class RANDOM_FOURIER_TIME_FREQ_COMBO(BaseDFCMethod):
    """Random Fourier dependence on local time-frequency spectral profiles."""

    MEASURE_NAME = "RandomFourierTimeFreqcombo"

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
            "min_periods",
            "freq_bins",
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
        if self.params["tf_window"] is None:
            self.params["tf_window"] = 30
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10
        if self.params["freq_bins"] is None:
            self.params["freq_bins"] = 12
        if self.params["n_random_features"] is None:
            self.params["n_random_features"] = 32

    @property
    def measure_name(self):
        return self.params["measure_name"]

    @staticmethod
    def _samples(value, Fs, minimum=1):
        return max(int(round(value * Fs)), minimum)

    def _spectral_profiles(self, time_series, tr, window_samples):
        start = max(0, tr - window_samples + 1)
        segment = time_series[:, start : tr + 1]
        segment = segment - segment.mean(axis=1, keepdims=True)
        if segment.shape[1] < window_samples:
            pad = np.zeros((segment.shape[0], window_samples - segment.shape[1]))
            segment = np.hstack([pad, segment])
        taper = np.hanning(window_samples)
        if not np.any(taper):
            taper = np.ones(window_samples)
        spectra = np.abs(np.fft.rfft(segment * taper[None, :], axis=1))[:, 1:]
        n_bins = min(int(self.params["freq_bins"]), spectra.shape[1])
        profiles = spectra[:, :n_bins]
        scale = np.linalg.norm(profiles, axis=1, keepdims=True)
        return np.divide(profiles, scale, out=np.zeros_like(profiles), where=scale > 0)

    def _feature_projector(self, n_input):
        rng = np.random.default_rng(self.params["random_state"])
        n_features = max(int(self.params["n_random_features"]), 2)
        omega = rng.normal(size=(n_input, n_features))
        phase = rng.uniform(0, 2 * np.pi, size=n_features)
        return omega, phase

    @staticmethod
    def _dependence(phi):
        phi = phi - np.mean(phi, axis=1, keepdims=True)
        norm = np.linalg.norm(phi, axis=1, keepdims=True)
        phi = np.divide(phi, norm, out=np.zeros_like(phi), where=norm > 0)
        matrix = phi @ phi.T
        matrix = 0.5 * (matrix + matrix.T)
        matrix[np.diag_indices_from(matrix)] = 1
        matrix[np.isnan(matrix)] = 0
        return np.clip(matrix, -1.0, 1.0)

    def dFC(self, time_series, Fs):
        min_periods = int(self.params["min_periods"])
        window_samples = self._samples(self.params["tf_window"], Fs, minimum=max(min_periods, 4))
        first_profiles = self._spectral_profiles(time_series, min_periods - 1, window_samples)
        omega, phase = self._feature_projector(first_profiles.shape[1])
        FCSs = []
        TR_array = []

        for tr in range(min_periods - 1, time_series.shape[1]):
            profiles = self._spectral_profiles(time_series, tr, window_samples)
            phi = np.sqrt(2.0 / omega.shape[1]) * np.cos(profiles @ omega + phase)
            FCSs.append(self._dependence(phi))
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
