"""
Time-frequency DCC dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class DCC_TIME_FREQ_COMBO(BaseDFCMethod):
    """DCC conditional correlations of local time-frequency power innovations."""

    MEASURE_NAME = "DccTimeFreqcombo"

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
            "garch_lambda",
            "dcc_a",
            "dcc_b",
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
        if self.params["tf_window"] is None:
            self.params["tf_window"] = 30
        if self.params["freq_bins"] is None:
            self.params["freq_bins"] = 12
        if self.params["garch_lambda"] is None:
            self.params["garch_lambda"] = 0.94
        if self.params["dcc_a"] is None:
            self.params["dcc_a"] = 0.05
        if self.params["dcc_b"] is None:
            self.params["dcc_b"] = 0.85
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10

    @property
    def measure_name(self):
        return self.params["measure_name"]

    @staticmethod
    def _samples(value, Fs, minimum=1):
        return max(int(round(value * Fs)), minimum)

    @staticmethod
    def _corr_from_cov(covariance):
        variance = np.diag(covariance)
        scale = np.sqrt(np.outer(variance, variance))
        corr = np.divide(
            covariance,
            scale,
            out=np.zeros_like(covariance, dtype=float),
            where=scale > 0,
        )
        corr = 0.5 * (corr + corr.T)
        corr[np.diag_indices_from(corr)] = 1
        corr[np.isnan(corr)] = 0
        return np.clip(corr, -1.0, 1.0)

    @staticmethod
    def _nearest_spd(matrix):
        matrix = 0.5 * (matrix + matrix.T)
        vals, vecs = np.linalg.eigh(matrix)
        return vecs @ np.diag(np.maximum(vals, 1e-6)) @ vecs.T

    def _spectral_power(self, time_series, tr, window_samples):
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
        power = np.sqrt(np.mean(spectra[:, :n_bins] ** 2, axis=1))
        return np.log1p(power)

    def _tf_innovations(self, time_series, Fs):
        min_periods = int(self.params["min_periods"])
        window_samples = self._samples(self.params["tf_window"], Fs, minimum=max(min_periods, 4))
        features = np.array([
            self._spectral_power(time_series, tr, window_samples)
            for tr in range(time_series.shape[1])
        ]).T
        return features - features.mean(axis=1, keepdims=True)

    def _standardize(self, features):
        n_regions, n_time = features.shape
        lam = float(self.params["garch_lambda"])
        sigma2 = np.zeros((n_regions, n_time))
        sigma2[:, 0] = np.maximum(np.var(features, axis=1), 1e-8)
        for tr in range(1, n_time):
            sigma2[:, tr] = lam * sigma2[:, tr - 1] + (1.0 - lam) * features[:, tr - 1] ** 2
        return features / np.sqrt(np.maximum(sigma2, 1e-12))

    def dFC(self, time_series, Fs):
        min_periods = int(self.params["min_periods"])
        a = float(self.params["dcc_a"])
        b = float(self.params["dcc_b"])
        if not 0 <= a < 1 or not 0 <= b < 1 or a + b >= 1:
            raise ValueError("dcc_a and dcc_b must be non-negative and sum to less than 1.")

        Z = self._standardize(self._tf_innovations(time_series, Fs))
        Q_bar = self._nearest_spd(np.corrcoef(Z))
        Q_bar[np.isnan(Q_bar)] = 0
        Q_bar[np.diag_indices_from(Q_bar)] = 1
        Q = Q_bar.copy()
        FCSs = []
        TR_array = []
        for tr in range(1, time_series.shape[1]):
            z = Z[:, tr - 1]
            Q = (1.0 - a - b) * Q_bar + a * np.outer(z, z) + b * Q
            R = self._corr_from_cov(Q)
            if tr >= min_periods - 1:
                FCSs.append(R)
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
