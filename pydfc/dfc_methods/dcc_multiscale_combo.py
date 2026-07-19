"""
Multiscale DCC dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class DCC_MULTISCALE_COMBO(BaseDFCMethod):
    """Average DCC conditional correlations over multiple recent scales."""

    MEASURE_NAME = "DccMultiscalecombo"

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
        if self.params["windows"] is None:
            self.params["windows"] = [15, 30, 60]
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

    def _standardized_residuals(self, time_series):
        n_regions, n_time = time_series.shape
        lam = float(self.params["garch_lambda"])
        eps = time_series - time_series.mean(axis=1, keepdims=True)
        sigma2 = np.zeros((n_regions, n_time))
        sigma2[:, 0] = np.maximum(np.var(eps, axis=1), 1e-8)
        for tr in range(1, n_time):
            sigma2[:, tr] = lam * sigma2[:, tr - 1] + (1.0 - lam) * eps[:, tr - 1] ** 2
        return eps / np.sqrt(np.maximum(sigma2, 1e-12))

    def _window_qbar(self, Z, tr, window):
        start = max(0, tr - window + 1)
        segment = Z[:, start : tr + 1]
        if segment.shape[1] < 2:
            return np.eye(Z.shape[0])
        qbar = np.corrcoef(segment)
        qbar[np.isnan(qbar)] = 0
        qbar[np.diag_indices_from(qbar)] = 1
        return self._nearest_spd(qbar)

    def dFC(self, time_series, Fs):
        windows = [self._samples(window, Fs, minimum=2) for window in self.params["windows"]]
        min_periods = int(self.params["min_periods"])
        a = float(self.params["dcc_a"])
        b = float(self.params["dcc_b"])
        if not 0 <= a < 1 or not 0 <= b < 1 or a + b >= 1:
            raise ValueError("dcc_a and dcc_b must be non-negative and sum to less than 1.")

        Z = self._standardized_residuals(time_series)
        Qs = [self._window_qbar(Z, 0, window) for window in windows]
        FCSs = []
        TR_array = []
        for tr in range(1, time_series.shape[1]):
            matrices = []
            scale_weights = []
            z = Z[:, tr - 1]
            for idx, window in enumerate(windows):
                qbar = self._window_qbar(Z, tr, window)
                Qs[idx] = (1.0 - a - b) * qbar + a * np.outer(z, z) + b * Qs[idx]
                matrices.append(self._corr_from_cov(Qs[idx]))
                scale_weights.append(np.sqrt(min(window, tr + 1)))
            if tr >= min_periods - 1:
                FCSs.append(np.average(np.array(matrices), axis=0, weights=scale_weights))
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
