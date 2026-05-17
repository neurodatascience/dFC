"""
Kalman-style covariance dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class KALMAN_COVARIANCE(BaseDFCMethod):
    MEASURE_NAME = "KalmanCovariance"
    """Recursive covariance tracking with a Kalman-style process floor."""

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
            "process_noise",
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
        self.params["measure_name"] = self.MEASURE_NAME
        self.params["is_state_based"] = False
        if self.params["half_life"] is None:
            self.params["half_life"] = 30
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10
        if self.params["process_noise"] is None:
            self.params["process_noise"] = 1e-4

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _alpha_from_half_life(self, Fs):
        if self.params["alpha"] is not None:
            return float(self.params["alpha"])
        half_life_samples = max(float(self.params["half_life"]) * Fs, 1.0)
        return 1.0 - np.exp(np.log(0.5) / half_life_samples)

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

    def dFC(self, time_series, Fs):
        alpha = self._alpha_from_half_life(Fs)
        min_periods = int(self.params["min_periods"])
        n_regions = time_series.shape[0]
        mean = time_series[:, 0].copy()
        covariance = np.eye(n_regions)
        FCSs = []
        TR_array = []
        for tr in range(1, time_series.shape[1]):
            sample = time_series[:, tr]
            innovation = sample - mean
            mean = mean + alpha * innovation
            covariance = (
                (1.0 - alpha) * covariance
                + alpha * np.outer(innovation, innovation)
                + self.params["process_noise"] * np.eye(n_regions)
            )
            if tr >= min_periods - 1:
                FCSs.append(self._corr_from_cov(covariance))
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
