"""
Random Fourier feature dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class RANDOM_FOURIER_DEPENDENCE(BaseDFCMethod):
    """Nonlinear edge dependence from random Fourier features of node activity."""

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
            "n_random_features",
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
        self.params["measure_name"] = "RandomFourierDependence"
        self.params["is_state_based"] = False
        if self.params["half_life"] is None:
            self.params["half_life"] = 30
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10
        if self.params["n_random_features"] is None:
            self.params["n_random_features"] = 32

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _alpha_from_half_life(self, Fs):
        if self.params["alpha"] is not None:
            return float(self.params["alpha"])
        half_life_samples = max(float(self.params["half_life"]) * Fs, 1.0)
        return 1.0 - np.exp(np.log(0.5) / half_life_samples)

    def _feature_map(self, samples):
        rng = np.random.default_rng()
        n_features = int(self.params["n_random_features"])
        omega = rng.normal(size=n_features)
        phase = rng.uniform(0, 2 * np.pi, size=n_features)
        return np.sqrt(2.0 / n_features) * np.cos(samples[:, :, None] * omega + phase)

    def dFC(self, time_series, Fs):
        min_periods = int(self.params["min_periods"])
        alpha = self._alpha_from_half_life(Fs)
        features = self._feature_map(time_series)
        n_regions = time_series.shape[0]
        dependence = np.eye(n_regions)
        FCSs = []
        TR_array = []
        for tr in range(time_series.shape[1]):
            phi = features[:, tr, :]
            phi = phi - np.mean(phi, axis=1, keepdims=True)
            norm = np.linalg.norm(phi, axis=1, keepdims=True)
            phi = np.divide(phi, norm, out=np.zeros_like(phi), where=norm > 0)
            instant = phi @ phi.T
            dependence = (1.0 - alpha) * dependence + alpha * instant
            dependence[np.diag_indices_from(dependence)] = 1
            if tr >= min_periods - 1:
                FCSs.append(dependence.copy())
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
