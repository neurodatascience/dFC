"""
Adaptive random Fourier feature dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class ADAPTIVE_RANDOM_FOURIER_COMBO(BaseDFCMethod):
    """Nonlinear random-feature dependence with volatility-adaptive forgetting."""

    MEASURE_NAME = "AdaptiveRandomFouriercombo"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "min_periods",
            "n_random_features",
            "alpha_min",
            "alpha_max",
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

        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10
        if self.params["n_random_features"] is None:
            self.params["n_random_features"] = 32
        if self.params["alpha_min"] is None:
            self.params["alpha_min"] = 0.02
        if self.params["alpha_max"] is None:
            self.params["alpha_max"] = 0.35

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _feature_map(self, samples):
        rng = np.random.default_rng(self.params["random_state"])
        n_features = max(int(self.params["n_random_features"]), 2)
        omega = rng.normal(size=n_features)
        phase = rng.uniform(0, 2 * np.pi, size=n_features)
        return np.sqrt(2.0 / n_features) * np.cos(
            samples[:, :, None] * omega + phase
        )

    def _instant_dependence(self, phi):
        phi = phi - np.mean(phi, axis=1, keepdims=True)
        norm = np.linalg.norm(phi, axis=1, keepdims=True)
        phi = np.divide(phi, norm, out=np.zeros_like(phi), where=norm > 0)
        instant = phi @ phi.T
        instant = 0.5 * (instant + instant.T)
        instant[np.diag_indices_from(instant)] = 1
        instant[np.isnan(instant)] = 0
        return instant

    def _adaptive_alpha(self, diff_energy, baseline, alpha_min, alpha_max, tr):
        local_energy = diff_energy[tr] / baseline
        adapt = local_energy / (1.0 + local_energy)
        return alpha_min + (alpha_max - alpha_min) * adapt

    def dFC(self, time_series, Fs):
        min_periods = int(self.params["min_periods"])
        alpha_min = float(self.params["alpha_min"])
        alpha_max = float(self.params["alpha_max"])
        if not 0 <= alpha_min <= alpha_max <= 1:
            raise ValueError(
                "alpha_min and alpha_max must satisfy "
                "0 <= alpha_min <= alpha_max <= 1."
            )

        diff_energy = np.zeros(time_series.shape[1])
        diff_energy[1:] = np.mean(np.abs(np.diff(time_series, axis=1)), axis=0)
        baseline = np.median(diff_energy[1 : max(min_periods, 2)]) + 1e-8
        features = self._feature_map(time_series)
        n_regions = time_series.shape[0]
        dependence = np.eye(n_regions)
        FCSs = []
        TR_array = []

        for tr in range(time_series.shape[1]):
            alpha = self._adaptive_alpha(
                diff_energy=diff_energy,
                baseline=baseline,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                tr=tr,
            )
            instant = self._instant_dependence(features[:, tr, :])
            dependence = (1.0 - alpha) * dependence + alpha * instant
            dependence = 0.5 * (dependence + dependence.T)
            dependence[np.diag_indices_from(dependence)] = 1
            dependence[np.isnan(dependence)] = 0
            if tr >= min_periods - 1:
                FCSs.append(dependence.copy())
                TR_array.append(tr)
            baseline = 0.98 * baseline + 0.02 * max(diff_energy[tr], 1e-8)

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
