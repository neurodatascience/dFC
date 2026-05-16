"""
Oja subspace dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class OJA_SUBSPACE_CONNECTIVITY(BaseDFCMethod):
    """Online low-rank connectivity from Oja-style latent subspace learning."""

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
            "n_components",
            "learning_rate",
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
        self.params["measure_name"] = "OjaSubspaceConnectivity"
        self.params["is_state_based"] = False
        if self.params["half_life"] is None:
            self.params["half_life"] = 30
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10
        if self.params["n_components"] is None:
            self.params["n_components"] = 5
        if self.params["learning_rate"] is None:
            self.params["learning_rate"] = 0.03

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
        min_periods = int(self.params["min_periods"])
        n_regions = time_series.shape[0]
        n_components = min(int(self.params["n_components"]), n_regions)
        rng = np.random.default_rng()
        basis = rng.normal(size=(n_regions, n_components))
        basis, _ = np.linalg.qr(basis)
        mean = np.zeros(n_regions)
        covariance = np.eye(n_regions)
        alpha = self._alpha_from_half_life(Fs)
        FCSs = []
        TR_array = []
        for tr in range(time_series.shape[1]):
            sample = time_series[:, tr]
            mean = (1.0 - alpha) * mean + alpha * sample
            centered = sample - mean
            covariance = (1.0 - alpha) * covariance + alpha * np.outer(centered, centered)
            scores = basis.T @ centered
            reconstruction = basis @ scores
            basis = basis + self.params["learning_rate"] * np.outer(
                centered - reconstruction, scores
            )
            basis, _ = np.linalg.qr(basis)
            projection = basis @ basis.T
            low_rank_covariance = projection @ covariance @ projection.T
            residual_variance = np.maximum(np.diag(covariance - low_rank_covariance), 0)
            reconstructed_covariance = low_rank_covariance + np.diag(residual_variance)
            if tr >= min_periods - 1:
                FCSs.append(self._corr_from_cov(reconstructed_covariance))
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
