"""
Volatility-Weighted FC (VWC) — novel dFC method.

Standard sliding-window correlation weights every timepoint equally.
This method weights each timepoint by its squared distance from the
window mean (its "volatility" contribution), so moments of high
co-fluctuation amplitude dominate the covariance estimate.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class VOLATILITY_WEIGHTED(BaseDFCMethod):
    """Sliding-window Pearson correlation with amplitude-deviation weighting.

    Within each window the contribution of timepoint t to the covariance
    estimate is weighted by w(t) = ‖x(t) − μ‖², the squared L2 distance
    of the multiregion BOLD state from the window mean.  High-amplitude
    co-fluctuation moments receive proportionally more weight, giving a
    covariance estimate that emphasises large, coordinated deviations from
    baseline.  The weighted covariance is normalised to a correlation matrix.
    """

    MEASURE_NAME = "VolatilityWeightedFC"

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
            "normalization",
            "num_select_nodes",
            "num_time_point",
            "Fs_ratio",
            "noise_ratio",
            "num_realization",
            "session",
        ]
        self.params = {}
        for p in self.params_name_lst:
            self.params[p] = params.get(p, None)

        self.params["measure_name"] = self.MEASURE_NAME
        self.params["is_state_based"] = False

        if self.params["W"] is None:
            self.params["W"] = 30
        if self.params["n_overlap"] is None:
            self.params["n_overlap"] = 0.0

    @property
    def measure_name(self):
        return self.params["measure_name"]

    @staticmethod
    def _weighted_correlation(data):
        """Volatility-weighted Pearson correlation for [n_regions, W] data."""
        mu = data.mean(axis=1, keepdims=True)
        residuals = data - mu  # [R, W]
        weights = np.sum(residuals**2, axis=0)  # [W] per-TR volatility
        w_sum = weights.sum()
        if w_sum < 1e-12:
            return np.corrcoef(data)

        weights = weights / w_sum  # normalise to sum=1
        # Weighted covariance: sum_t w_t * r_i(t) * r_j(t)
        wcov = (residuals * weights[np.newaxis, :]) @ residuals.T  # [R, R]
        # Weighted variance per region
        wvar = np.diag(wcov)
        denom = np.sqrt(np.outer(wvar, wvar))
        denom = np.where(denom > 1e-12, denom, 1.0)
        C = wcov / denom
        C[np.isnan(C)] = 0.0
        np.fill_diagonal(C, 1.0)
        return np.clip(C, -1.0, 1.0)

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        _, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            FCSs.append(self._weighted_correlation(seg))
            TR_array.append(int(l + W_samples // 2))

        return np.array(FCSs), np.array(TR_array)

    def estimate_FCS(self, time_series):
        return self

    def estimate_dFC(self, time_series):
        assert len(time_series.subj_id_lst) == 1, "one subject per call"
        assert type(time_series) is TIME_SERIES, "must be TIME_SERIES"

        time_series = self.manipulate_time_series4dFC(time_series)

        tic = time.time()
        FCSs, TR_array = self.dFC(time_series.data, time_series.Fs)
        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.info_dict)
        return dFC
