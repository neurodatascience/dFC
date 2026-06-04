"""
Positive-Negative Asymmetry FC (PNAFC) — novel dFC method.

Standard Pearson correlation treats all BOLD deviations symmetrically.
This method asks: is the coupling between regions i and j the same when
region i is in an up-state (above its mean) as when it is in a down-state?
The dFC value captures the asymmetry between up-state and down-state
conditional connectivity — a measure of state-dependent coupling directionality.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class POSITIVE_NEGATIVE_ASYMMETRY(BaseDFCMethod):
    """Windowed asymmetry of up-state vs. down-state conditional correlation.

    Within each sliding window, for each conditioning region i:
      • FC_up[i,j]  = Pearson corr(x_i, x_j) restricted to timepoints
                      where x_i > μ_i   (region i above its window mean)
      • FC_down[i,j] = Pearson corr(x_i, x_j) restricted to timepoints
                      where x_i ≤ μ_i  (region i below its window mean)

    The asymmetry matrix A[i,j] = |FC_up[i,j] − FC_down[i,j]| is asymmetric
    in i because the conditioning is on region i.  It is symmetrised as:

        FC[i,j] = (A[i,j] + A[j,i]) / 2

    A large value indicates that the coupling between i and j is qualitatively
    different during the "up" and "down" BOLD phases of either region.
    """

    MEASURE_NAME = "PositiveNegativeAsymmetryFC"

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
    def _conditional_corr(xi, xj, mask):
        """Pearson correlation of xi and xj over selected timepoints."""
        if mask.sum() < 3:
            return 0.0
        a = xi[mask]
        b = xj[mask]
        a = a - a.mean()
        b = b - b.mean()
        sa = np.sqrt((a**2).sum())
        sb = np.sqrt((b**2).sum())
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        return float(np.clip(np.dot(a, b) / (sa * sb), -1.0, 1.0))

    def _pna_matrix(self, data):
        """[R, R] positive-negative asymmetry matrix for [R, W] data."""
        n_regions, W = data.shape
        mu = data.mean(axis=1)  # [R]

        # Asymmetry conditioned on each row region
        A = np.zeros((n_regions, n_regions))
        for i in range(n_regions):
            up_mask = data[i] > mu[i]
            dn_mask = ~up_mask
            for j in range(n_regions):
                if i == j:
                    continue
                r_up = self._conditional_corr(data[i], data[j], up_mask)
                r_dn = self._conditional_corr(data[i], data[j], dn_mask)
                A[i, j] = abs(r_up - r_dn)

        # Symmetrise
        C = (A + A.T) / 2.0
        np.fill_diagonal(C, 1.0)
        return np.clip(C, 0.0, 1.0)

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        _, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            FCSs.append(self._pna_matrix(seg))
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
