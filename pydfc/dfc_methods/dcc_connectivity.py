"""
Dynamic Conditional Correlation Connectivity (DCC-FC) — from financial econometrics.

Source: Engle (2002). Dynamic conditional correlation. Journal of Business &
Economic Statistics, 20(3), 339-350.

BOLD signals, like asset returns, are heteroskedastic: their conditional
variance clusters over time.  DCC-GARCH captures this with a per-region
EWMA variance model and a DCC update for the conditional correlation matrix,
giving a per-TR estimate R(t) that explicitly accounts for time-varying
signal volatility — unlike standard correlation which treats all timepoints
as equally noisy.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class DCC_CONNECTIVITY(BaseDFCMethod):
    """Per-TR conditional correlation via EWMA-GARCH + DCC recursion.

    Step 1 — Per-region EWMA variance (RiskMetrics GARCH approximation):
        σ²_i(t) = λ · σ²_i(t−1) + (1−λ) · ε²_i(t−1)

    Step 2 — Standardise residuals:  z_i(t) = ε_i(t) / σ_i(t)

    Step 3 — DCC update (Engle 2002):
        Q(t) = (1−a−b) Q̄ + a z(t−1)z(t−1)ᵀ + b Q(t−1)

    Step 4 — Rescale to correlation:
        R_ij(t) = Q_ij(t) / √(Q_ii(t) · Q_jj(t))

    Unlike windowed Pearson correlation, DCC up-weights low-volatility periods
    and down-weights high-variance bursts, capturing "conditional" coupling.
    """

    MEASURE_NAME = "DCCConnectivity"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "garch_lambda",
            "dcc_a",
            "dcc_b",
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

        if self.params["garch_lambda"] is None:
            self.params["garch_lambda"] = 0.94  # RiskMetrics standard
        if self.params["dcc_a"] is None:
            self.params["dcc_a"] = 0.05
        if self.params["dcc_b"] is None:
            self.params["dcc_b"] = 0.85

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def dFC(self, time_series, Fs):
        n_regions, T = time_series.shape
        lam = float(self.params["garch_lambda"])
        a = float(self.params["dcc_a"])
        b = float(self.params["dcc_b"])

        # Demeaned residuals
        eps = time_series - time_series.mean(axis=1, keepdims=True)

        # EWMA conditional variance: σ²_i(t) = λ σ²_i(t-1) + (1-λ) ε²_i(t-1)
        sigma2 = np.zeros((n_regions, T))
        sigma2[:, 0] = np.var(eps, axis=1)
        for t in range(1, T):
            sigma2[:, t] = lam * sigma2[:, t - 1] + (1.0 - lam) * eps[:, t - 1] ** 2
        sigma2 = np.maximum(sigma2, 1e-12)

        # Standardised residuals
        Z = eps / np.sqrt(sigma2)  # [R, T]

        # Unconditional correlation of standardised residuals
        Q_bar = np.corrcoef(Z)
        Q_bar[np.isnan(Q_bar)] = 0.0
        np.fill_diagonal(Q_bar, 1.0)
        Q_bar = self._nearest_spd(Q_bar)

        # DCC recursion
        Q = Q_bar.copy()
        FCSs, TR_array = [], []
        for t in range(1, T):
            z = Z[:, t - 1]  # [R]
            Q = (1.0 - a - b) * Q_bar + a * np.outer(z, z) + b * Q
            d = np.sqrt(np.maximum(np.diag(Q), 1e-12))
            R = Q / np.outer(d, d)
            np.fill_diagonal(R, 1.0)
            FCSs.append(np.clip(R, -1.0, 1.0))
            TR_array.append(t)

        return np.array(FCSs), np.array(TR_array)

    @staticmethod
    def _nearest_spd(A):
        A = (A + A.T) / 2.0
        vals, vecs = np.linalg.eigh(A)
        return vecs @ np.diag(np.maximum(vals, 1e-6)) @ vecs.T

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
