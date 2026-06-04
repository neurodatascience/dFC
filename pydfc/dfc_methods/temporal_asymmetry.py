"""
Temporal Asymmetry FC (TAFC) — novel dFC method.

For each region pair (i, j) the forward lagged correlation r(i→j, τ)
and backward lagged correlation r(j→i, τ) are both computed within each
sliding window.  The dFC value is |r(i→j, τ) − r(j→i, τ)|: the absolute
asymmetry of directed influence.  Pairs with high asymmetry have one-way
driving dynamics; pairs near zero are bidirectionally symmetric.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class TEMPORAL_ASYMMETRY(BaseDFCMethod):
    """Absolute directed-coupling asymmetry at a fixed temporal lag.

    r_fwd[i,j] = corr(x_i[0:W-τ], x_j[τ:W])  — i leads j
    r_bwd[i,j] = corr(x_j[0:W-τ], x_i[τ:W])  — j leads i

    FC[i,j] = |r_fwd[i,j] − r_bwd[i,j]|

    The matrix is symmetric (|a−b| = |b−a|) and bounded in [0, 2].
    It is zero for bidirectionally symmetric coupling and maximal when
    influence is entirely one-directional.  Unlike Granger causality,
    no autoregressive model is fit; the measure is simply the signed
    difference of lagged Pearson correlations.
    """

    MEASURE_NAME = "TemporalAsymmetryFC"

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
            "lag",
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
        if self.params["lag"] is None:
            self.params["lag"] = 1

    @property
    def measure_name(self):
        return self.params["measure_name"]

    @staticmethod
    def _corr(a, b):
        """Pearson correlation between vectors a and b."""
        a = a - a.mean()
        b = b - b.mean()
        sa = np.sqrt((a**2).sum())
        sb = np.sqrt((b**2).sum())
        if sa < 1e-12 or sb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (sa * sb))

    def _asymmetry_matrix(self, data):
        """[R, R] asymmetry matrix for [R, W] window data."""
        n_regions, W = data.shape
        lag = int(self.params["lag"])
        if lag >= W:
            return np.zeros((n_regions, n_regions))

        # forward[i,j]: i leads j by lag
        r_fwd = np.zeros((n_regions, n_regions))
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    r_fwd[i, j] = self._corr(data[i, : W - lag], data[j, lag:])

        # backward is just r_fwd transposed
        r_bwd = r_fwd.T
        asym = np.abs(r_fwd - r_bwd)
        # Already symmetric: asym[i,j] = |r_fwd[i,j] - r_bwd[i,j]|
        #                               = |r_fwd[i,j] - r_fwd[j,i]|
        #                   asym[j,i] = |r_fwd[j,i] - r_fwd[i,j]| = asym[i,j] ✓
        np.fill_diagonal(asym, 0.0)
        return asym / 2.0  # normalise to [0, 1]

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        _, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            FCSs.append(self._asymmetry_matrix(seg))
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
