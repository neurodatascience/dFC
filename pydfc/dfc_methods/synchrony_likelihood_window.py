"""
Windowed Synchrony Likelihood (SL) dFC method.

Reference: Stam & van Dijk (2002). Synchronization likelihood: an unbiased
measure of generalized synchronization in multivariate data sets.
Physica D, 163(3-4), 236-251. doi:10.1016/S0167-2789(01)00386-4

Application to fMRI: Stam & van Straaten (2012). The organization of
physiological brain networks. Clin Neurophysiol, 123(6), 1067-1087.
doi:10.1016/j.clinph.2012.01.011
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class SYNCHRONY_LIKELIHOOD_WINDOW(BaseDFCMethod):
    """Sliding-window nonlinear synchrony based on joint phase-space recurrence.

    Each region's BOLD is delay-embedded into an m-dimensional phase-space
    trajectory.  Within each window, for every reference time t₀ and every
    region i, a "hit" is counted when the embedded state at time t is within
    radius r_i of the reference state (Theiler correction applied to exclude
    temporal neighbours).  r_i is chosen adaptively so that the average hit
    probability equals ref_prob.

    Synchrony Likelihood for the pair (i, j) is:

        SL_ij = P(hit_i AND hit_j) / (P(hit_i) × P(hit_j))

    A value near 1 indicates no more joint recurrences than expected by
    chance; values > 1 indicate nonlinear generalized synchrony.  The matrix
    is symmetrised and mapped to [0, 1] by capping at an empirical maximum.
    """

    MEASURE_NAME = "SynchronyLikelihoodWindow"

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
            "embed_dim",
            "embed_lag",
            "ref_prob",
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

        if self.params["W"] is None:
            self.params["W"] = 30
        if self.params["n_overlap"] is None:
            self.params["n_overlap"] = 0.0
        if self.params["embed_dim"] is None:
            self.params["embed_dim"] = 2
        if self.params["embed_lag"] is None:
            self.params["embed_lag"] = 1
        if self.params["ref_prob"] is None:
            self.params["ref_prob"] = 0.05

    @property
    def measure_name(self):
        return self.params["measure_name"]

    @staticmethod
    def _embed(x, m, lag):
        """Delay-embed scalar time series x into m-dimensional vectors."""
        T = len(x) - (m - 1) * lag
        if T <= 0:
            return x[:, np.newaxis]
        return np.stack([x[k * lag : k * lag + T] for k in range(m)], axis=1)

    @staticmethod
    def _recurrence_matrix(X, ref_prob, theiler=1):
        """Boolean recurrence matrix with adaptive radius for hit_prob = ref_prob."""
        T = X.shape[0]
        # Pairwise L2 distances
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # [T, T, m]
        D = np.sqrt((diff**2).sum(axis=2))  # [T, T]
        # Exclude diagonal and Theiler strip from radius estimation
        mask = np.abs(np.arange(T)[:, None] - np.arange(T)[None, :]) > theiler
        valid_dists = D[mask]
        r = np.quantile(valid_dists, ref_prob) if len(valid_dists) else 0.0
        R = (D <= r) & mask
        return R.astype(float)

    def _sl_window(self, data_window):
        """Compute [R, R] SL matrix for one window of data [R, W]."""
        n_regions, W = data_window.shape
        m = int(self.params["embed_dim"])
        lag = int(self.params["embed_lag"])
        ref_prob = float(self.params["ref_prob"])
        theiler = max(1, lag)

        # Build per-region recurrence matrices
        R_mats = []
        for i in range(n_regions):
            Xi = self._embed(data_window[i], m, lag)
            Ri = self._recurrence_matrix(Xi, ref_prob, theiler)
            R_mats.append(Ri)

        T_emb = R_mats[0].shape[0]
        n_pairs = T_emb * (T_emb - 1)  # denominator (upper + lower triangle)

        # Marginal hit probabilities
        p = np.array([Ri.sum() / max(n_pairs, 1) for Ri in R_mats])

        SL = np.zeros((n_regions, n_regions))
        np.fill_diagonal(SL, 1.0)
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                p_ij = (R_mats[i] * R_mats[j]).sum() / max(n_pairs, 1)
                denom = p[i] * p[j]
                sl = (p_ij / denom) if denom > 1e-12 else 0.0
                # Map to [0, 1]: SL=1 is chance; typical range [1, 1/(ref_prob)]
                sl_norm = min(sl / (1.0 / ref_prob), 1.0) if sl > 0 else 0.0
                SL[i, j] = sl_norm
                SL[j, i] = sl_norm

        return SL

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        n_regions, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            C = self._sl_window(seg)
            FCSs.append(C)
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
