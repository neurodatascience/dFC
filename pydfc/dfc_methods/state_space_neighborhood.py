"""
State-Space Neighborhood FC (SSNFC) — novel dFC method.

Rather than averaging BOLD signals over a temporal window, this method
finds the k timepoints in the entire recording whose whole-brain state
vector is most similar to the current state, then computes the
correlation matrix over those k nearest neighbors in state space.

Each timepoint gets its own FC matrix based on *where* in state space
the brain is, not *when* the measurement was taken.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class STATE_SPACE_NEIGHBORHOOD(BaseDFCMethod):
    """Per-timepoint FC computed over k nearest neighbors in BOLD state space.

    For each timepoint t, the k timepoints with the smallest Euclidean
    distance ‖x(t') − x(t)‖ are found (excluding a Theiler window of
    ±theiler TRs to avoid temporal autocorrelation bias).  The Pearson
    correlation matrix computed over these k state-space neighbors is
    the instantaneous dFC estimate at t.

    Unlike windowed methods, the "averaging region" is defined by
    similarity in brain state rather than proximity in time.  Timepoints
    that are far apart in time but share the same brain configuration will
    contribute to each other's FC estimate, capturing attractor-relative
    connectivity.
    """

    MEASURE_NAME = "StateSpaceNeighborhoodFC"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "k_neighbors",
            "theiler",
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

        if self.params["k_neighbors"] is None:
            self.params["k_neighbors"] = 20
        if self.params["theiler"] is None:
            self.params["theiler"] = 5

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def dFC(self, time_series, Fs):
        n_regions, T = time_series.shape
        k = int(self.params["k_neighbors"])
        theiler = int(self.params["theiler"])

        k = min(k, T - 2 * theiler - 1)
        if k < 3:
            # Not enough data for meaningful FC; return identity matrices
            return (np.tile(np.eye(n_regions), (T, 1, 1)), np.arange(T))

        # Pairwise squared Euclidean distances [T, T]
        # ||x(t) - x(t')||^2 = ||x(t)||^2 + ||x(t')||^2 - 2 x(t)·x(t')
        X = time_series.T  # [T, R]
        sq_norms = np.sum(X**2, axis=1)  # [T]
        D2 = sq_norms[:, None] + sq_norms[None, :] - 2 * (X @ X.T)  # [T, T]
        D2 = np.maximum(D2, 0.0)

        FCSs, TR_array = [], []
        for t in range(T):
            # Mask out Theiler window
            d = D2[t].copy()
            lo = max(0, t - theiler)
            hi = min(T, t + theiler + 1)
            d[lo:hi] = np.inf
            # k nearest neighbors
            nn_idx = np.argpartition(d, k)[:k]
            seg = time_series[:, nn_idx]  # [R, k]
            C = np.corrcoef(seg)
            C[np.isnan(C)] = 0.0
            np.fill_diagonal(C, 1.0)
            FCSs.append(np.clip(C, -1.0, 1.0))
            TR_array.append(t)

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
