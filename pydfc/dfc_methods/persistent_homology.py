"""
Persistent Homology Connectivity (PHC) — from algebraic topology / TDA.

Source domain: Carlsson (2009). Topology and data. Bull. Amer. Math. Soc.,
46(2), 255-308. Applied to brain networks in Petri et al. (2014), Nature
Communications.

Within each window a Vietoris-Rips filtration is built over the correlation
distance matrix d_ij = 1 − |C_ij|.  The H0 bottleneck distance between
nodes i and j — the maximum edge weight on the minimum-spanning-tree path
from i to j — gives the "topological connectivity": a value that rewards
strong hub-mediated pathways, not just direct pairwise correlation.
"""

import time

import numpy as np
import scipy.sparse
from scipy.sparse.csgraph import minimum_spanning_tree

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class PERSISTENT_HOMOLOGY(BaseDFCMethod):
    """Windowed MST bottleneck distance as a topologically filtered dFC measure.

    For each sliding window the pairwise correlation distance matrix
    D[i,j] = 1 − |C_ij| is computed and its minimum spanning tree (MST)
    is extracted.  The H0 persistent homology bottleneck distance between
    nodes i and j equals the maximum edge weight on the unique MST path
    connecting them.  Converting to similarity:

        FC[i,j] = 1 − bottleneck(i, j) / max(D)

    This differs from raw correlation in two ways:
      • Indirect hub-mediated paths elevate FC between nodes that are
        weakly directly coupled but strongly hub-connected.
      • Spurious weak edges (noise) are penalised because they inflate
        the MST path weight.
    """

    MEASURE_NAME = "PersistentHomologyFC"

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
    def _mst_bottleneck(D):
        """Bottleneck distance matrix from MST of distance matrix D."""
        n = D.shape[0]
        # Build MST (scipy returns upper-triangle sparse)
        mst_sparse = minimum_spanning_tree(scipy.sparse.csr_matrix(D))
        mst = mst_sparse.toarray()
        mst = mst + mst.T  # symmetrise

        # Initialise bottleneck matrix from MST edges
        B = np.full((n, n), np.inf)
        np.fill_diagonal(B, 0.0)
        mask = mst > 0
        B[mask] = mst[mask]

        # Vectorised Floyd-Warshall for min-bottleneck (max-edge) paths
        for k in range(n):
            # Candidate via k: max(B[i,k], B[k,j])
            via_k = np.maximum(B[:, k : k + 1], B[k : k + 1, :])
            B = np.minimum(B, via_k)

        # Remaining inf means no path (shouldn't happen for complete graph)
        max_d = (
            B[np.isfinite(B) & (B > 0)].max() if np.any(np.isfinite(B) & (B > 0)) else 1.0
        )
        return B, max_d

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        n_regions, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            C = np.corrcoef(seg)
            C[np.isnan(C)] = 0.0
            np.fill_diagonal(C, 1.0)

            D = 1.0 - np.abs(C)
            np.fill_diagonal(D, 0.0)

            B, max_d = self._mst_bottleneck(D)

            FC = 1.0 - B / max(max_d, 1e-12)
            FC[~np.isfinite(FC)] = 0.0
            np.fill_diagonal(FC, 1.0)
            FC = np.clip(FC, 0.0, 1.0)

            FCSs.append(FC)
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
