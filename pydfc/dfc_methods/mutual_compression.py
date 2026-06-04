"""
Mutual Compression FC (MCFC) — novel dFC method.

Two signals share information if their joint sequence is more compressible
than their individual sequences.  This method quantifies pairwise dynamic
functional coupling as the "compression savings" when encoding region i
and region j together rather than separately, using the Lempel-Ziv 1976
complexity of binarized BOLD signals within sliding windows.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class MUTUAL_COMPRESSION(BaseDFCMethod):
    """Windowed Lempel-Ziv compression savings as a dFC measure.

    For a sliding window, each region's BOLD signal is binarized at its
    within-window median.  The Lempel-Ziv 1976 complexity C(s) (number of
    novel substrings encountered on one pass through the sequence) is then
    computed for each region individually and for each pair's concatenated
    sequence.  The mutual compression index is:

        MCI[i,j] = (C(s_i) + C(s_j) − C(s_i ‖ s_j)) / max(C(s_i), C(s_j))

    Positive values indicate that the joint sequence is more compressible
    than its parts — regions share repeating co-activation patterns.
    The measure is bounded in [0, 1], symmetric, and free of distributional
    assumptions.
    """

    MEASURE_NAME = "MutualCompressionFC"

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
    def _lz76(seq):
        """Lempel-Ziv 76 complexity (number of novel substrings, not normalised)."""
        n = len(seq)
        if n == 0:
            return 1
        i, l, k, c, kmax = 0, 1, 1, 1, 1
        while l + k <= n:
            if seq[i + k - 1] == seq[l + k - 1]:
                k += 1
            else:
                kmax = max(kmax, k)
                i += 1
                if i == l:
                    c += 1
                    l += kmax
                    i = 0
                    k = 1
                    kmax = 1
                else:
                    k = 1
        return c + (1 if k > 1 else 0)

    def _mci_matrix(self, data):
        """[R, R] mutual compression index for [R, W] data."""
        n_regions, W = data.shape
        # Binarise at median
        binary = (data >= np.median(data, axis=1, keepdims=True)).astype(np.int8)

        lz = np.array([self._lz76(binary[i]) for i in range(n_regions)])

        C = np.zeros((n_regions, n_regions))
        np.fill_diagonal(C, 1.0)
        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                joint = np.concatenate([binary[i], binary[j]])
                lz_ij = self._lz76(joint)
                norm = max(lz[i], lz[j], 1)
                mci = (lz[i] + lz[j] - lz_ij) / norm
                mci = float(np.clip(mci, 0.0, 1.0))
                C[i, j] = mci
                C[j, i] = mci

        return C

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        _, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            FCSs.append(self._mci_matrix(seg))
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
