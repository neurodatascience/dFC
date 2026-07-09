"""
Quantum Mutual Information Connectivity (QMIC) — from quantum information theory.

Source domain: Nielsen & Chuang (2000). Quantum Computation and Quantum
Information. Cambridge University Press.

A normalised covariance matrix has trace 1 and non-negative eigenvalues,
making it formally identical to a quantum density matrix ρ.  The quantum
mutual information I(i:j) = S(ρ_i) + S(ρ_j) − S(ρ_ij) — where S is the
von Neumann entropy −Tr(ρ log ρ) — measures how much the 2×2 joint state
of pair (i,j) differs from a product (independent) state.  Unlike linear
correlation, QMIC is sensitive to off-diagonal structure regardless of sign,
captures higher-order Gaussian dependencies, and is bounded in [0, 1].
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class QUANTUM_MUTUAL_INFORMATION(BaseDFCMethod):
    """Windowed von Neumann mutual information of normalised covariance pairs.

    Within each window, the sample covariance C is normalised to a density
    matrix ρ = C / Tr(C).  For each pair (i, j) the 2×2 marginal is:

        ρ_ij = [[ρ_ii, ρ_ij], [ρ_ji, ρ_jj]] / (ρ_ii + ρ_jj)

    The quantum mutual information is then:

        QMI(i,j) = S_product − S_joint

    where S_product = −p log p − q log q  (p = ρ_ii_norm, q = ρ_jj_norm)
    is the von Neumann entropy of the hypothetical product state, and
    S_joint = −Σ λ_k log λ_k  (eigenvalues of ρ_ij_norm)  is the
    actual joint entropy.  The result is bounded in [0, log 2] and
    normalised to [0, 1]; it equals zero iff the two regions are
    uncorrelated within the window.
    """

    MEASURE_NAME = "QuantumMutualInformationFC"

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
    def _vn_entropy_2x2(mat):
        """Von Neumann entropy −Σ λ log λ of a 2×2 density matrix."""
        eigvals = np.linalg.eigvalsh(mat)
        eigvals = np.maximum(eigvals, 1e-12)
        eigvals = eigvals / eigvals.sum()  # ensure normalised
        return -float(np.sum(eigvals * np.log(eigvals)))

    def _qmi_matrix(self, data):
        """[R, R] QMI matrix for [R, W] window data."""
        n_regions, W = data.shape
        C = np.cov(data)
        tr_C = np.trace(C)
        if tr_C < 1e-12:
            return np.eye(n_regions)
        rho = C / tr_C

        FC = np.zeros((n_regions, n_regions))
        np.fill_diagonal(FC, 1.0)
        log2 = np.log(2.0)

        for i in range(n_regions):
            for j in range(i + 1, n_regions):
                # 2×2 marginal
                rho_ij = np.array([[rho[i, i], rho[i, j]], [rho[j, i], rho[j, j]]])
                tr_ij = rho_ij[0, 0] + rho_ij[1, 1]
                if tr_ij < 1e-12:
                    continue
                rho_ij_norm = rho_ij / tr_ij

                p = rho_ij_norm[0, 0]  # marginal probability for i
                q = rho_ij_norm[1, 1]  # marginal probability for j  (= 1-p)

                # Product-state entropy (independence baseline)
                p = np.clip(p, 1e-12, 1.0 - 1e-12)
                q = np.clip(q, 1e-12, 1.0 - 1e-12)
                s_product = -p * np.log(p) - q * np.log(q)

                # Joint von Neumann entropy
                s_joint = self._vn_entropy_2x2(rho_ij_norm)

                qmi = max(0.0, s_product - s_joint) / log2
                FC[i, j] = np.clip(qmi, 0.0, 1.0)
                FC[j, i] = FC[i, j]

        return FC

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        _, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            FCSs.append(self._qmi_matrix(seg))
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
