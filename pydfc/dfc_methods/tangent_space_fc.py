"""
Tangent Space FC (TangentFC) — novel dFC method.

The space of symmetric positive definite (SPD) matrices is a curved
Riemannian manifold.  Ordinary subtraction of FC matrices mixes static
and dynamic components in a geometrically ill-defined way.  This method
maps each windowed FC matrix into the tangent space at the group-level
geometric mean FC, providing a linearised, mean-centred dFC representation
that is invariant to the dominant static connectivity pattern.
"""

import time

import numpy as np
from scipy.linalg import logm, sqrtm

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class TANGENT_SPACE_FC(BaseDFCMethod):
    """Riemannian tangent-space projection of windowed FC matrices.

    estimate_FCS computes the group-level geometric mean connectivity M̄
    (approximated here as the Euclidean mean of windowed FC matrices, then
    projected to the nearest SPD matrix).

    For each subject window with FC matrix S, the tangent-space projection is:

        T = M̄^{-½} S M̄^{-½}    (whitening)
        T_log = logm(T)           (Riemannian log-map to flat space)

    T_log is symmetric, zero-mean across windows, and captures dynamic
    deviations from the group mean in a geometrically principled way.
    It is rescaled element-wise to [-1, 1] for compatibility with the
    standard dFC matrix convention.
    """

    MEASURE_NAME = "TangentSpaceFC"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.mean_FC_ = None
        self.sqrt_inv_mean_ = None

        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "W",
            "n_overlap",
            "normalization",
            "num_subj",
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

    def _windowed_fc(self, data, Fs):
        """Compute list of FC matrices for sliding windows of [R, T] data."""
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        _, T = data.shape
        matrices, centers = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = data[:, l : l + W_samples]
            C = np.corrcoef(seg)
            C[np.isnan(C)] = 0.0
            np.fill_diagonal(C, 1.0)
            matrices.append(C)
            centers.append(int(l + W_samples // 2))
        return matrices, centers

    @staticmethod
    def _nearest_spd(A):
        """Project a symmetric matrix to the nearest SPD matrix."""
        A = (A + A.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-6)
        return eigvecs @ np.diag(eigvals) @ eigvecs.T

    def estimate_FCS(self, time_series):
        assert type(time_series) is TIME_SERIES, "must be TIME_SERIES"
        time_series = self.manipulate_time_series4FCS(time_series)
        Fs = time_series.Fs

        tic = time.time()

        all_matrices = []
        for subj_id in time_series.subj_id_lst:
            subj_data = time_series.get_subj_ts(subjs_id=subj_id).data
            mats, _ = self._windowed_fc(subj_data, Fs)
            all_matrices.extend(mats)

        # Group mean FC (Euclidean approximation)
        mean_FC = np.mean(all_matrices, axis=0)
        self.mean_FC_ = self._nearest_spd(mean_FC)

        # Pre-compute M^{-1/2}
        sqrt_mean = np.real(sqrtm(self.mean_FC_))
        try:
            self.sqrt_inv_mean_ = np.linalg.inv(sqrt_mean)
        except np.linalg.LinAlgError:
            self.sqrt_inv_mean_ = np.eye(self.mean_FC_.shape[0])

        self.set_FCS_fit_time(time.time() - tic)
        return self

    def _project_to_tangent(self, S):
        """Map windowed FC matrix S to tangent space at mean_FC_."""
        n = S.shape[0]
        S_spd = self._nearest_spd(S)
        T = self.sqrt_inv_mean_ @ S_spd @ self.sqrt_inv_mean_
        T = self._nearest_spd(T)
        try:
            T_log = np.real(logm(T))
        except Exception:
            T_log = T - np.eye(n)  # first-order fallback
        T_log = (T_log + T_log.T) / 2.0
        # Rescale to [-1, 1]
        abs_max = np.abs(T_log[np.triu_indices(n, k=1)]).max()
        if abs_max > 1e-12:
            T_log = T_log / abs_max
        np.fill_diagonal(T_log, 1.0)
        return np.clip(T_log, -1.0, 1.0)

    def estimate_dFC(self, time_series):
        assert len(time_series.subj_id_lst) == 1, "one subject per call"
        assert type(time_series) is TIME_SERIES, "must be TIME_SERIES"

        time_series = self.manipulate_time_series4dFC(time_series)

        tic = time.time()
        matrices, centers = self._windowed_fc(time_series.data, time_series.Fs)

        if self.mean_FC_ is None:
            # Fallback: use subject-level mean if estimate_FCS was not called
            mean_FC = np.mean(matrices, axis=0)
            self.mean_FC_ = self._nearest_spd(mean_FC)
            from scipy.linalg import sqrtm as _sqrtm

            sqrt_mean = np.real(_sqrtm(self.mean_FC_))
            try:
                self.sqrt_inv_mean_ = np.linalg.inv(sqrt_mean)
            except np.linalg.LinAlgError:
                self.sqrt_inv_mean_ = np.eye(self.mean_FC_.shape[0])

        FCSs = np.array([self._project_to_tangent(S) for S in matrices])
        TR_array = np.array(centers)

        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.info_dict)
        return dFC
