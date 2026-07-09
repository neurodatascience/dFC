"""
Non-negative Matrix Factorization States (NMF-States) dFC method.

Reference: Yousefi et al. (2021). Quasi-periodic patterns of intrinsic brain
activity in individuals and their relationship to global signal.
NeuroImage, 225, 117479. doi:10.1016/j.neuroimage.2020.117479

Also: Chai et al. (2017). Evolution of brain network dynamics in
neurodevelopment. Network Neuroscience, 1(1), 14-30.
doi:10.1162/NETN_a_00006
"""

import time

import numpy as np
from sklearn.decomposition import NMF

from ..dfc import DFC
from ..dfc_utils import SW_downsample
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class NMF_STATES(BaseDFCMethod):
    """FC states discovered by Non-negative Matrix Factorization of windowed FC.

    Group-level sliding-window FC matrices are vectorised (upper triangle),
    globally shifted to be non-negative, and stacked into a data matrix V.
    NMF decomposes V ≈ W H where H contains the latent FC patterns (states)
    and W their temporal activations.  At estimation time each subject's
    window is projected onto the learned components and assigned to the state
    with the highest activation.  NMF enforces non-negativity, yielding
    additive, parts-based FC components that do not cancel each other.
    """

    MEASURE_NAME = "NMFStates"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.nmf_ = None
        self._v_min = 0.0

        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "n_states",
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
        for params_name in self.params_name_lst:
            self.params[params_name] = params.get(params_name, None)

        self.params["measure_name"] = self.MEASURE_NAME
        self.params["is_state_based"] = True

        if self.params["n_states"] is None:
            self.params["n_states"] = 5
        if self.params["W"] is None:
            self.params["W"] = 30
        if self.params["n_overlap"] is None:
            self.params["n_overlap"] = 0.0

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _windowed_fc_vecs(self, data, Fs):
        """Vectorised upper-triangle FC for all windows of [R, T] data."""
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        n_regions, T = data.shape
        upper_idx = np.triu_indices(n_regions, k=1)
        vecs, trs = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = data[:, l : l + W_samples]
            C = np.corrcoef(seg)
            C[np.isnan(C)] = 0.0
            vecs.append(C[upper_idx])
            trs.append(int(l + W_samples // 2))
        return np.array(vecs), np.array(trs)

    @staticmethod
    def _vec_to_full(vec, n_regions):
        upper_idx = np.triu_indices(n_regions, k=1)
        C = np.zeros((n_regions, n_regions))
        C[upper_idx] = vec
        C += C.T
        np.fill_diagonal(C, 1.0)
        return C

    def estimate_FCS(self, time_series):
        assert type(time_series) is TIME_SERIES, "must be TIME_SERIES"
        time_series = self.manipulate_time_series4FCS(time_series)
        Fs = time_series.Fs
        n_regions = time_series.n_regions

        tic = time.time()

        all_vecs = []
        for subj_id in time_series.subj_id_lst:
            subj_data = time_series.get_subj_ts(subjs_id=subj_id).data
            vecs, _ = self._windowed_fc_vecs(subj_data, Fs)
            all_vecs.append(vecs)
        V = np.vstack(all_vecs)  # [total_windows, n_pairs]

        # Shift to non-negative for NMF
        self._v_min = float(V.min())
        V_nn = V - self._v_min

        self.nmf_ = NMF(
            n_components=int(self.params["n_states"]),
            max_iter=500,
            random_state=0,
        )
        W_coef = self.nmf_.fit_transform(V_nn)  # [windows, n_states]

        # Reconstruct state FC matrices (shift back)
        H = self.nmf_.components_  # [n_states, n_pairs]
        H_shifted = H + self._v_min
        self.FCS_ = np.array([self._vec_to_full(h, n_regions) for h in H_shifted])

        # Group-level Z for set_mean_activity
        self.Z = W_coef.argmax(axis=1)
        self._set_mean_activity_nmf(time_series)
        self.set_FCS_fit_time(time.time() - tic)
        return self

    def _set_mean_activity_nmf(self, time_series):
        """Mean window-averaged BOLD per NMF state."""
        TS_data = None
        for subject in time_series.subj_id_lst:
            subj_ts = time_series.get_subj_ts(subjs_id=subject)
            win_data = SW_downsample(
                data=subj_ts.data.T,
                Fs=time_series.Fs,
                W=self.params["W"],
                n_overlap=self.params["n_overlap"],
                tapered_window=False,
            ).T  # [n_regions, n_windows]
            TS_data = (
                win_data
                if TS_data is None
                else np.concatenate((TS_data, win_data), axis=1)
            )

        mean_act = []
        for i in np.unique(self.Z):
            ids = np.array([int(s == i) for s in self.Z])
            mean_act.append(np.average(TS_data, weights=ids, axis=1))
        self.mean_act = np.array(mean_act)

    def estimate_dFC(self, time_series):
        assert type(time_series) is TIME_SERIES, "must be TIME_SERIES"
        assert len(time_series.subj_id_lst) == 1, "one subject per call"

        time_series = self.manipulate_time_series4dFC(time_series)

        tic = time.time()

        vecs, TR_array = self._windowed_fc_vecs(time_series.data, time_series.Fs)
        V_nn = vecs - self._v_min
        W_coef = self.nmf_.transform(np.clip(V_nn, 0, None))  # [n_win, n_states]

        Z = W_coef.argmax(axis=1)
        row_sums = W_coef.sum(axis=1, keepdims=True)
        Z_proba = W_coef / np.where(row_sums > 0, row_sums, 1.0)

        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(
            FCSs=self.FCS_,
            FCS_idx=Z,
            FCS_proba=Z_proba,
            TS_info=time_series.info_dict,
            TR_array=TR_array,
        )
        return dFC
