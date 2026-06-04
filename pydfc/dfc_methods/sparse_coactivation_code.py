"""
Sparse Co-activation Code Connectivity (SCCC) — from sparse coding / CS.

Source domain: Olshausen & Field (1996). Emergence of simple-cell receptive
field properties by learning a sparse code for natural images. Nature, 381,
607-609.  Aharon et al. (2006). K-SVD: An algorithm for designing
overcomplete dictionaries for sparse representation. IEEE Trans. Signal
Process., 54(11), 4311-4322.

The brain may encode information in sparse patterns of co-active regions.
A dictionary D of K activity atoms is learned from the group-level BOLD
data.  At each timepoint the BOLD vector x(t) is sparsely reconstructed as
x̃(t) = D α(t) where α(t) is a sparse code.  The sparse reconstruction
denoises the BOLD signal while preserving coherent co-activation structure.
Windowed correlation of the reconstructed signals x̃_i(t) captures coupling
that is mediated by shared dictionary atoms — connectivity as participation
in the same latent activity patterns.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class SPARSE_COACTIVATION_CODE(BaseDFCMethod):
    """Group-level dictionary learning + per-subject sparse-code correlation.

    estimate_FCS (group level):
      Learns a dictionary D ∈ ℝ^{K×R} from all subjects' BOLD data via
      sklearn DictionaryLearning with a sparsity-inducing LASSO penalty.

    estimate_dFC (per subject):
      1. Sparse-code each TR: α(t) = argmin ½‖x(t) − Dα‖² + λ‖α‖₁
      2. Reconstruct: x̃(t) = D α(t)   (sparse denoised BOLD, same shape)
      3. Windowed Pearson correlation of x̃_i(t) across the window

    Unlike NMF_STATES (which learns a basis for FC matrices), SCCC learns a
    basis for BOLD *activity patterns* and computes connectivity on the
    sparse reconstruction rather than the raw signal.
    """

    MEASURE_NAME = "SparseCoactivationCodeFC"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.dictionary_ = None

        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "W",
            "n_overlap",
            "n_atoms",
            "dict_alpha",
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
        if self.params["n_atoms"] is None:
            self.params["n_atoms"] = 20
        if self.params["dict_alpha"] is None:
            self.params["dict_alpha"] = 0.1

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def estimate_FCS(self, time_series):
        assert type(time_series) is TIME_SERIES, "must be TIME_SERIES"
        time_series = self.manipulate_time_series4FCS(time_series)

        tic = time.time()

        from sklearn.decomposition import DictionaryLearning

        all_data = []
        for subj_id in time_series.subj_id_lst:
            subj_data = time_series.get_subj_ts(subjs_id=subj_id).data  # [R, T]
            all_data.append(subj_data.T)  # [T, R]
        X = np.vstack(all_data)  # [total_T, R]

        n_atoms = int(self.params["n_atoms"])
        alpha = float(self.params["dict_alpha"])

        dl = DictionaryLearning(
            n_components=n_atoms,
            alpha=alpha,
            max_iter=200,
            random_state=0,
            fit_algorithm="cd",
            transform_algorithm="lasso_cd",
            n_jobs=1,
        )
        dl.fit(X)
        self.dictionary_ = dl.components_  # [n_atoms, R]

        self.set_FCS_fit_time(time.time() - tic)
        return self

    def _sparse_reconstruct(self, data):
        """Sparse-reconstruct [R, T] data → [R, T] via learned dictionary."""
        from sklearn.decomposition import SparseCoder

        if self.dictionary_ is None:
            return data  # fallback: no dictionary learned

        coder = SparseCoder(
            dictionary=self.dictionary_,
            transform_algorithm="lasso_cd",
            transform_alpha=float(self.params["dict_alpha"]),
            n_jobs=1,
        )
        codes = coder.transform(data.T)  # [T, n_atoms]
        reconstruction = codes @ self.dictionary_  # [T, R]
        return reconstruction.T  # [R, T]

    def estimate_dFC(self, time_series):
        assert len(time_series.subj_id_lst) == 1, "one subject per call"
        assert type(time_series) is TIME_SERIES, "must be TIME_SERIES"

        time_series = self.manipulate_time_series4dFC(time_series)

        tic = time.time()

        recon = self._sparse_reconstruct(time_series.data)  # [R, T]
        n_regions, T = recon.shape

        W_samples = int(self.params["W"] * time_series.Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = recon[:, l : l + W_samples]
            C = np.corrcoef(seg)
            C[np.isnan(C)] = 0.0
            np.fill_diagonal(C, 1.0)
            FCSs.append(np.clip(C, -1.0, 1.0))
            TR_array.append(int(l + W_samples // 2))

        self.set_dFC_assess_time(time.time() - tic)

        dFC_out = DFC(measure=self)
        dFC_out.set_dFC(
            FCSs=np.array(FCSs),
            TR_array=np.array(TR_array),
            TS_info=time_series.info_dict,
        )
        return dFC_out
