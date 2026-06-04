"""
Leading Eigenvector Dynamics (LED) dFC method.

Reference: Leonardi & Van De Ville (2015). On spurious and real fluctuations of
dynamic functional connectivity during rest. NeuroImage, 114, 430-436.
doi:10.1016/j.neuroimage.2015.04.004
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class LEADING_EIGENVECTOR_DYNAMICS(BaseDFCMethod):
    """Rank-1 dFC estimate from the leading eigenvector of the windowed FC matrix.

    Within each sliding window the standard Pearson correlation matrix C is
    computed, then eigendecomposed.  The dFC estimate at that window centre is
    the rank-1 reconstruction v vᵀ where v is the eigenvector belonging to the
    largest eigenvalue.  This low-rank projection retains the dominant mode of
    co-fluctuation while suppressing the noise contained in smaller eigenmodes,
    yielding a smoother and more robust instantaneous FC estimate.
    """

    MEASURE_NAME = "LeadingEigenvectorDynamics"

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
        for params_name in self.params_name_lst:
            self.params[params_name] = params.get(params_name, None)

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
    def _rank1_from_eigvec(v):
        """Symmetric rank-1 correlation-scale matrix from unit eigenvector v."""
        r1 = np.outer(v, v)
        diag = np.sqrt(np.abs(np.diag(r1)))
        denom = np.outer(diag, diag)
        denom[denom == 0] = 1.0
        r1 = r1 / denom
        np.fill_diagonal(r1, 1.0)
        return np.clip(r1, -1.0, 1.0)

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
            vals, vecs = np.linalg.eigh(C)
            v = vecs[:, -1]  # eigenvector of the largest eigenvalue
            FCSs.append(self._rank1_from_eigvec(v))
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
