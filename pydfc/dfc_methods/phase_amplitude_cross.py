"""
Phase-Amplitude Cross-Region FC (PAFC) — novel dFC method.

Standard AEC correlates amplitude envelopes; standard phase methods
correlate phases.  This method crosses them: it asks whether the
instantaneous phase of region i is correlated with the amplitude
envelope of region j — a spatial generalisation of cross-frequency
phase-amplitude coupling, applied as a between-region dFC measure.
"""

import time

import numpy as np
from scipy.signal import hilbert

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class PHASE_AMPLITUDE_CROSS(BaseDFCMethod):
    """Windowed cross-coupling between Hilbert phase and amplitude envelope.

    Within each sliding window:
      • phase_i(t) = cos(∠ hilbert(x_i(t)))  — instantaneous phase (as cosine)
      • amp_j(t)   = |hilbert(x_j(t))|        — instantaneous amplitude

    A raw asymmetric coupling matrix A[i,j] = corr(phase_i, amp_j) is
    computed, then symmetrised as FC[i,j] = (|A[i,j]| + |A[j,i]|) / 2.

    A large value for pair (i, j) indicates that the oscillatory phase
    of region i consistently predicts the energy level of region j (and/or
    vice versa), orthogonal to both amplitude-only and phase-only coupling.
    """

    MEASURE_NAME = "PhaseAmplitudeCrossFC"

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
    def _pac_matrix(data):
        """[R, R] symmetrised phase-amplitude cross-coupling for [R, W] data."""
        n_regions, W = data.shape
        analytic = hilbert(data, axis=1)  # [R, W]
        cos_phase = np.cos(np.angle(analytic))  # [R, W]
        amplitude = np.abs(analytic)  # [R, W]

        # Standardise each signal to zero mean, unit std
        def _standardise(x):
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True)
            sd = np.where(sd > 1e-12, sd, 1.0)
            return (x - mu) / sd

        cp = _standardise(cos_phase)  # [R, W]
        am = _standardise(amplitude)  # [R, W]

        # A[i,j] = corr(phase_i, amp_j) = (cp_i · am_j) / W
        A = (cp @ am.T) / max(W - 1, 1)  # [R, R]

        # Symmetrise absolute values
        C = (np.abs(A) + np.abs(A.T)) / 2.0
        np.fill_diagonal(C, 1.0)
        return np.clip(C, 0.0, 1.0)

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        _, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            FCSs.append(self._pac_matrix(seg))
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
