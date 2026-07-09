"""
Time-Reversal Asymmetry Connectivity (TRAC) — from stochastic thermodynamics.

Source domain: Roldan & Parrondo (2010). Estimating dissipation from
single stationary trajectories. PRL, 105, 150607.

A system in thermodynamic equilibrium is time-symmetric: its statistics
are the same forwards and backwards in time.  The joint time-reversal
asymmetry A_ij = E[(x_i(t+1) − x_i(t−1)) · x_j(t)] measures how much
the velocity of region i co-varies with the position of region j — a
quantity that is zero for time-symmetric (equilibrium) dynamics and
nonzero for irreversible (non-equilibrium) processes.  The symmetric
combination |A_ij + A_ji| captures the joint departure from time-symmetry.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class TIME_REVERSAL_ASYMMETRY(BaseDFCMethod):
    """Windowed joint time-reversal asymmetry as a dFC measure.

    Within each window, the central-difference velocity of region i is:
        v_i(t) = x_i(t+1) − x_i(t−1)

    The normalised cross-covariance A[i,j] = corr(v_i, x_j) is computed.
    The symmetric joint asymmetry matrix is:

        FC[i,j] = |A[i,j] + A[j,i]| / 2

    A[i,j] captures "how much does region i's velocity predict region j's
    position?" while A[j,i] captures the reverse.  Their sum is large when
    both directions show asymmetric velocity-position coupling — the
    signature of a coupled non-equilibrium process.  The measure is bounded
    in [0, 1] and zero for time-symmetric (e.g., white noise) inputs.
    """

    MEASURE_NAME = "TimeReversalAsymmetryFC"

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
    def _zscore_rows(x):
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        sd = np.where(sd > 1e-12, sd, 1.0)
        return (x - mu) / sd

    def _trac_matrix(self, data):
        """[R, R] TRAC matrix for [R, W] window data."""
        # Central-difference velocity (reduces window by 2)
        vel = data[:, 2:] - data[:, :-2]  # [R, W-2]
        pos = data[:, 1:-1]  # [R, W-2]
        W2 = vel.shape[1]
        if W2 < 2:
            return np.zeros((data.shape[0], data.shape[0]))

        vel_z = self._zscore_rows(vel)  # [R, W-2]
        pos_z = self._zscore_rows(pos)  # [R, W-2]

        # A[i,j] = normalised cross-covariance: corr(vel_i, pos_j)
        A = (vel_z @ pos_z.T) / max(W2 - 1, 1)  # [R, R]

        # Symmetric joint asymmetry
        FC = np.abs(A + A.T) / 2.0
        np.fill_diagonal(FC, 1.0)
        return np.clip(FC, 0.0, 1.0)

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        _, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            FCSs.append(self._trac_matrix(seg))
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
