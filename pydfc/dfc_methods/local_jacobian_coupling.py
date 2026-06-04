"""
Local Jacobian Coupling Connectivity (LJCC) — from dynamical systems theory.

Source domain: Takens (1981). Detecting strange attractors in turbulence.
Lecture Notes in Mathematics, 898, 366-381.
Kantz & Schreiber (2004). Nonlinear Time Series Analysis. Cambridge.

The Jacobian J(t) of a dynamical system at time t gives the instantaneous
linear sensitivity of each state variable to all others.  Reconstructing
the local Jacobian from the delay-embedded BOLD trajectory via sliding-window
least-squares regression yields a time-varying coupling matrix J_ij(t):
the instantaneous influence of region j's state on region i's next state.
Symmetrising |J_ij + J_ji|/2 gives an undirected measure of dynamical
coupling — connectivity as local phase-space sensitivity.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class LOCAL_JACOBIAN_COUPLING(BaseDFCMethod):
    """Windowed local Jacobian from delay-embedded BOLD trajectory.

    Each region's BOLD is delay-embedded into an m-dimensional phase space
    state.  Within each sliding window the local linear map

        X(t+1) ≈ J(t) X(t)

    is fit by least squares, where X(t) is the [R·m, W−1] matrix of
    delay-embedded states.  The instantaneous coupling between regions i and
    j is read from the top-left [R, R] block of J(t) (which captures how
    region j's current activity influences region i's next activity),
    and is symmetrised:

        FC[i,j] = (|J_ij| + |J_ji|) / 2,   normalised to [0, 1].

    Unlike linear correlation, the Jacobian captures the local dynamical
    geometry of the BOLD attractor, sensitive to transient coupling states.
    """

    MEASURE_NAME = "LocalJacobianCouplingFC"

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
            "embed_dim",
            "embed_lag",
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
        if self.params["embed_dim"] is None:
            self.params["embed_dim"] = 2
        if self.params["embed_lag"] is None:
            self.params["embed_lag"] = 1

    @property
    def measure_name(self):
        return self.params["measure_name"]

    @staticmethod
    def _delay_embed(x, m, lag):
        """Delay-embed [R, T] data into [R*m, T-(m-1)*lag] matrix."""
        R, T = x.shape
        offset = (m - 1) * lag
        T_emb = T - offset
        if T_emb <= 0:
            return x[:, :1]
        X = np.zeros((R * m, T_emb))
        for k in range(m):
            X[k * R : (k + 1) * R, :] = x[:, offset - k * lag : T - k * lag]
        return X

    def dFC(self, time_series, Fs):
        n_regions, T = time_series.shape
        m = int(self.params["embed_dim"])
        lag = int(self.params["embed_lag"])
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)

        X_emb = self._delay_embed(time_series, m, lag)  # [R*m, T_emb]
        T_emb = X_emb.shape[1]
        offset = (m - 1) * lag

        FCSs, TR_array = [], []
        for l in range(0, T_emb - W_samples + 1, step):
            seg = X_emb[:, l : l + W_samples]  # [R*m, W]
            X0 = seg[:, :-1]  # [R*m, W-1]
            X1 = seg[:, 1:]  # [R*m, W-1]

            # Least-squares local Jacobian: J = X1 @ pinv(X0)
            try:
                J = X1 @ np.linalg.pinv(X0, rcond=1e-6)  # [R*m, R*m]
            except np.linalg.LinAlgError:
                J = np.zeros((n_regions * m, n_regions * m))

            # Top-left [R, R] block: instantaneous coupling
            J_sub = J[:n_regions, :n_regions]

            # Symmetrised absolute coupling
            FC = (np.abs(J_sub) + np.abs(J_sub.T)) / 2.0

            # Normalise to [0, 1]
            upper = FC[np.triu_indices(n_regions, k=1)]
            fc_max = upper.max() if len(upper) > 0 else 1.0
            if fc_max > 1e-10:
                FC = FC / fc_max
            np.fill_diagonal(FC, 1.0)
            FC = np.clip(FC, 0.0, 1.0)

            FCSs.append(FC)
            TR_array.append(int(offset + l + W_samples // 2))

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
