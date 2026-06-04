"""
Reservoir Echo State Connectivity (RESC) — from reservoir computing.

Source domain: Jaeger (2001). The "echo state" approach to analysing and
training recurrent neural networks. GMD Technical Report 148.
Maass et al. (2002). Real-time computing without stable states. Neural
Computation, 14(11), 2531-2560.

A fixed random recurrent network (the reservoir) provides a high-dimensional
nonlinear expansion with memory of the input.  Driving the reservoir with
BOLD and reading out each region's contribution to the reservoir state
yields a nonlinearly filtered signal per region.  Windowed correlation of
these readouts captures temporal coupling that linear correlation misses,
because the reservoir acts as a universal nonlinear temporal filter.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class RESERVOIR_ECHO_STATE(BaseDFCMethod):
    """Windowed correlation of reservoir-filtered BOLD readouts.

    A fixed random recurrent network of N units is driven by the R-region
    BOLD signal:

        h(t) = tanh(W_res h(t−1) + W_in x(t))

    W_res (N×N) has spectral radius < 1 to satisfy the echo-state property;
    W_in (N×R) is a random input weight matrix.  The readout for region i is:

        y_i(t) = W_in[:, i] · h(t)   — projection of reservoir state onto
                                        region i's input subspace

    Within each sliding window the Pearson correlation of the readout
    signals [y_1(t), …, y_R(t)] is the dFC estimate.  Unlike linear
    correlation on raw BOLD, the reservoir nonlinearly expands the signal
    history before correlation, capturing higher-order temporal dependencies.

    The reservoir is fixed (random seed 42) for reproducibility.
    """

    MEASURE_NAME = "ReservoirEchoStateFC"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self._W_res = None
        self._W_in = None

        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "W",
            "n_overlap",
            "reservoir_dim",
            "spectral_radius",
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
        if self.params["reservoir_dim"] is None:
            self.params["reservoir_dim"] = 100
        if self.params["spectral_radius"] is None:
            self.params["spectral_radius"] = 0.9

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _build_reservoir(self, n_regions):
        N = int(self.params["reservoir_dim"])
        rho = float(self.params["spectral_radius"])
        rng = np.random.RandomState(42)
        W_res = rng.randn(N, N) * 0.1
        sr = np.max(np.abs(np.linalg.eigvals(W_res)))
        if sr > 1e-10:
            W_res = W_res * (rho / sr)
        W_in = rng.randn(N, n_regions) * 0.1
        self._W_res = W_res
        self._W_in = W_in

    def _reservoir_readout(self, time_series):
        """Drive reservoir with [R, T] BOLD, return [R, T] readout signals."""
        n_regions, T = time_series.shape
        if self._W_res is None or self._W_in.shape[1] != n_regions:
            self._build_reservoir(n_regions)

        N = self._W_res.shape[0]
        h = np.zeros(N)
        Y = np.zeros((n_regions, T))

        for t in range(T):
            h = np.tanh(self._W_res @ h + self._W_in @ time_series[:, t])
            # Readout for each region: projection of h onto region i's input weights
            Y[:, t] = self._W_in.T @ h  # [R,]  — W_in.T is [R, N]

        return Y  # [R, T]

    def dFC(self, time_series, Fs):
        Y = self._reservoir_readout(time_series)  # [R, T]
        _, T = Y.shape

        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = Y[:, l : l + W_samples]
            C = np.corrcoef(seg)
            C[np.isnan(C)] = 0.0
            np.fill_diagonal(C, 1.0)
            FCSs.append(np.clip(C, -1.0, 1.0))
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
