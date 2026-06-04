"""
Differential Coactivation FC (DiffCoact) — novel dFC method.

Instead of correlating BOLD amplitude levels, correlates first-order
temporal differences (dx_i(t) = x_i(t) - x_i(t-1)).  Two regions are
"differentially co-activated" when they change in the same direction at
the same instant — a measure of co-derivative rather than co-level.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class DIFFERENTIAL_COACTIVATION(BaseDFCMethod):
    """Windowed correlation of first temporal differences of BOLD signals.

    For each region i, compute dx_i(t) = x_i(t) - x_i(t-1).  Within each
    sliding window the Pearson correlation matrix of these difference signals
    is the dFC estimate.  The measure answers: do these regions consistently
    accelerate and decelerate together, independent of their absolute levels?
    """

    MEASURE_NAME = "DifferentialCoactivationFC"

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

    def dFC(self, time_series, Fs):
        # Compute first differences across the entire signal
        dx = np.diff(time_series, axis=1)  # [n_regions, T-1]
        n_regions, T_dx = dx.shape

        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)

        FCSs, TR_array = [], []
        for l in range(0, T_dx - W_samples + 1, step):
            seg = dx[:, l : l + W_samples]
            C = np.corrcoef(seg)
            C[np.isnan(C)] = 0.0
            np.fill_diagonal(C, 1.0)
            FCSs.append(np.clip(C, -1.0, 1.0))
            TR_array.append(int(l + 1 + W_samples // 2))  # +1 offset for diff

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
