"""
Phase-locking window dFC method.
"""

import time

import numpy as np
from scipy.signal import hilbert

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class PHASE_LOCKING_WINDOW(BaseDFCMethod):
    """Windowed phase-locking value from Hilbert analytic phases."""

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
        self.params["measure_name"] = "PhaseLockingWindow"
        self.params["is_state_based"] = False
        if self.params["W"] is None:
            self.params["W"] = 30

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _samples(self, value, Fs, minimum=1):
        return max(int(round(value * Fs)), minimum)

    def dFC(self, time_series, Fs):
        window = self._samples(self.params["W"], Fs, minimum=2)
        phase = np.angle(hilbert(time_series, axis=1))
        complex_phase = np.exp(1j * phase)
        FCSs = []
        TR_array = []
        for tr in range(window - 1, time_series.shape[1]):
            segment = complex_phase[:, tr - window + 1 : tr + 1]
            plv = np.abs(segment @ np.conjugate(segment.T)) / segment.shape[1]
            plv[np.diag_indices_from(plv)] = 1
            FCSs.append(plv)
            TR_array.append(tr)
        return np.array(FCSs), np.array(TR_array)

    def estimate_FCS(self, time_series):
        return self

    def estimate_dFC(self, time_series):
        assert (
            len(time_series.subj_id_lst) == 1
        ), "this function takes only one subject as input."
        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."
        time_series = self.manipulate_time_series4dFC(time_series)
        tic = time.time()
        FCSs, TR_array = self.dFC(time_series=time_series.data, Fs=time_series.Fs)
        self.set_dFC_assess_time(time.time() - tic)
        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.info_dict)
        return dFC
