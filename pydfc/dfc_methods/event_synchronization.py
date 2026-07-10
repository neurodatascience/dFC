"""
Event synchronization dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class EVENT_SYNCHRONIZATION(BaseDFCMethod):
    MEASURE_NAME = "EventSynchronization"
    """FC from co-occurring high-amplitude activity events."""

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "min_periods",
            "event_quantile",
            "event_decay",
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
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10
        if self.params["event_quantile"] is None:
            self.params["event_quantile"] = 0.85
        if self.params["event_decay"] is None:
            self.params["event_decay"] = 0.97

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def dFC(self, time_series, Fs):
        min_periods = int(self.params["min_periods"])
        thresholds = np.quantile(
            np.abs(time_series), self.params["event_quantile"], axis=1
        )
        event_rate = np.zeros(time_series.shape[0])
        coincidence = np.eye(time_series.shape[0])
        FCSs = []
        TR_array = []
        for tr in range(time_series.shape[1]):
            event = (np.abs(time_series[:, tr]) >= thresholds).astype(float)
            event_rate = self.params["event_decay"] * event_rate + event
            coincidence = self.params["event_decay"] * coincidence + np.outer(
                event, event
            )
            scale = np.sqrt(np.outer(event_rate, event_rate))
            matrix = np.divide(
                coincidence,
                scale,
                out=np.zeros_like(coincidence),
                where=scale > 0,
            )
            matrix[np.diag_indices_from(matrix)] = 1
            if tr >= min_periods - 1:
                FCSs.append(matrix)
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
