"""
Windowed Phase Lag Index (PLI) dFC method.

Reference: Stam et al. (2007). Phase lag index: Assessment of functional
connectivity from multi channel EEG and MEG with diminished bias from common
sources. Hum Brain Map, 28(11), 1178-1193. doi:10.1002/hbm.20346

Application to sliding-window fMRI dFC: Aydore et al. (2013). A note on the
phase locking value and its properties. NeuroImage, 74, 231-244.
doi:10.1016/j.neuroimage.2013.02.008
"""

import time

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class PHASE_LAG_INDEX_WINDOW(BaseDFCMethod):
    """Sliding-window Phase Lag Index (PLI) measuring consistent phase asymmetry.

    PLV (phase locking value) inflates connectivity estimates when two channels
    share a common zero-lag source (e.g. volume conduction in EEG, global
    signal in fMRI).  PLI instead measures whether the *sign* of the imaginary
    part of the cross-spectrum is consistently positive or negative:

        PLI_ij = |E_t[sign(sin(φ_j(t) − φ_i(t)))]|

    A value of 1 means the phase difference consistently falls on one side of
    zero (strict leading/lagging); 0 means no consistent asymmetry.  Pure
    zero-lag coupling contributes nothing to PLI, making it robust to common
    sources and global signal fluctuations.
    """

    MEASURE_NAME = "PhaseLagIndexWindow"

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
            "f_low",
            "f_high",
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
        if self.params["f_low"] is None:
            self.params["f_low"] = 0.01
        if self.params["f_high"] is None:
            self.params["f_high"] = 0.1

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        n_regions, T = time_series.shape

        f_low = float(self.params["f_low"])
        f_high = float(self.params["f_high"])
        nyq = Fs / 2.0

        if f_low > 0 and f_high < nyq:
            b, a = butter(4, [f_low / nyq, f_high / nyq], btype="band")
            filtered = filtfilt(b, a, time_series, axis=1)
        else:
            filtered = time_series.copy()

        phases = np.angle(hilbert(filtered, axis=1))  # [R, T]

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            phi = phases[:, l : l + W_samples]  # [R, W]

            # sin(φ_j(t) - φ_i(t)) for every pair via broadcasting
            # [R, W] - [R, 1, W] broadcasting → [R, R, W] of differences
            delta_phi = phi[np.newaxis, :, :] - phi[:, np.newaxis, :]  # [R, R, W]
            signed = np.sign(np.sin(delta_phi))  # [R, R, W]
            C = np.abs(signed.mean(axis=2))  # PLI: [R, R]
            np.fill_diagonal(C, 1.0)
            FCSs.append(C)
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
