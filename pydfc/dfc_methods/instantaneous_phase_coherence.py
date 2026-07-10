"""
Instantaneous Phase Coherence (IPC) dFC method.

Reference: Glerean et al. (2012). Functional Magnetic Resonance Imaging Phase
Synchronization as a Measure of Dynamic Functional Connectivity. Brain Connectivity,
2(6), 353-365. doi:10.1089/brain.2012.0088

Also: Cabral et al. (2014). Exploring the network dynamics underlying brain activity
during rest. Progress in Neurobiology, 114, 102-122.
doi:10.1016/j.pneurobio.2013.12.005
"""

import time

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class INSTANTANEOUS_PHASE_COHERENCE(BaseDFCMethod):
    """Frame-by-frame FC from instantaneous BOLD phase differences (windowless).

    Each region's BOLD signal is bandpass-filtered to the low-frequency
    resting-state band (default 0.01–0.1 Hz) and Hilbert-transformed to
    extract the instantaneous phase φ_i(t).  The functional connectivity
    between regions i and j at time t is

        FC_ij(t) = cos(φ_i(t) − φ_j(t))

    which equals +1 for in-phase synchrony, −1 for anti-phase, and 0 for
    quadrature relationships.  No sliding window is required: every TR yields
    a full symmetric FC matrix, making this the highest temporal-resolution
    method in the suite.
    """

    MEASURE_NAME = "InstantaneousPhaseCoherence"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
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

        if self.params["f_low"] is None:
            self.params["f_low"] = 0.01
        if self.params["f_high"] is None:
            self.params["f_high"] = 0.1

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def dFC(self, time_series, Fs):
        n_regions, T = time_series.shape
        f_low = float(self.params["f_low"])
        f_high = float(self.params["f_high"])
        nyq = Fs / 2.0

        if f_low > 0 and f_high < nyq:
            b, a = butter(4, [f_low / nyq, f_high / nyq], btype="band")
            filtered = filtfilt(b, a, time_series, axis=1)
        else:
            filtered = time_series.copy()

        phases = np.angle(hilbert(filtered, axis=1))  # [n_regions, T]

        # FC_ij(t) = cos(φ_i(t) − φ_j(t))
        # cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        cos_phi = np.cos(phases)
        sin_phi = np.sin(phases)
        FCSs = np.einsum("it,jt->tij", cos_phi, cos_phi) + np.einsum(
            "it,jt->tij", sin_phi, sin_phi
        )  # [T, R, R]; diagonal is cos(0) = 1 by construction

        TR_array = np.arange(T)
        return FCSs, TR_array

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
