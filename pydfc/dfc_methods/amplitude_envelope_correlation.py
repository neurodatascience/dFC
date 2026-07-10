"""
Amplitude Envelope Correlation (AEC) dFC method.

Reference: Brookes et al. (2011). Investigating the electrophysiological basis of
resting state networks using magnetoencephalography. PNAS, 108(40), 16783-16788.
doi:10.1073/pnas.1112685108

Extended to fMRI BOLD by Hipp et al. (2012). Low-frequency fluctuations in MEG
and correlates in fMRI. Nat Neurosci, 15, 1067-1070. doi:10.1038/nn.3101
"""

import time

import numpy as np
from scipy.signal import hilbert

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class AMPLITUDE_ENVELOPE_CORRELATION(BaseDFCMethod):
    """Sliding-window Pearson correlation of Hilbert amplitude envelopes.

    Each region's BOLD signal is transformed to its analytic signal via the
    Hilbert transform; the instantaneous amplitude (envelope) replaces the raw
    signal.  Standard Pearson correlation is then computed within each sliding
    window of the envelope time series, yielding one FC matrix per window
    centre.  This method specifically captures amplitude-modulation coupling,
    separating it from the phase-synchrony component measured by PLV methods.
    """

    MEASURE_NAME = "AmplitudeEnvelopeCorrelation"

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

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        n_regions, T = time_series.shape

        envelope = np.abs(hilbert(time_series, axis=1))

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = envelope[:, l : l + W_samples]
            C = np.corrcoef(seg)
            C[np.isnan(C)] = 0.0
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
