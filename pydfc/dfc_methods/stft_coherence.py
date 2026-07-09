"""
Short-Time Fourier Transform Coherence (STFT-Coh) dFC method.

Reference: Wacker & Witte (2011). Time-frequency techniques in biomedical signal
analysis. Methods Inf Med, 50(5), 435-444. doi:10.3414/ME10-01-0083

Application to fMRI: Chang & Glover (2010). Time-frequency dynamics of resting-
state brain connectivity measured with fMRI. NeuroImage, 50(1), 81-98.
doi:10.1016/j.neuroimage.2009.12.011
"""

import time

import numpy as np
from scipy.signal import csd, welch

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class STFT_COHERENCE(BaseDFCMethod):
    """Sliding-window spectral coherence via Welch cross-spectral estimation.

    Within each window the magnitude-squared coherence between every pair of
    regions is computed by Welch's method:

        Coh_ij(f) = |S_xy(f)|² / (S_xx(f) · S_yy(f))

    and then summed over the low-frequency resting-state band (default
    0.01–0.1 Hz).  Unlike correlation, coherence is frequency-specific and
    normalised, capturing oscillatory coupling strength independent of
    signal amplitude.  Unlike continuous-wavelet coherence, STFT coherence
    uses a uniform frequency resolution.
    """

    MEASURE_NAME = "STFTCoherence"

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
            self.params["W"] = 60
        if self.params["n_overlap"] is None:
            self.params["n_overlap"] = 0.0
        if self.params["f_low"] is None:
            self.params["f_low"] = 0.01
        if self.params["f_high"] is None:
            self.params["f_high"] = 0.1

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _band_coherence(self, xi, xj, Fs, f_low, f_high):
        """Summed magnitude-squared coherence over [f_low, f_high]."""
        nperseg = min(len(xi), max(8, len(xi) // 4))
        freqs, Pxx = welch(xi, fs=Fs, nperseg=nperseg)
        _, Pyy = welch(xj, fs=Fs, nperseg=nperseg)
        _, Pxy = csd(xi, xj, fs=Fs, nperseg=nperseg)
        band = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(band):
            return 0.0
        denom = Pxx[band] * Pyy[band]
        coh = np.where(denom > 0, np.abs(Pxy[band]) ** 2 / denom, 0.0)
        return float(coh.mean())

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        n_regions, T = time_series.shape
        f_low = float(self.params["f_low"])
        f_high = float(self.params["f_high"])

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            C = np.zeros((n_regions, n_regions))
            for i in range(n_regions):
                C[i, i] = 1.0
                for j in range(i + 1, n_regions):
                    c = self._band_coherence(seg[i], seg[j], Fs, f_low, f_high)
                    C[i, j] = c
                    C[j, i] = c
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
