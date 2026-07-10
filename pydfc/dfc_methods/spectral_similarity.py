"""
Spectral Similarity FC (SpecSimFC) — novel dFC method.

Two brain regions are "spectrally similar" if their BOLD power spectra
have the same shape — they oscillate in the same frequency bands at the
same relative strengths.  This method quantifies pairwise dynamic
connectivity as the Bhattacharyya coefficient between normalised
within-window power spectra, independent of signal amplitude.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class SPECTRAL_SIMILARITY(BaseDFCMethod):
    """Windowed Bhattacharyya coefficient between power spectral densities.

    Within each sliding window, the power spectral density (PSD) of each
    region is estimated via the squared FFT magnitude and normalised to
    a probability distribution over frequencies.  The pairwise similarity
    between regions i and j is:

        FC[i,j] = Σ_f √(PSD_i(f) · PSD_j(f))    (Bhattacharyya coefficient)

    This equals 1 when spectra are identical and 0 when they have no
    overlapping frequency content.  Unlike coherence, spectral similarity
    does not require phase alignment — two regions with identical spectral
    profiles but random phase relations will have FC = 1.  It therefore
    captures "oscillatory repertoire coupling" rather than phase-based
    synchrony.
    """

    MEASURE_NAME = "SpectralSimilarityFC"

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
    def _spectral_similarity_matrix(data):
        """[R, R] Bhattacharyya coefficient matrix for [R, W] data."""
        n_regions, W = data.shape
        # Power spectral density via FFT (positive frequencies only)
        fft_mag = np.abs(np.fft.rfft(data, axis=1)) ** 2  # [R, W//2+1]
        # Add tiny constant for numerical stability, then normalise
        fft_mag += 1e-10
        psd = fft_mag / fft_mag.sum(axis=1, keepdims=True)  # [R, F] probability

        # Bhattacharyya coefficient: BC[i,j] = sum_f sqrt(psd_i * psd_j)
        sqrt_psd = np.sqrt(psd)  # [R, F]
        C = sqrt_psd @ sqrt_psd.T  # [R, R]
        C = np.clip(C, 0.0, 1.0)
        np.fill_diagonal(C, 1.0)
        return C

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        _, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            FCSs.append(self._spectral_similarity_matrix(seg))
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
