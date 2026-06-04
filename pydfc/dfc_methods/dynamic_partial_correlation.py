"""
Dynamic Partial Correlation (DyPC) dFC method.

Reference: Smith et al. (2011). Network modelling methods for fMRI. NeuroImage,
54(2), 875-891. doi:10.1016/j.neuroimage.2010.08.063

Also: Varoquaux & Craddock (2013). Learning and comparing functional connectomes
across subjects. NeuroImage, 80, 405-415. doi:10.1016/j.neuroimage.2013.04.007
"""

import time

import numpy as np
from sklearn.covariance import LedoitWolf

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class DYNAMIC_PARTIAL_CORRELATION(BaseDFCMethod):
    """Sliding-window partial correlation via Ledoit-Wolf regularised precision.

    Within each window, the regularised covariance (Ledoit-Wolf optimal
    shrinkage) is inverted to the precision matrix Θ.  Partial correlation is
    then obtained by symmetric normalisation:

        PartCorr_ij = −Θ_ij / √(Θ_ii · Θ_jj)

    Partial correlation controls for the linear influence of all other regions,
    yielding sparser, more direct coupling estimates than full Pearson
    correlation and avoiding inflated connectivity due to shared third-region
    drivers.
    """

    MEASURE_NAME = "DynamicPartialCorrelation"

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

    @staticmethod
    def _precision_to_partial_corr(precision):
        """Normalise precision matrix to partial correlation scale."""
        d = np.sqrt(np.abs(np.diag(precision)))
        denom = np.outer(d, d)
        denom[denom == 0] = 1.0
        pcorr = -precision / denom
        np.fill_diagonal(pcorr, 1.0)
        return np.clip(pcorr, -1.0, 1.0)

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        n_regions, T = time_series.shape

        lw = LedoitWolf(assume_centered=False)
        FCSs, TR_array = [], []

        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples].T  # [W, R]
            try:
                lw.fit(seg)
                C = self._precision_to_partial_corr(lw.precision_)
            except Exception:
                C = np.zeros((n_regions, n_regions))
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
