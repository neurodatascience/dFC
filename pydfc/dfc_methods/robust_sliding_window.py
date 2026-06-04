"""
Robust Sliding Window (Spearman) dFC method.

Reference: Pernet et al. (2013). Robust correlation analyses: false positive
and power validation using a new open source Matlab toolbox. Front Psychol,
3, 606. doi:10.3389/fpsyg.2012.00606

Application to dFC: discussed as a robust alternative to Pearson in
Lindquist et al. (2014). Evaluating dynamic bivariate correlations in
resting-state fMRI. NeuroImage, 101, 531-546.
doi:10.1016/j.neuroimage.2014.06.043
"""

import time

import numpy as np
from scipy.stats import spearmanr

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class ROBUST_SLIDING_WINDOW(BaseDFCMethod):
    """Sliding-window Spearman rank correlation as a robust dFC estimator.

    Standard Pearson correlation is sensitive to outliers and non-Gaussian
    BOLD amplitude distributions.  Spearman rank correlation replaces raw
    values with their within-window ranks before computing correlation,
    providing resistance to heavy-tailed noise and single-TR artefacts
    without requiring explicit outlier removal.  The output shares the same
    windowed structure as the standard sliding window but with substantially
    improved robustness under real fMRI noise conditions.
    """

    MEASURE_NAME = "RobustSlidingWindow"

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
    def _spearman_matrix(data):
        """Spearman correlation matrix for [n_regions, W] data."""
        # Rank each row
        from scipy.stats import rankdata

        ranked = np.array([rankdata(row) for row in data])
        C = np.corrcoef(ranked)
        C[np.isnan(C)] = 0.0
        np.fill_diagonal(C, 1.0)
        return np.clip(C, -1.0, 1.0)

    def dFC(self, time_series, Fs):
        W_samples = int(self.params["W"] * Fs)
        n_overlap = float(self.params["n_overlap"])
        step = max(int((1.0 - n_overlap) * W_samples), 1)
        n_regions, T = time_series.shape

        FCSs, TR_array = [], []
        for l in range(0, T - W_samples + 1, step):
            seg = time_series[:, l : l + W_samples]
            FCSs.append(self._spearman_matrix(seg))
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
