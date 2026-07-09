"""
Multiplication of Temporal Derivatives (MTD) dFC method.

Reference: Shine et al. (2015). Estimation of dynamic functional connectivity
using Multiplication of Temporal Derivatives (MTD). NeuroImage, 122, 399-407.
doi:10.1016/j.neuroimage.2015.07.064
"""

import time

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class TEMPORAL_DERIVATIVE_MULTIPLICATION(BaseDFCMethod):
    """Instantaneous FC from the outer product of z-scored BOLD temporal
    derivatives, smoothed with a Gaussian kernel (Shine et al., 2015).

    For each pair of regions (i, j) the raw MTD signal at time t is
    dx_i(t) * dx_j(t), where dx_i(t) = x_i(t) - x_i(t-1).  After z-scoring
    each regional derivative series, the stack of outer products is convolved
    with a Gaussian kernel of width sigma_s seconds along the time axis and
    then normalized to a correlation scale.  The result is one FC matrix per TR
    with no explicit window length parameter.
    """

    MEASURE_NAME = "TemporalDerivativeMultiplication"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "sigma_s",
            "min_periods",
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

        # sigma_s: Gaussian smoothing kernel width in seconds
        if self.params["sigma_s"] is None:
            self.params["sigma_s"] = 3.0
        # min_periods: burn-in TRs to skip while the Gaussian kernel warms up
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _normalize_to_corr(self, cov_stack):
        """Normalize a [T, R, R] covariance stack to correlation scale."""
        R = cov_stack.shape[1]
        diag = cov_stack[:, np.arange(R), np.arange(R)]  # [T, R]
        scale = np.sqrt(np.einsum("ti,tj->tij", diag, diag))  # [T, R, R]
        scale = np.where(scale > 0, scale, 1.0)
        corr = cov_stack / scale
        corr = np.clip(corr, -1.0, 1.0)
        corr[:, np.arange(R), np.arange(R)] = 1.0
        corr[np.isnan(corr)] = 0.0
        return corr

    def dFC(self, time_series, Fs):
        sigma_s = float(self.params["sigma_s"])
        min_periods = int(self.params["min_periods"])
        # convert smoothing width from seconds to samples (floor at 0.5)
        sigma_samples = max(sigma_s * Fs, 0.5)

        # first-order temporal derivatives: shape [n_regions, T-1]
        dX = np.diff(time_series, axis=1).astype(float)

        # z-score each region's derivative series across time
        mu = dX.mean(axis=1, keepdims=True)
        sd = dX.std(axis=1, keepdims=True)
        dX = (dX - mu) / np.where(sd > 1e-10, sd, 1e-10)

        # instantaneous outer-product matrices: shape [T-1, n_regions, n_regions]
        C_raw = np.einsum("it,jt->tij", dX, dX)

        # Gaussian smoothing along the time axis
        C_smooth = gaussian_filter1d(C_raw, sigma=sigma_samples, axis=0)

        # normalize to correlation scale
        C_corr = self._normalize_to_corr(C_smooth)

        # skip early TRs where the Gaussian has not yet accumulated enough signal
        start = max(min_periods - 1, 0)
        FCSs = C_corr[start:]
        # derivative at dX[:, t] corresponds to TR index t+1 in the original signal
        TR_array = np.arange(start + 1, time_series.shape[1])

        return FCSs, TR_array

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
