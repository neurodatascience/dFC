"""
Point Process Connectivity (PPFC) dFC method.

Reference: Tagliazucchi et al. (2012). Criticality in Large-Scale Brain fMRI
Dynamics Unveiled by a Novel Point Process Analysis. Front Physiol, 3, 15.
doi:10.3389/fphys.2012.00015

Also: Liu & Duyn (2013). Time-varying functional network information extracted
from brief instances of spontaneous brain activity. PNAS, 110(11), 4392-4397.
doi:10.1073/pnas.1216856110
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class POINT_PROCESS_CONNECTIVITY(BaseDFCMethod):
    """Event-driven FC from discrete BOLD threshold-crossing frames.

    At each TR where at least one region's z-scored BOLD amplitude exceeds a
    positive threshold, an "event" is detected.  The instantaneous FC at that
    TR is the standardised co-activation matrix: z(t) zᵀ(t) with diagonal
    forced to 1, where z_i = x_i / σ_i (globally standardised).  TRs without
    events contribute no FC estimate, yielding a sparse output.  This approach
    preserves the high-amplitude, high-SNR moments of the BOLD signal and
    discards near-baseline TRs that contribute noise to time-averaged FC.
    """

    MEASURE_NAME = "PointProcessConnectivity"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "z_threshold",
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

        if self.params["z_threshold"] is None:
            self.params["z_threshold"] = 1.0

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def dFC(self, time_series, Fs):
        n_regions, T = time_series.shape
        threshold = float(self.params["z_threshold"])

        # Globally standardise each region (zero mean, unit variance over time)
        sigma = time_series.std(axis=1, keepdims=True)
        sigma = np.where(sigma > 1e-10, sigma, 1e-10)
        z = (time_series - time_series.mean(axis=1, keepdims=True)) / sigma

        # Detect events: TR where max |z| across regions >= threshold
        event_mask = np.max(np.abs(z), axis=0) >= threshold
        event_trs = np.where(event_mask)[0]

        if len(event_trs) == 0:
            # Fallback: return a single global mean FC centred at the midpoint
            C = np.corrcoef(time_series)
            C[np.isnan(C)] = 0.0
            np.fill_diagonal(C, 1.0)
            return C[np.newaxis], np.array([T // 2])

        FCSs, TR_array = [], []
        for t in event_trs:
            frame = z[:, t]
            C = np.outer(frame, frame)
            C = np.clip(C, -1.0, 1.0)
            np.fill_diagonal(C, 1.0)
            FCSs.append(C)
            TR_array.append(int(t))

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
