"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import time

import numpy as np
from joblib import Parallel, delayed

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod

################################# Time-Frequency #################################

"""
PyCWT:
Authors
Sebastian Krieger, Nabil Freij, Alexey Brazhe, Christopher Torrence,
Gilbert P. Compo and contributors.

Disclaimer
This module is based on routines provided by C. Torrence and G. P. Compo available
at http://paos.colorado.edu/research/wavelets/, on routines provided by A. Grinsted,
J. Moore and S. Jevrejeva available at
http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence, and on routines
provided by A. Brazhe available at http://cell.biophys.msu.ru/static/swan/.

This software is released under a BSD-style open source license. Please read
the license file for further information. This routine is provided as is
without any express or implied warranties whatsoever.

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.
    dj : float, optional
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    s0 : float, optional
        Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
        Number of scales less one. Scales range from s0 up to
        s0 * 2**(J * dj), which gives a total of (J + 1) scales.
        Default is J = (log2(N*dt/so))/dj.
    sig : bool
        set to compute significance, default is True
    significance_level (float, optional) :
        Significance level to use. Default is 0.95.
    normalize (boolean, optional) :
        If set to true, normalizes CWT by the standard deviation of
        the signals.

- if n_jobs is None => no parallelization

todo:

- consider COI and edge effect in averaging:
    => should we truncate the time points having at less than 20 freqs as done in Savva et al. ?

"""
import pycwt as wavelet


class TIME_FREQ(BaseDFCMethod):

    def __init__(self, coi_correction=True, **params):

        # assert TF_method in self.TF_methods_name_lst, \
        #     "Time-frequency method not recognized."

        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "TF_method",
            "coi_correction",
            "n_jobs",
            "verbose",
            "backend",
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
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None

        self.params["measure_name"] = "Time-Freq"
        self.params["is_state_based"] = False
        self.params["coi_correction"] = coi_correction

    @property
    def measure_name(self):
        return self.params["measure_name"]  # + '_' + self.params['TF_method']

    def coi_correct(self, X, coi, freqs):
        # correct the edge effect in matrix X = [freqs, time] using coi
        # if self.coi_correction=True

        if not self.params["coi_correction"]:
            return X
        periods = 1 / freqs
        periods = np.repeat(periods[:, None], X.shape[1], axis=1)
        coi = np.repeat(coi[None, :], X.shape[0], axis=0)
        X_corrected = np.multiply(X, (coi >= periods))
        return X_corrected

    def WT_dFC(self, Y1, Y2, Fs, J, s0, dj):
        if (
            self.params["TF_method"] == "CWT_mag"
            or self.params["TF_method"] == "CWT_phase_r"
            or self.params["TF_method"] == "CWT_phase_a"
        ):
            # Cross Wavelet Transform
            WT_xy, coi, freqs, _ = wavelet.xwt(
                Y1,
                Y2,
                dt=1 / Fs,
                dj=dj,
                s0=s0,
                J=J,
                significance_level=0.95,
                wavelet="morlet",
                normalize=True,
            )

            if self.params["TF_method"] == "CWT_mag":
                WT_xy_corrected = self.coi_correct(WT_xy, coi, freqs)
                wt = np.abs(np.mean(WT_xy_corrected, axis=0))

            if (
                self.params["TF_method"] == "CWT_phase_r"
                or self.params["TF_method"] == "CWT_phase_a"
            ):
                cosA = np.cos(np.angle(WT_xy))
                sinA = np.sin(np.angle(WT_xy))

                cosA_corrected = self.coi_correct(cosA, coi, freqs)
                sinA_corrected = self.coi_correct(sinA, coi, freqs)

                A = cosA_corrected + sinA_corrected * 1j

                if self.params["TF_method"] == "CWT_phase_r":
                    wt = np.abs(np.mean(A, axis=0))
                else:
                    wt = np.angle(np.mean(A, axis=0))

        if self.params["TF_method"] == "WTC":
            # Wavelet Transform Coherence
            WT_xy, _, coi, freqs, _ = wavelet.wct(
                Y1,
                Y2,
                dt=1 / Fs,
                dj=dj,
                s0=s0,
                J=J,
                sig=False,
                significance_level=0.95,
                wavelet="morlet",
                normalize=True,
            )
            WT_xy_corrected = self.coi_correct(WT_xy, coi, freqs)
            wt = np.abs(np.mean(WT_xy_corrected, axis=0))

        return wt

    def estimate_FCS(self, time_series):

        return self

    def estimate_dFC(self, time_series):
        """
        we assume calc is applied on subjects separately
        """
        assert (
            len(time_series.subj_id_lst) == 1
        ), "this function takes only one subject as input."

        # params
        J = 100  # -1
        s0 = 1  # -1
        dj = 1 / 12  # 1/12

        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        WT = np.zeros((time_series.n_time, time_series.n_regions, time_series.n_regions))

        for i in range(time_series.n_regions):
            if self.params["n_jobs"] is None:
                Q = list()
                for j in range(time_series.n_regions):
                    Q.append(
                        self.WT_dFC(
                            Y1=time_series.data[i, :],
                            Y2=time_series.data[j, :],
                            Fs=time_series.Fs,
                            J=J,
                            s0=s0,
                            dj=dj,
                        )
                    )
            else:
                Q = Parallel(
                    n_jobs=self.params["n_jobs"],
                    verbose=self.params["verbose"],
                    backend=self.params["backend"],
                )(
                    delayed(self.WT_dFC)(
                        Y1=time_series.data[i, :],
                        Y2=time_series.data[j, :],
                        Fs=time_series.Fs,
                        J=J,
                        s0=s0,
                        dj=dj,
                    )
                    for j in range(time_series.n_regions)
                )
            WT[:, i, :] = np.array(Q).T

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=WT, TS_info=time_series.info_dict)
        return dFC


################################################################################
