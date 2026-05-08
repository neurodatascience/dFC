"""Gaussian mixture state-based dFC."""

import time

import numpy as np
from sklearn.mixture import GaussianMixture

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


def _corr(samples):
    if samples.shape[0] < 2:
        return np.eye(samples.shape[1], dtype=float)
    cov = np.cov(samples, rowvar=False)
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    den = np.outer(std, std)
    corr = np.divide(cov, den, out=np.zeros_like(cov), where=den > 0)
    corr[np.diag_indices_from(corr)] = 1.0
    return 0.5 * (corr + corr.T)


class GAUSSIAN_MIXTURE_STATES(BaseDFCMethod):
    """Elliptical state emissions learned with a Gaussian mixture."""

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "n_states",
            "random_state",
            "covariance_type",
            "reg_covar",
            "max_iter",
            "smoothing",
            "train_sample_limit",
            "normalization",
            "num_subj",
            "num_select_nodes",
            "num_time_point",
            "Fs_ratio",
            "noise_ratio",
            "num_realization",
            "session",
        ]
        self.params = {name: params.get(name, None) for name in self.params_name_lst}
        self.params["measure_name"] = "GaussianMixtureStates"
        self.params["is_state_based"] = True
        if self.params["n_states"] is None:
            self.params["n_states"] = 5
        if self.params["random_state"] is None:
            self.params["random_state"] = 42
        if self.params["covariance_type"] is None:
            self.params["covariance_type"] = "full"
        if self.params["reg_covar"] is None:
            self.params["reg_covar"] = 1e-6
        if self.params["max_iter"] is None:
            self.params["max_iter"] = 300
        if self.params["smoothing"] is None:
            self.params["smoothing"] = 1.0
        if self.params["train_sample_limit"] is None:
            self.params["train_sample_limit"] = 5000

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _chunks(self, time_series):
        return [
            time_series.get_subj_ts(subjs_id=subj).data.T.copy()
            for subj in time_series.subj_id_lst
        ]

    def _fit_matrix(self, chunks):
        each = max(1, int(self.params["train_sample_limit"]) // max(len(chunks), 1))
        sampled = []
        for chunk in chunks:
            if chunk.shape[0] <= each:
                sampled.append(chunk)
            else:
                idx = np.linspace(0, chunk.shape[0] - 1, each, dtype=int)
                sampled.append(chunk[idx, :])
        return np.concatenate(sampled, axis=0)

    def estimate_FCS(self, time_series):
        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."
        time_series = self.manipulate_time_series4FCS(time_series)
        tic = time.time()

        chunks = self._chunks(time_series)
        fit_matrix = self._fit_matrix(chunks)
        n_states = min(int(self.params["n_states"]), fit_matrix.shape[0])
        self.params["n_states"] = n_states

        self.gmm_ = GaussianMixture(
            n_components=n_states,
            covariance_type=self.params["covariance_type"],
            reg_covar=self.params["reg_covar"],
            max_iter=self.params["max_iter"],
            random_state=self.params["random_state"],
        ).fit(fit_matrix)

        labels_chunks = [self.gmm_.predict(chunk).astype(int) for chunk in chunks]
        self.Z = np.concatenate(labels_chunks, axis=0)
        self.FCS_ = np.zeros(
            (n_states, chunks[0].shape[1], chunks[0].shape[1]), dtype=float
        )
        for state in range(n_states):
            samples = [
                chunk[labels == state, :]
                for chunk, labels in zip(chunks, labels_chunks)
                if np.any(labels == state)
            ]
            self.FCS_[state, :, :] = (
                _corr(np.concatenate(samples, axis=0))
                if samples
                else np.eye(chunks[0].shape[1])
            )

        counts = np.full((n_states, n_states), float(self.params["smoothing"]))
        for labels in labels_chunks:
            for a, b in zip(labels[:-1], labels[1:]):
                counts[a, b] += 1.0
        self.TPM = counts / np.maximum(np.sum(counts, axis=1, keepdims=True), 1e-12)

        self.set_mean_activity(time_series)
        self.set_FCS_fit_time(time.time() - tic)
        return self

    def estimate_dFC(self, time_series):
        assert (
            type(time_series) is TIME_SERIES
        ), "time_series must be of TIME_SERIES class."
        assert (
            len(time_series.subj_id_lst) == 1
        ), "this function takes only one subject as input."
        time_series = self.manipulate_time_series4dFC(time_series)
        tic = time.time()

        features = time_series.data.T.copy()
        probs = self.gmm_.predict_proba(features)
        labels = np.argmax(probs, axis=1).astype(int)

        self.set_dFC_assess_time(time.time() - tic)
        dFC = DFC(measure=self)
        dFC.set_dFC(
            FCSs=self.FCS_,
            FCS_idx=labels,
            FCS_proba=probs,
            TS_info=time_series.info_dict,
            TR_array=np.arange(features.shape[0], dtype=int),
        )
        return dFC
