"""Markov-smoothed Gaussian mixture state-based dFC."""

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


def _viterbi(emission_logp, tpm, startprob):
    n_time, n_states = emission_logp.shape
    log_tpm = np.log(np.maximum(tpm, 1e-12))
    log_start = np.log(np.maximum(startprob, 1e-12))
    dp = np.zeros((n_time, n_states), dtype=float)
    bp = np.zeros((n_time, n_states), dtype=int)
    dp[0, :] = log_start + emission_logp[0, :]
    for t in range(1, n_time):
        scores = dp[t - 1, :, None] + log_tpm
        bp[t, :] = np.argmax(scores, axis=0)
        dp[t, :] = scores[bp[t, :], np.arange(n_states)] + emission_logp[t, :]
    z = np.zeros((n_time,), dtype=int)
    z[-1] = np.argmax(dp[-1, :])
    for t in range(n_time - 2, -1, -1):
        z[t] = bp[t + 1, z[t + 1]]
    return z


class MARKOV_SMOOTHED_GMM_STATES(BaseDFCMethod):
    """Gaussian mixture emissions refined by a Markov transition prior."""

    def __init__(self, **params):
        self._covariance_type = "full"
        self._reg_covar = 1e-6
        self._max_iter = 300
        self._train_sample_limit = 5000
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "n_states",
            "transition_smoothing",
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
        self.params["measure_name"] = "MarkovSmoothedGMMStates"
        self.params["is_state_based"] = True
        if self.params["n_states"] is None:
            self.params["n_states"] = 5
        if self.params["transition_smoothing"] is None:
            self.params["transition_smoothing"] = 1.0

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _chunks(self, time_series):
        return [
            time_series.get_subj_ts(subjs_id=subj).data.T.copy()
            for subj in time_series.subj_id_lst
        ]

    def _fit_matrix(self, chunks):
        each = max(1, int(self._train_sample_limit) // max(len(chunks), 1))
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
            covariance_type=self._covariance_type,
            reg_covar=self._reg_covar,
            max_iter=self._max_iter,
            random_state=None,
        ).fit(fit_matrix)

        base_labels = [self.gmm_.predict(chunk).astype(int) for chunk in chunks]
        trans = np.full((n_states, n_states), float(self.params["transition_smoothing"]))
        start = np.full((n_states,), float(self.params["transition_smoothing"]))
        for labels in base_labels:
            start[labels[0]] += 1.0
            for a, b in zip(labels[:-1], labels[1:]):
                trans[a, b] += 1.0
        self.TPM = trans / np.maximum(np.sum(trans, axis=1, keepdims=True), 1e-12)
        self.startprob_ = start / np.maximum(np.sum(start), 1e-12)

        labels_chunks = []
        for chunk in chunks:
            emission = np.log(np.maximum(self.gmm_.predict_proba(chunk), 1e-12))
            labels_chunks.append(_viterbi(emission, self.TPM, self.startprob_))

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
        emission = np.log(np.maximum(self.gmm_.predict_proba(features), 1e-12))
        labels = _viterbi(emission, self.TPM, self.startprob_)
        proba = np.zeros((features.shape[0], int(self.params["n_states"])), dtype=float)
        proba[np.arange(labels.shape[0]), labels] = 1.0
        self.set_dFC_assess_time(time.time() - tic)
        dFC = DFC(measure=self)
        dFC.set_dFC(
            FCSs=self.FCS_,
            FCS_idx=labels,
            FCS_proba=proba,
            TS_info=time_series.info_dict,
            TR_array=np.arange(features.shape[0], dtype=int),
        )
        return dFC
