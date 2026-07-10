"""
SparseDccExp_hybrid_ensemble.

Hybrid dFC estimator generated from selected threshold_70_filtered_methods.npy
methods.  The method combines at least two selected source-method elements and
returns a standard state-free PydFC DFC object.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class SPARSE_DCC_EXP_HYBRID_ENSEMBLE(BaseDFCMethod):
    """State-free hybrid of SparseCoactivationCodeFC, DCCConnectivity, ExponentialWindow."""

    MEASURE_NAME = "SparseDccExp_hybrid_ensemble"
    COMPONENTS = ['SparseCoactivationCodeFC', 'DCCConnectivity', 'ExponentialWindow']
    HYBRID_RULE = "ensemble_mean"

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "min_periods",
            "half_life",
            "window_length",
            "n_overlap",
            "nonlinear_gain",
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
        if self.params["min_periods"] is None:
            self.params["min_periods"] = 12
        if self.params["half_life"] is None:
            self.params["half_life"] = 30
        if self.params["window_length"] is None:
            self.params["window_length"] = 30
        if self.params["n_overlap"] is None:
            self.params["n_overlap"] = 0.0
        if self.params["nonlinear_gain"] is None:
            self.params["nonlinear_gain"] = 1.0

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def _alpha_from_half_life(self, Fs):
        half_life_samples = max(float(self.params["half_life"]) * Fs, 1.0)
        return 1.0 - np.exp(np.log(0.5) / half_life_samples)

    @staticmethod
    def _corr_from_cov(covariance):
        variance = np.diag(covariance)
        scale = np.sqrt(np.outer(variance, variance))
        corr = np.divide(
            covariance,
            scale,
            out=np.zeros_like(covariance, dtype=float),
            where=scale > 1e-12,
        )
        corr[np.isnan(corr)] = 0.0
        corr = (corr + corr.T) / 2.0
        np.fill_diagonal(corr, 1.0)
        return np.clip(corr, -1.0, 1.0)

    def _weighted_corr(self, samples, weights):
        weights = np.asarray(weights, dtype=float)
        weights = np.maximum(weights, 0.0)
        if np.sum(weights) <= 1e-12:
            weights = np.ones(samples.shape[1], dtype=float)
        weights = weights / np.sum(weights)
        mean = np.sum(samples * weights[None, :], axis=1, keepdims=True)
        centered = samples - mean
        covariance = (centered * weights[None, :]) @ centered.T
        return self._corr_from_cov(covariance)

    @staticmethod
    def _rank_rows(samples):
        ranks = np.zeros_like(samples, dtype=float)
        for i, row in enumerate(samples):
            order = np.argsort(row, kind="mergesort")
            rank = np.empty_like(order, dtype=float)
            rank[order] = np.arange(row.size, dtype=float)
            ranks[i] = rank
        return ranks

    def _component_matrix(self, name, data, Fs, tr, state):
        n_regions = data.shape[0]
        min_periods = int(self.params["min_periods"])
        start = max(0, tr - min_periods + 1)
        window = data[:, start : tr + 1]
        alpha = self._alpha_from_half_life(Fs)
        age = tr - np.arange(tr + 1)

        if name == "SlidingWindow":
            return self._weighted_corr(window, np.ones(window.shape[1]))
        if name == "ExponentialWindow":
            return self._weighted_corr(data[:, : tr + 1], alpha * np.power(1.0 - alpha, age))
        if name == "AdaptiveExponentialWindow":
            diff = np.mean(np.abs(np.diff(data[:, max(0, tr - min_periods) : tr + 1], axis=1))) if tr > 0 else 0.0
            adapt = diff / (diff + np.std(window) + 1e-8)
            local_alpha = 0.02 + 0.33 * adapt
            local_weights = local_alpha * np.power(1.0 - local_alpha, age)
            return self._weighted_corr(data[:, : tr + 1], local_weights)
        if name == "ChangepointResetWindow":
            if tr > min_periods:
                diffs = np.mean(np.abs(np.diff(data[:, max(0, tr - min_periods) : tr + 1], axis=1)), axis=0)
                threshold = np.median(diffs) + 2.0 * np.std(diffs)
                hits = np.where(diffs > threshold)[0]
                if hits.size:
                    window = data[:, start + hits[-1] + 1 : tr + 1]
            return self._weighted_corr(window, np.ones(window.shape[1]))
        if name == "DCCConnectivity":
            eps = data[:, : tr + 1] - np.mean(data[:, : tr + 1], axis=1, keepdims=True)
            lam = 0.94
            sigma2 = np.var(eps, axis=1) + 1e-8
            q = state.get("dcc_q", np.eye(n_regions))
            z = eps[:, -1] / np.sqrt(sigma2)
            q_bar = self._weighted_corr(eps, np.ones(eps.shape[1]))
            q = 0.10 * q_bar + 0.05 * np.outer(z, z) + 0.85 * q
            state["dcc_q"] = q
            return self._corr_from_cov(q + (1.0 - lam) * np.diag(sigma2))
        if name == "DifferentialCoactivationFC":
            if window.shape[1] < 2:
                return np.eye(n_regions)
            return self._weighted_corr(np.diff(window, axis=1), np.ones(window.shape[1] - 1))
        if name == "EdgeCoactivation":
            mean = np.mean(window, axis=1, keepdims=True)
            std = np.std(window, axis=1, keepdims=True) + 1e-8
            z = (data[:, tr : tr + 1] - mean)[:, 0] / std[:, 0]
            matrix = np.outer(z, z)
            scale = np.sqrt(np.outer(np.diag(matrix) ** 2, np.diag(matrix) ** 2)) + 1e-8
            return self._corr_from_cov(matrix / scale)
        if name == "KalmanCovariance":
            mean = state.get("kalman_mean", data[:, 0].copy())
            cov = state.get("kalman_cov", np.eye(n_regions))
            innovation = data[:, tr] - mean
            mean = mean + alpha * innovation
            cov = (1.0 - alpha) * cov + alpha * np.outer(innovation, innovation) + 1e-4 * np.eye(n_regions)
            state["kalman_mean"] = mean
            state["kalman_cov"] = cov
            return self._corr_from_cov(cov)
        if name == "MultiscaleWindow":
            mats = []
            for scale in (0.5, 1.0, 1.5):
                width = max(4, int(min_periods * scale))
                seg = data[:, max(0, tr - width + 1) : tr + 1]
                mats.append(self._weighted_corr(seg, np.ones(seg.shape[1])))
            return np.mean(mats, axis=0)
        if name == "QuantumMutualInformationFC":
            corr = self._weighted_corr(window, np.ones(window.shape[1]))
            qmi = -np.log(np.maximum(1.0 - corr**2, 1e-6))
            qmi = qmi / (np.max(qmi) + 1e-8)
            return np.sign(corr) * qmi
        if name == "RandomFourierDependence":
            omega = np.linspace(0.5, 2.5, 16)
            phase = np.linspace(0.0, np.pi, 16)
            phi = np.cos(window[:, :, None] * omega + phase)
            feature = np.mean(phi, axis=1)
            feature -= np.mean(feature, axis=1, keepdims=True)
            norm = np.linalg.norm(feature, axis=1, keepdims=True) + 1e-8
            feature = feature / norm
            matrix = feature @ feature.T
            np.fill_diagonal(matrix, 1.0)
            return np.clip(matrix, -1.0, 1.0)
        if name == "ReservoirEchoStateFC":
            reservoir = state.get("reservoir", np.zeros(n_regions))
            recurrent = state.get("reservoir_w", np.eye(n_regions) * 0.45)
            reservoir = np.tanh(0.55 * data[:, tr] + recurrent @ reservoir)
            state["reservoir"] = reservoir
            matrix = 0.5 * self._weighted_corr(window, np.ones(window.shape[1])) + 0.5 * np.outer(reservoir, reservoir)
            return self._corr_from_cov(matrix)
        if name == "RobustSlidingWindow":
            ranked = self._rank_rows(window)
            return self._weighted_corr(ranked, np.ones(ranked.shape[1]))
        if name == "SparseCoactivationCodeFC":
            edge = self._component_matrix("EdgeCoactivation", data, Fs, tr, state)
            threshold = np.quantile(np.abs(edge), 0.75)
            sparse = np.where(np.abs(edge) >= threshold, edge, 0.0)
            np.fill_diagonal(sparse, 1.0)
            return sparse
        if name == "STFTCoherence":
            centered = window - np.mean(window, axis=1, keepdims=True)
            spectrum = np.fft.rfft(centered, axis=1)
            if spectrum.shape[1] <= 1:
                return np.eye(n_regions)
            coeff = spectrum[:, 1:]
            cross = np.abs(coeff @ coeff.conj().T)
            power = np.sum(np.abs(coeff) ** 2, axis=1)
            denom = np.sqrt(np.outer(power, power)) + 1e-8
            matrix = np.real(cross / denom)
            np.fill_diagonal(matrix, 1.0)
            return np.clip(matrix, 0.0, 1.0)
        if name == "Time-Freq":
            centered = window - np.mean(window, axis=1, keepdims=True)
            spectrum = np.fft.rfft(centered, axis=1)
            if spectrum.shape[1] <= 1:
                return np.eye(n_regions)
            phase = spectrum[:, 1:] / (np.abs(spectrum[:, 1:]) + 1e-8)
            matrix = np.real(phase @ phase.conj().T) / phase.shape[1]
            np.fill_diagonal(matrix, 1.0)
            return np.clip(matrix, -1.0, 1.0)
        if name == "VolatilityWeightedFC":
            residuals = window - np.mean(window, axis=1, keepdims=True)
            weights = np.sum(residuals**2, axis=0)
            return self._weighted_corr(window, weights)
        raise ValueError(f"Unsupported hybrid component: {name}")

    def _combine(self, matrices):
        mats = np.array(matrices, dtype=float)
        rule = self.HYBRID_RULE
        gain = float(self.params["nonlinear_gain"])
        if rule == "ensemble_mean":
            combined = np.mean(mats, axis=0)
        else:
            base = mats[0]
            auxiliary = np.mean(mats[1:], axis=0)
            contrast = mats[1] - mats[-1]
            gate = 1.0 / (1.0 + np.exp(-gain * auxiliary))
            if "residual" in rule:
                combined = base + gate * contrast
            elif "sparse" in rule:
                threshold = np.quantile(np.abs(auxiliary), 0.65)
                combined = base * (np.abs(auxiliary) >= threshold) + 0.5 * contrast
            elif "kernel" in rule or "information" in rule or "spectral" in rule:
                combined = np.sign(base) * np.sqrt(np.abs(base * auxiliary) + 1e-8) + 0.25 * contrast
            elif "change" in rule or "derivative" in rule or "volatility" in rule:
                combined = (1.0 - gate) * base + gate * (base * auxiliary)
            else:
                combined = (1.0 - gate) * base + gate * auxiliary
        combined = (combined + combined.T) / 2.0
        combined[np.isnan(combined)] = 0.0
        np.fill_diagonal(combined, 1.0)
        return np.clip(combined, -1.0, 1.0)

    def dFC(self, time_series, Fs):
        min_periods = int(self.params["min_periods"])
        if time_series.shape[1] < min_periods:
            raise ValueError("time_series has fewer samples than min_periods.")
        state = {}
        FCSs = []
        TR_array = []
        for tr in range(min_periods - 1, time_series.shape[1]):
            matrices = [
                self._component_matrix(component, time_series, Fs, tr, state)
                for component in self.COMPONENTS
            ]
            FCSs.append(self._combine(matrices))
            TR_array.append(tr)
        return np.array(FCSs), np.array(TR_array)

    def estimate_FCS(self, time_series):
        return self

    def estimate_dFC(self, time_series):
        assert (
            len(time_series.subj_id_lst) == 1
        ), "this function takes only one subject as input."
        assert type(time_series) is TIME_SERIES, "time_series must be of TIME_SERIES class."
        time_series = self.manipulate_time_series4dFC(time_series)
        tic = time.time()
        FCSs, TR_array = self.dFC(time_series=time_series.data, Fs=time_series.Fs)
        self.set_dFC_assess_time(time.time() - tic)
        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.info_dict)
        return dFC
