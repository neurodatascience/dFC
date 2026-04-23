"""DFC method adapters for validation.

These wrappers do not reimplement any pydfc method. They only adapt the
existing pydfc classes to the synthetic validation data by constructing the
required ``TIME_SERIES`` objects and converting each returned ``DFC`` object to
a dense ``[n_subjects, n_timepoints, n_regions, n_regions]`` array.
"""

import sys
import types
from abc import ABC, abstractmethod
from collections import OrderedDict
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _ensure_pydfc_namespace():
    pydfc_path = str(_PACKAGE_ROOT / "pydfc")
    pydfc_methods_path = str(_PACKAGE_ROOT / "pydfc" / "dfc_methods")

    if "pydfc" not in sys.modules:
        pydfc_module = types.ModuleType("pydfc")
        pydfc_module.__path__ = [pydfc_path]
        sys.modules["pydfc"] = pydfc_module

    if "pydfc.dfc_methods" not in sys.modules:
        pydfc_methods_module = types.ModuleType("pydfc.dfc_methods")
        pydfc_methods_module.__path__ = [pydfc_methods_path]
        sys.modules["pydfc.dfc_methods"] = pydfc_methods_module


def _load_pydfc_class(module_name: str, class_name: str):
    _ensure_pydfc_namespace()
    module = import_module(module_name)
    return getattr(module, class_name)


def _normalize_method_name(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def _make_node_metadata(n_regions: int):
    locs = np.zeros((n_regions, 3), dtype=float)
    node_labels = [f"roi_{idx:03d}" for idx in range(n_regions)]
    return locs, node_labels


def _make_time_series(subject_data: np.ndarray, subj_id: str, fs: float = 1.0):
    from pydfc.time_series import TIME_SERIES

    n_timepoints, n_regions = subject_data.shape
    locs, node_labels = _make_node_metadata(n_regions)
    return TIME_SERIES(
        data=subject_data.T.copy(),
        subj_id=subj_id,
        Fs=fs,
        locs=locs,
        node_labels=node_labels,
        TS_name="synthetic_validation",
        session_name="validation",
    )


def _make_group_time_series(timeseries: np.ndarray, fs: float = 1.0):
    if timeseries.ndim != 3:
        raise ValueError(
            f"Expected timeseries with shape [n_subjects, n_timepoints, n_regions], got {timeseries.shape}"
        )

    group_ts = _make_time_series(timeseries[0], "sub_000", fs=fs)
    for subj_idx in range(1, timeseries.shape[0]):
        group_ts.append_ts(
            new_time_series=timeseries[subj_idx].T.copy(),
            subj_id=f"sub_{subj_idx:03d}",
        )
    return group_ts


def _dense_dfc_from_result(dfc_obj, n_timepoints: int) -> np.ndarray:
    matrices = dfc_obj.get_dFC_mat(TRs=dfc_obj.TR_array)
    tr_array = np.asarray(dfc_obj.TR_array, dtype=int)

    if matrices.ndim != 3:
        raise ValueError(
            f"Expected dFC matrices with shape [n_time, n_regions, n_regions], got {matrices.shape}"
        )

    n_regions = matrices.shape[1]
    dense = np.full((n_timepoints, n_regions, n_regions), np.nan, dtype=np.float32)

    for matrix, tr in zip(matrices, tr_array):
        if 0 <= tr < n_timepoints:
            dense[tr, :, :] = matrix

    return dense


class DFCMethodWrapper(ABC):
    """Base class for direct adapters around pydfc methods."""

    def __init__(self, name: str, **params):
        self.name = name
        self.params = params

    @abstractmethod
    def run(self, timeseries: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PydfcMethodWrapper(DFCMethodWrapper):
    """Generic adapter for an existing pydfc method class."""

    def __init__(
        self,
        name: str,
        method_factory: Callable[..., object],
        fit_on_dataset: bool = False,
        fs: float = 1.0,
        **params,
    ):
        super().__init__(name=name, **params)
        self.method_factory = method_factory
        self.fit_on_dataset = fit_on_dataset
        self.fs = fs

    def _new_method(self):
        return self.method_factory(**self.params)

    def run(self, timeseries: np.ndarray) -> np.ndarray:
        method = self._new_method()
        n_subjects, n_timepoints, _ = timeseries.shape
        outputs = []

        if self.fit_on_dataset:
            group_ts = _make_group_time_series(timeseries, fs=self.fs)
            if hasattr(method, "estimate_FCS"):
                method.estimate_FCS(time_series=group_ts)

        for subj_idx in range(n_subjects):
            subject_ts = _make_time_series(
                timeseries[subj_idx], f"sub_{subj_idx:03d}", fs=self.fs
            )
            if not hasattr(method, "estimate_dFC"):
                raise AttributeError(
                    f"{type(method).__name__} does not implement estimate_dFC"
                )
            dFC = method.estimate_dFC(time_series=subject_ts)
            outputs.append(_dense_dfc_from_result(dFC, n_timepoints=n_timepoints))

        return np.stack(outputs, axis=0)


class SlidingWindowWrapper(PydfcMethodWrapper):
    def __init__(self, W: int = 30, n_overlap: float = 0.5, **kwargs):
        SLIDING_WINDOW = _load_pydfc_class(
            "pydfc.dfc_methods.sliding_window", "SLIDING_WINDOW"
        )

        params = {
            "W": W,
            "n_overlap": n_overlap,
            "sw_method": kwargs.get("sw_method", "pear_corr"),
            "tapered_window": kwargs.get("tapered_window", True),
            "window_std": kwargs.get("window_std", None),
            "normalization": kwargs.get("normalization", True),
            "num_select_nodes": kwargs.get("num_select_nodes", None),
            "n_jobs_sw": kwargs.get("n_jobs_sw", 1),
            "backend_sw": kwargs.get("backend_sw", "threading"),
        }
        super().__init__(
            name=f"SlidingWindow_W{W}_overlap{n_overlap}_{params['sw_method']}",
            method_factory=SLIDING_WINDOW,
            fit_on_dataset=False,
            **params,
        )


class TimeFreqWrapper(PydfcMethodWrapper):
    def __init__(self, **kwargs):
        TIME_FREQ = _load_pydfc_class("pydfc.dfc_methods.time_freq", "TIME_FREQ")

        params = {
            "TF_method": kwargs.get("TF_method", "WTC"),
            "coi_correction": kwargs.get("coi_correction", True),
            "n_jobs_tf": kwargs.get("n_jobs_tf", 1),
            "verbose": kwargs.get("verbose", 0),
            "backend_tf": kwargs.get("backend_tf", "loky"),
            "normalization": kwargs.get("normalization", True),
            "num_select_nodes": kwargs.get("num_select_nodes", None),
        }
        super().__init__(
            name=f"TimeFreq_{params['TF_method']}",
            method_factory=TIME_FREQ,
            fit_on_dataset=False,
            **params,
        )


class ExperimentalStateFreeWrapper(PydfcMethodWrapper):
    """Adapter for experimental state-free dFC methods."""

    def __init__(
        self,
        module_name: str,
        class_name: str,
        display_name: str,
        half_life: float = 30,
        W: int = 30,
        min_periods: int = 10,
        **kwargs,
    ):
        method_class = _load_pydfc_class(f"pydfc.dfc_methods.{module_name}", class_name)

        params = {
            "half_life": half_life,
            "W": W,
            "min_periods": min_periods,
            "normalization": kwargs.get("normalization", True),
            "num_select_nodes": kwargs.get("num_select_nodes", None),
        }
        optional_params = [
            "alpha",
            "alpha_min",
            "alpha_max",
            "windows",
            "max_lag",
            "change_threshold",
            "shrinkage",
            "process_noise",
            "kernel_width",
            "n_random_features",
            "random_seed",
            "event_quantile",
            "event_decay",
            "tail_quantile",
            "learning_rate",
            "n_components",
            "diffusion_rate",
            "instantaneous_weight",
        ]
        for param_name in optional_params:
            if param_name in kwargs:
                params[param_name] = kwargs[param_name]

        super().__init__(
            name=display_name,
            method_factory=method_class,
            fit_on_dataset=False,
            **params,
        )


class ExponentialWindowWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="exponential_window",
            class_name="EXPONENTIAL_WINDOW",
            display_name="ExponentialWindow_halfLife30",
            **kwargs,
        )


class AdaptiveExponentialWindowWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="adaptive_exponential_window",
            class_name="ADAPTIVE_EXPONENTIAL_WINDOW",
            display_name="AdaptiveExponentialWindow",
            **kwargs,
        )


class MultiscaleWindowWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="multiscale_window",
            class_name="MULTISCALE_WINDOW",
            display_name="MultiscaleWindow",
            windows=kwargs.pop("windows", [15, 30, 60]),
            **kwargs,
        )


class EdgeCoactivationWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="edge_coactivation",
            class_name="EDGE_COACTIVATION",
            display_name="EdgeCoactivation",
            **kwargs,
        )


class PhaseLockingWindowWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="phase_locking_window",
            class_name="PHASE_LOCKING_WINDOW",
            display_name="PhaseLockingWindow",
            **kwargs,
        )


class DerivativeWeightedWindowWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="derivative_weighted_window",
            class_name="DERIVATIVE_WEIGHTED_WINDOW",
            display_name="DerivativeWeightedWindow",
            **kwargs,
        )


class ChangepointResetWindowWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="changepoint_reset_window",
            class_name="CHANGEPOINT_RESET_WINDOW",
            display_name="ChangepointResetWindow",
            **kwargs,
        )


class KalmanCovarianceWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="kalman_covariance",
            class_name="KALMAN_COVARIANCE",
            display_name="KalmanCovariance",
            **kwargs,
        )


class LaggedMaxCorrelationWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="lagged_max_correlation",
            class_name="LAGGED_MAX_CORRELATION",
            display_name="LaggedMaxCorrelation",
            max_lag=kwargs.pop("max_lag", 2),
            **kwargs,
        )


class PrecisionShrinkageWindowWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="precision_shrinkage_window",
            class_name="PRECISION_SHRINKAGE_WINDOW",
            display_name="PrecisionShrinkageWindow",
            **kwargs,
        )


class RecurrenceKernelDependenceWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="recurrence_kernel_dependence",
            class_name="RECURRENCE_KERNEL_DEPENDENCE",
            display_name="RecurrenceKernelDependence",
            **kwargs,
        )


class RandomFourierDependenceWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="random_fourier_dependence",
            class_name="RANDOM_FOURIER_DEPENDENCE",
            display_name="RandomFourierDependence",
            **kwargs,
        )


class EventSynchronizationWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="event_synchronization",
            class_name="EVENT_SYNCHRONIZATION",
            display_name="EventSynchronization",
            **kwargs,
        )


class CopulaTailDependenceWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="copula_tail_dependence",
            class_name="COPULA_TAIL_DEPENDENCE",
            display_name="CopulaTailDependence",
            **kwargs,
        )


class OjaSubspaceConnectivityWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="oja_subspace_connectivity",
            class_name="OJA_SUBSPACE_CONNECTIVITY",
            display_name="OjaSubspaceConnectivity",
            **kwargs,
        )


class GraphDiffusionCoactivationWrapper(ExperimentalStateFreeWrapper):
    def __init__(self, **kwargs):
        super().__init__(
            module_name="graph_diffusion_coactivation",
            class_name="GRAPH_DIFFUSION_COACTIVATION",
            display_name="GraphDiffusionCoactivation",
            **kwargs,
        )


class CAPWrapper(PydfcMethodWrapper):
    def __init__(self, **kwargs):
        CAP = _load_pydfc_class("pydfc.dfc_methods.cap", "CAP")

        params = {
            "n_states": kwargs.get("n_states", 5),
            "n_subj_clstrs": kwargs.get("n_subj_clstrs", 10),
            "normalization": kwargs.get("normalization", True),
            "num_select_nodes": kwargs.get("num_select_nodes", None),
        }
        super().__init__(
            name=f"CAP_nstates{params['n_states']}",
            method_factory=CAP,
            fit_on_dataset=True,
            **params,
        )


class ContinuousHMMWrapper(PydfcMethodWrapper):
    def __init__(self, **kwargs):
        HMM_CONT = _load_pydfc_class("pydfc.dfc_methods.continuous_hmm", "HMM_CONT")

        params = {
            "n_states": kwargs.get("n_states", 5),
            "hmm_iter": kwargs.get("hmm_iter", 3),
            "normalization": kwargs.get("normalization", True),
            "num_select_nodes": kwargs.get("num_select_nodes", None),
        }
        super().__init__(
            name=f"ContinuousHMM_nstates{params['n_states']}",
            method_factory=HMM_CONT,
            fit_on_dataset=True,
            **params,
        )


class DiscreteHMMWrapper(PydfcMethodWrapper):
    def __init__(self, **kwargs):
        HMM_DISC = _load_pydfc_class("pydfc.dfc_methods.discrete_hmm", "HMM_DISC")

        params = {
            "clstr_base_measure": kwargs.get("clstr_base_measure", "SlidingWindow"),
            "clstr_distance": kwargs.get("clstr_distance", "manhattan"),
            "sw_method": kwargs.get("sw_method", "pear_corr"),
            "dhmm_obs_state_ratio": kwargs.get("dhmm_obs_state_ratio", 2),
            "hmm_iter": kwargs.get("hmm_iter", 3),
            "n_states": kwargs.get("n_states", 5),
            "n_subj_clstrs": kwargs.get("n_subj_clstrs", 10),
            "W": kwargs.get("W", 30),
            "window_std": kwargs.get("window_std", None),
            "n_overlap": kwargs.get("n_overlap", 0.5),
            "tapered_window": kwargs.get("tapered_window", True),
            "normalization": kwargs.get("normalization", True),
            "n_jobs_swc": kwargs.get("n_jobs_swc", 1),
            "backend_swc": kwargs.get("backend_swc", "threading"),
            "n_jobs_sw": kwargs.get("n_jobs_sw", 1),
            "backend_sw": kwargs.get("backend_sw", "threading"),
            "n_jobs_tf": kwargs.get("n_jobs_tf", 1),
            "backend_tf": kwargs.get("backend_tf", "loky"),
            "num_select_nodes": kwargs.get("num_select_nodes", None),
        }
        super().__init__(
            name=f"DiscreteHMM_nstates{params['n_states']}",
            method_factory=HMM_DISC,
            fit_on_dataset=True,
            **params,
        )


class WindowlessWrapper(PydfcMethodWrapper):
    def __init__(self, **kwargs):
        WINDOWLESS = _load_pydfc_class("pydfc.dfc_methods.windowless", "WINDOWLESS")

        params = {
            "n_states": kwargs.get("n_states", 5),
            "normalization": kwargs.get("normalization", True),
            "num_select_nodes": kwargs.get("num_select_nodes", None),
        }
        super().__init__(
            name=f"Windowless_nstates{params['n_states']}",
            method_factory=WINDOWLESS,
            fit_on_dataset=True,
            **params,
        )


class SlidingWindowClustrWrapper(PydfcMethodWrapper):
    def __init__(self, **kwargs):
        SLIDING_WINDOW_CLUSTR = _load_pydfc_class(
            "pydfc.dfc_methods.sliding_window_clustr", "SLIDING_WINDOW_CLUSTR"
        )

        params = {
            "clstr_base_measure": kwargs.get("clstr_base_measure", "SlidingWindow"),
            "clstr_distance": kwargs.get("clstr_distance", "manhattan"),
            "sw_method": kwargs.get("sw_method", "pear_corr"),
            "n_states": kwargs.get("n_states", 5),
            "n_subj_clstrs": kwargs.get("n_subj_clstrs", 10),
            "W": kwargs.get("W", 30),
            "window_std": kwargs.get("window_std", None),
            "n_overlap": kwargs.get("n_overlap", 0.5),
            "tapered_window": kwargs.get("tapered_window", True),
            "normalization": kwargs.get("normalization", True),
            "n_jobs_swc": kwargs.get("n_jobs_swc", 1),
            "backend_swc": kwargs.get("backend_swc", "threading"),
            "n_jobs_sw": kwargs.get("n_jobs_sw", 1),
            "backend_sw": kwargs.get("backend_sw", "threading"),
            "n_jobs_tf": kwargs.get("n_jobs_tf", 1),
            "backend_tf": kwargs.get("backend_tf", "loky"),
            "num_select_nodes": kwargs.get("num_select_nodes", None),
        }
        super().__init__(
            name=f"SlidingWindowClustr_nstates{params['n_states']}",
            method_factory=SLIDING_WINDOW_CLUSTR,
            fit_on_dataset=True,
            **params,
        )


class DummyMethod(DFCMethodWrapper):
    """Small synthetic method for framework smoke tests."""

    def __init__(self, **kwargs):
        super().__init__(name="DummyMethod", **kwargs)

    def run(self, timeseries: np.ndarray) -> np.ndarray:
        n_subjects, n_timepoints, n_regions = timeseries.shape
        dfc_output = np.zeros(
            (n_subjects, n_timepoints, n_regions, n_regions), dtype=np.float32
        )

        for t in range(n_timepoints):
            t_factor = 0.5 + 0.5 * np.sin(2 * np.pi * t / max(n_timepoints, 1))
            for i in range(n_regions):
                for j in range(n_regions):
                    if i // 10 == j // 10:
                        dfc_output[:, t, i, j] = 0.7 * t_factor
                    else:
                        dfc_output[:, t, i, j] = 0.2 * t_factor

        return dfc_output


def _method_registry() -> "OrderedDict[str, Dict[str, object]]":
    return OrderedDict(
        {
            "SlidingWindow_W30": {
                "factory": lambda: SlidingWindowWrapper(W=30, n_overlap=0.5),
                "aliases": ["sw", "slidingwindow", "slidingwindowwrapper"],
            },
            "TimeFreq_WTC": {
                "factory": lambda: TimeFreqWrapper(TF_method="WTC"),
                "aliases": ["tf", "timefreq", "timefreqwrapper", "wtc"],
            },
            "ExponentialWindow_halfLife30": {
                "factory": lambda: ExponentialWindowWrapper(),
                "aliases": ["ew", "exponentialwindow", "exponentialwindowwrapper"],
            },
            "AdaptiveExponentialWindow": {
                "factory": lambda: AdaptiveExponentialWindowWrapper(),
                "aliases": ["aew", "adaptiveew", "adaptiveexponentialwindow"],
            },
            "MultiscaleWindow": {
                "factory": lambda: MultiscaleWindowWrapper(),
                "aliases": ["msw", "multiscale", "multiscalewindow"],
            },
            "EdgeCoactivation": {
                "factory": lambda: EdgeCoactivationWrapper(),
                "aliases": ["eca", "edgecoactivation", "edgecofluctuation"],
            },
            "PhaseLockingWindow": {
                "factory": lambda: PhaseLockingWindowWrapper(),
                "aliases": ["plv", "phase", "phaselocking", "phaselockingwindow"],
            },
            "DerivativeWeightedWindow": {
                "factory": lambda: DerivativeWeightedWindowWrapper(),
                "aliases": ["dww", "derivativeweighted", "derivativeweightedwindow"],
            },
            "ChangepointResetWindow": {
                "factory": lambda: ChangepointResetWindowWrapper(),
                "aliases": ["crw", "changepoint", "changepointresetwindow"],
            },
            "KalmanCovariance": {
                "factory": lambda: KalmanCovarianceWrapper(),
                "aliases": ["kalman", "kalman_covariance", "kcv"],
            },
            "LaggedMaxCorrelation": {
                "factory": lambda: LaggedMaxCorrelationWrapper(),
                "aliases": ["lmc", "laggedmax", "laggedmaxcorrelation"],
            },
            "PrecisionShrinkageWindow": {
                "factory": lambda: PrecisionShrinkageWindowWrapper(),
                "aliases": ["psw", "partial", "precisionshrinkagewindow"],
            },
            "RecurrenceKernelDependence": {
                "factory": lambda: RecurrenceKernelDependenceWrapper(
                    min_periods=20, kernel_width=1.5
                ),
                "aliases": ["rkd", "recurrence", "recurrencekernel"],
            },
            "RandomFourierDependence": {
                "factory": lambda: RandomFourierDependenceWrapper(
                    min_periods=20, half_life=25, n_random_features=32
                ),
                "aliases": ["rfd", "randomfourier", "nonlinearfeatures"],
            },
            "EventSynchronization": {
                "factory": lambda: EventSynchronizationWrapper(
                    min_periods=20, event_quantile=0.85, event_decay=0.97
                ),
                "aliases": ["event", "eventsync", "event_synchronization"],
            },
            "CopulaTailDependence": {
                "factory": lambda: CopulaTailDependenceWrapper(
                    min_periods=20, half_life=25, tail_quantile=0.8
                ),
                "aliases": ["ctd", "copulatail", "taildependence"],
            },
            "OjaSubspaceConnectivity": {
                "factory": lambda: OjaSubspaceConnectivityWrapper(
                    min_periods=20, half_life=25, n_components=10, learning_rate=0.03
                ),
                "aliases": ["oja", "ojasubspace", "subspaceconnectivity"],
            },
            "GraphDiffusionCoactivation": {
                "factory": lambda: GraphDiffusionCoactivationWrapper(
                    min_periods=20,
                    half_life=20,
                    diffusion_rate=0.2,
                    instantaneous_weight=0.15,
                ),
                "aliases": ["gdc", "graphdiffusion", "diffusioncoactivation"],
            },
            "CAP_nstates5": {
                "factory": lambda: CAPWrapper(n_states=5),
                "aliases": ["cap", "capwrapper"],
            },
            "ContinuousHMM_nstates5": {
                "factory": lambda: ContinuousHMMWrapper(n_states=5),
                "aliases": ["chmm", "continuoushmm", "continuoushmmwrapper"],
            },
            "DiscreteHMM_nstates5": {
                "factory": lambda: DiscreteHMMWrapper(n_states=5),
                "aliases": ["dhmm", "discretehmm", "discretehmmwrapper"],
            },
            "Windowless_nstates5": {
                "factory": lambda: WindowlessWrapper(n_states=5),
                "aliases": ["windowless", "windowlesswrapper"],
            },
            "SlidingWindowClustr_nstates5": {
                "factory": lambda: SlidingWindowClustrWrapper(n_states=5),
                "aliases": ["swc", "slidingwindowclustr", "slidingwindowclustrwrapper"],
            },
            "DummyMethod": {
                "factory": lambda: DummyMethod(),
                "aliases": ["dummy"],
            },
        }
    )


def list_registered_methods() -> List[str]:
    """List all registered validation method keys (available or unavailable)."""
    return list(_method_registry().keys())


def get_method_catalog() -> List[Dict[str, object]]:
    """Return registry entries with aliases, availability, and reasons."""
    registry = _method_registry()
    available, unavailable = get_method_availability()
    catalog = []

    for idx, (method_key, spec) in enumerate(registry.items(), start=1):
        is_available = method_key in available
        catalog.append(
            {
                "id": idx,
                "key": method_key,
                "aliases": list(spec["aliases"]),
                "available": is_available,
                "reason": "" if is_available else unavailable.get(method_key, "Unknown"),
            }
        )

    return catalog


def get_method_availability() -> Tuple[Dict[str, DFCMethodWrapper], Dict[str, str]]:
    """Build methods and return available methods plus unavailable reasons."""
    available = {}
    unavailable = {}

    for method_key, spec in _method_registry().items():
        factory = spec["factory"]
        try:
            available[method_key] = factory()
        except Exception as exc:
            unavailable[method_key] = f"{type(exc).__name__}: {exc}"

    return available, unavailable


def resolve_method_requests(
    requested_methods: List[str],
) -> Tuple[Dict[str, DFCMethodWrapper], List[str], Dict[str, str]]:
    """Resolve user-provided names to available methods with alias support."""
    registry = _method_registry()
    available, unavailable = get_method_availability()

    alias_to_key = {}
    for method_key, spec in registry.items():
        alias_to_key[_normalize_method_name(method_key)] = method_key
        for alias in spec["aliases"]:
            alias_to_key.setdefault(_normalize_method_name(alias), method_key)

    selected = {}
    missing = []
    unavailable_selected = {}

    for requested_name in requested_methods:
        canonical_key = alias_to_key.get(_normalize_method_name(requested_name))
        if canonical_key is None:
            missing.append(requested_name)
            continue

        if canonical_key in available:
            selected[canonical_key] = available[canonical_key]
        else:
            unavailable_selected[canonical_key] = unavailable.get(
                canonical_key, "Unavailable for unknown reason"
            )

    return selected, missing, unavailable_selected


def get_available_methods() -> Dict[str, DFCMethodWrapper]:
    """Return available wrappers around the registered pydfc methods."""
    available, _ = get_method_availability()
    return available
