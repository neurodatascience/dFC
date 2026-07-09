"""Visualize outputs from the experimental state-free dFC methods."""

import sys
import types
import warnings
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
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


def _load_pydfc_class(module_name, class_name):
    _ensure_pydfc_namespace()
    module = import_module(module_name)
    return getattr(module, class_name)


_ensure_pydfc_namespace()
data_loader = import_module("pydfc.data_loader")

SLIDING_WINDOW = _load_pydfc_class("pydfc.dfc_methods.sliding_window", "SLIDING_WINDOW")
EXPONENTIAL_WINDOW = _load_pydfc_class(
    "pydfc.dfc_methods.exponential_window", "EXPONENTIAL_WINDOW"
)
ADAPTIVE_EXPONENTIAL_WINDOW = _load_pydfc_class(
    "pydfc.dfc_methods.adaptive_exponential_window", "ADAPTIVE_EXPONENTIAL_WINDOW"
)
MULTISCALE_WINDOW = _load_pydfc_class(
    "pydfc.dfc_methods.multiscale_window", "MULTISCALE_WINDOW"
)
EDGE_COACTIVATION = _load_pydfc_class(
    "pydfc.dfc_methods.edge_coactivation", "EDGE_COACTIVATION"
)
PHASE_LOCKING_WINDOW = _load_pydfc_class(
    "pydfc.dfc_methods.phase_locking_window", "PHASE_LOCKING_WINDOW"
)
DERIVATIVE_WEIGHTED_WINDOW = _load_pydfc_class(
    "pydfc.dfc_methods.derivative_weighted_window", "DERIVATIVE_WEIGHTED_WINDOW"
)
CHANGEPOINT_RESET_WINDOW = _load_pydfc_class(
    "pydfc.dfc_methods.changepoint_reset_window", "CHANGEPOINT_RESET_WINDOW"
)
KALMAN_COVARIANCE = _load_pydfc_class(
    "pydfc.dfc_methods.kalman_covariance", "KALMAN_COVARIANCE"
)
LAGGED_MAX_CORRELATION = _load_pydfc_class(
    "pydfc.dfc_methods.lagged_max_correlation", "LAGGED_MAX_CORRELATION"
)
PRECISION_SHRINKAGE_WINDOW = _load_pydfc_class(
    "pydfc.dfc_methods.precision_shrinkage_window", "PRECISION_SHRINKAGE_WINDOW"
)
RECURRENCE_KERNEL_DEPENDENCE = _load_pydfc_class(
    "pydfc.dfc_methods.recurrence_kernel_dependence", "RECURRENCE_KERNEL_DEPENDENCE"
)
RANDOM_FOURIER_DEPENDENCE = _load_pydfc_class(
    "pydfc.dfc_methods.random_fourier_dependence", "RANDOM_FOURIER_DEPENDENCE"
)
EVENT_SYNCHRONIZATION = _load_pydfc_class(
    "pydfc.dfc_methods.event_synchronization", "EVENT_SYNCHRONIZATION"
)
COPULA_TAIL_DEPENDENCE = _load_pydfc_class(
    "pydfc.dfc_methods.copula_tail_dependence", "COPULA_TAIL_DEPENDENCE"
)
OJA_SUBSPACE_CONNECTIVITY = _load_pydfc_class(
    "pydfc.dfc_methods.oja_subspace_connectivity", "OJA_SUBSPACE_CONNECTIVITY"
)
GRAPH_DIFFUSION_COACTIVATION = _load_pydfc_class(
    "pydfc.dfc_methods.graph_diffusion_coactivation", "GRAPH_DIFFUSION_COACTIVATION"
)


warnings.simplefilter("ignore")

OUTPUT_DIR = Path("validation_results") / "visualize_dfc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep this subset modest so every method is fast and the matrices remain readable.
NUM_SELECT_NODES = 50


def load_demo_bold():
    return data_loader.nifti2timeseries(
        nifti_file=(
            "examples/sample_data/sub-0001_task-restingstate_acq-mb3_"
            "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        ),
        n_rois=100,
        Fs=1 / 0.75,
        subj_id="sub-0001",
        confound_strategy="no_motion",  # no_motion, no_motion_no_gsr, or none
        standardize=False,
        TS_name=None,
        session=None,
    )


def representative_trs(tr_array, n_samples=8):
    """Pick a compact, evenly spaced subset of available dFC samples."""
    tr_array = np.asarray(tr_array, dtype=int)
    if len(tr_array) <= n_samples:
        return tr_array
    sample_idx = np.linspace(0, len(tr_array) - 1, n_samples, dtype=int)
    return tr_array[sample_idx]


def method_specs():
    base_params = {
        "normalization": True,
        "num_select_nodes": NUM_SELECT_NODES,
    }

    return [
        (
            "00_sliding_window",
            SLIDING_WINDOW,
            {
                **base_params,
                "W": 44,
                "n_overlap": 0.5,
                "sw_method": "pear_corr",
                "tapered_window": True,
                "window_std": None,
                "n_jobs_sw": 1,
                "backend_sw": "threading",
            },
        ),
        (
            "01_exponential_window",
            EXPONENTIAL_WINDOW,
            {
                **base_params,
                "half_life": 30,
                "min_periods": 12,
            },
        ),
        (
            "02_adaptive_exponential_window",
            ADAPTIVE_EXPONENTIAL_WINDOW,
            {
                **base_params,
                "min_periods": 12,
                "alpha_min": 0.02,
                "alpha_max": 0.35,
            },
        ),
        (
            "03_multiscale_window",
            MULTISCALE_WINDOW,
            {
                **base_params,
                "windows": [15, 30, 60],
            },
        ),
        (
            "04_edge_coactivation",
            EDGE_COACTIVATION,
            {
                **base_params,
                "half_life": 20,
                "min_periods": 12,
            },
        ),
        (
            "05_phase_locking_window",
            PHASE_LOCKING_WINDOW,
            {
                **base_params,
                "W": 44,
            },
        ),
        (
            "06_derivative_weighted_window",
            DERIVATIVE_WEIGHTED_WINDOW,
            {
                **base_params,
                "W": 44,
            },
        ),
        (
            "07_changepoint_reset_window",
            CHANGEPOINT_RESET_WINDOW,
            {
                **base_params,
                "half_life": 20,
                "min_periods": 12,
                "change_threshold": 4.0,
            },
        ),
        (
            "08_kalman_covariance",
            KALMAN_COVARIANCE,
            {
                **base_params,
                "half_life": 25,
                "min_periods": 12,
                "process_noise": 1e-4,
            },
        ),
        (
            "09_lagged_max_correlation",
            LAGGED_MAX_CORRELATION,
            {
                **base_params,
                "W": 44,
                "max_lag": 2,
            },
        ),
        (
            "10_precision_shrinkage_window",
            PRECISION_SHRINKAGE_WINDOW,
            {
                **base_params,
                "W": 60,
            },
        ),
        (
            "11_recurrence_kernel_dependence",
            RECURRENCE_KERNEL_DEPENDENCE,
            {
                **base_params,
                "min_periods": 20,
                "kernel_width": 1.5,
            },
        ),
        (
            "12_random_fourier_dependence",
            RANDOM_FOURIER_DEPENDENCE,
            {
                **base_params,
                "half_life": 25,
                "min_periods": 20,
                "n_random_features": 128,
                "random_seed": 42,
            },
        ),
        (
            "13_event_synchronization",
            EVENT_SYNCHRONIZATION,
            {
                **base_params,
                "min_periods": 20,
                "event_quantile": 0.85,
                "event_decay": 0.97,
            },
        ),
        (
            "14_copula_tail_dependence",
            COPULA_TAIL_DEPENDENCE,
            {
                **base_params,
                "half_life": 25,
                "min_periods": 20,
                "tail_quantile": 0.8,
            },
        ),
        (
            "15_oja_subspace_connectivity",
            OJA_SUBSPACE_CONNECTIVITY,
            {
                **base_params,
                "half_life": 25,
                "min_periods": 20,
                "n_components": 10,
                "learning_rate": 0.03,
                "random_seed": 42,
            },
        ),
        (
            "16_graph_diffusion_coactivation",
            GRAPH_DIFFUSION_COACTIVATION,
            {
                **base_params,
                "half_life": 20,
                "min_periods": 20,
                "diffusion_rate": 0.2,
                "instantaneous_weight": 0.15,
            },
        ),
    ]


def validate_method_params(method_name, method_cls, params):
    """Catch unsupported hyperparameters before running a method."""
    method = method_cls(**params)
    unsupported = sorted(set(params) - set(method.params_name_lst))
    if unsupported:
        raise ValueError(
            f"{method_name} received unsupported parameter(s): {unsupported}. "
            f"Supported parameters are: {method.params_name_lst}"
        )
    return method


def plot_connectivity_strength(dfc_obj, method_name):
    """Save a compact time-course summary for each dFC output."""
    mats = dfc_obj.get_dFC_mat(TRs=dfc_obj.TR_array)
    upper = np.triu_indices(mats.shape[1], k=1)
    mean_abs_connectivity = np.nanmean(np.abs(mats[:, upper[0], upper[1]]), axis=1)

    plt.figure(figsize=(10, 3))
    plt.plot(dfc_obj.TR_array, mean_abs_connectivity, linewidth=1.5)
    plt.xlabel("TR")
    plt.ylabel("Mean |connectivity|")
    plt.title(method_name)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{method_name}_mean_abs_connectivity.png", dpi=150)
    plt.close()


def main():
    BOLD = load_demo_bold()

    for method_name, method_cls, params in method_specs():
        print(f"Running {method_name}...")
        measure = validate_method_params(method_name, method_cls, params)
        dFC = measure.estimate_dFC(time_series=BOLD)
        TRs = representative_trs(dFC.TR_array, n_samples=8)

        dFC.visualize_dFC(
            TRs=TRs,
            normalize=False,
            fix_lim=False,
            save_image=True,
            output_root=str(OUTPUT_DIR / f"{method_name}_"),
        )
        plot_connectivity_strength(dFC, method_name)

    print(f"Saved visualization outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
