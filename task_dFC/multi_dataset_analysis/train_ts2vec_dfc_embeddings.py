import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pydfc.ml_utils import (
    dFC_feature_extraction_subj_lvl,
    find_available_subjects,
    load_dFC,
    load_task_data,
)


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_json_arg(value: Optional[str]) -> Dict[str, Any]:
    if value is None or value.strip() == "":
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must be an object/dict.")
    return parsed


def normalize_optional_token(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value in {"None", "none", "null"}:
        return None
    return value


def choose_subjects(
    subjects: Sequence[str],
    max_subjects_per_scan: Optional[int],
    rng: np.random.Generator,
) -> List[str]:
    subjects = sorted(list(subjects))
    if max_subjects_per_scan is None or len(subjects) <= max_subjects_per_scan:
        return subjects
    idx = rng.choice(len(subjects), size=max_subjects_per_scan, replace=False)
    idx = np.sort(idx)
    return [subjects[i] for i in idx]


def load_multi_dataset_spec(
    multi_dataset_info_path: str, simul_or_real: str
) -> Tuple[Dict[str, Any], str, List[str], List[str]]:
    with open(multi_dataset_info_path, "r") as f:
        multi_dataset_info = json.load(f)

    if simul_or_real == "real":
        spec = multi_dataset_info["real_data"]
    elif simul_or_real == "simulated":
        spec = multi_dataset_info["simulated_data"]
    else:
        raise ValueError("--simul_or_real must be 'real' or 'simulated'")

    main_root = spec["main_root"]
    datasets = list(spec["DATASETS"])
    tasks_to_include = list(spec["TASKS_to_include"])
    return multi_dataset_info, main_root, datasets, tasks_to_include


def load_dataset_info(
    dataset_info_file: str,
) -> Tuple[List[Optional[str]], List[str], Dict[str, List[Optional[str]]]]:
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    sessions = dataset_info.get("SESSIONS", None)
    if sessions is None:
        sessions = [None]

    tasks = dataset_info["TASKS"]

    runs = dataset_info.get("RUNS", None)
    if runs is None:
        runs = {task: [None] for task in tasks}
    else:
        runs = {
            task: (runs[task] if runs[task] is not None else [None]) for task in tasks
        }

    return sessions, tasks, runs


def prepare_ts2vec_input(
    sequences: Sequence[np.ndarray],
    seq_len_mode: str,
    pad_value: float,
    target_seq_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(sequences) == 0:
        raise ValueError("No sequences provided.")

    lengths = np.array([seq.shape[0] for seq in sequences], dtype=np.int32)
    feature_dims = {seq.shape[1] for seq in sequences}
    if len(feature_dims) != 1:
        raise ValueError(f"Inconsistent feature dimensions found: {sorted(feature_dims)}")
    feat_dim = next(iter(feature_dims))

    if target_seq_len is None:
        if seq_len_mode == "truncate_min":
            target_seq_len = int(lengths.min())
        elif seq_len_mode == "pad_max":
            target_seq_len = int(lengths.max())
        else:
            raise ValueError(f"Unknown seq_len_mode: {seq_len_mode}")
    target_seq_len = int(target_seq_len)
    if target_seq_len <= 0:
        raise ValueError("target_seq_len must be positive.")

    X = np.full((len(sequences), target_seq_len, feat_dim), pad_value, dtype=np.float32)
    for i, seq in enumerate(sequences):
        seq = seq.astype(np.float32, copy=False)
        if seq.shape[0] >= target_seq_len:
            X[i] = seq[:target_seq_len, :]
        else:
            X[i, : seq.shape[0], :] = seq

    return X, lengths


def standardize_ts2vec_input(
    X: np.ndarray, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=(0, 1), keepdims=True)
    std = X.std(axis=(0, 1), keepdims=True)
    std = np.where(std < eps, 1.0, std)
    Xz = (X - mean) / std
    return Xz.astype(np.float32, copy=False), mean.squeeze(), std.squeeze()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load dFC feature sequences across multiple datasets and train a TS2Vec "
            "model to learn embeddings."
        )
    )
    parser.add_argument(
        "--multi_dataset_info",
        type=str,
        required=True,
        help="Path to task_dFC/run_scripts_slurm/multi_dataset_info.json",
    )
    parser.add_argument(
        "--simul_or_real",
        type=str,
        required=True,
        choices=["real", "simulated"],
        help="Which section of the multi-dataset config to use.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to <multi_dataset_info.output_root>/TS2Vec/<simul_or_real>.",
    )
    parser.add_argument(
        "--dFC_ids",
        type=int,
        nargs="+",
        required=True,
        help="One or more dFC method IDs to process. A separate TS2Vec model is trained per compatible group.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of dataset IDs to include.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of task labels to include (e.g., task-Axcpt).",
    )
    parser.add_argument(
        "--sessions",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of session labels to include.",
    )
    parser.add_argument(
        "--runs",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of run labels to include.",
    )
    parser.add_argument(
        "--dynamic_pred",
        type=str,
        default="no",
        choices=["no", "past", "past_and_future"],
        help="Feature stacking mode reused from pydfc.ml_utils.dFC_feature_extraction_subj_lvl.",
    )
    parser.add_argument(
        "--normalize_dFC",
        type=str2bool,
        default=True,
        help="Apply rank normalization to state-free dFC matrices before vectorization.",
    )
    parser.add_argument(
        "--FCS_proba_for_SB",
        type=str2bool,
        default=True,
        help="For state-based dFC, use FCS probabilities instead of vectorized dFC matrices.",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=10,
        help="Minimum sequence length (TRs) after feature extraction.",
    )
    parser.add_argument(
        "--max_subjects_per_scan",
        type=int,
        default=None,
        help="Randomly subsample subjects per (dataset, session, task, run, dFC_id).",
    )
    parser.add_argument(
        "--max_total_sequences",
        type=int,
        default=None,
        help="Optional global cap on number of sequences per TS2Vec training group.",
    )
    parser.add_argument(
        "--seq_len_mode",
        type=str,
        default="truncate_min",
        choices=["truncate_min", "pad_max"],
        help="How to make variable-length sequences compatible for TS2Vec input.",
    )
    parser.add_argument(
        "--target_seq_len",
        type=int,
        default=None,
        help="Override sequence length used for training input (truncate/pad to this length).",
    )
    parser.add_argument(
        "--pad_value",
        type=float,
        default=0.0,
        help="Padding value when --seq_len_mode=pad_max or --target_seq_len exceeds sequence length.",
    )
    parser.add_argument(
        "--standardize_features",
        type=str2bool,
        default=False,
        help="Z-score features globally across sequences and timepoints before TS2Vec training.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for subsampling."
    )

    # TS2Vec common args (kept optional and overridable via JSON)
    parser.add_argument(
        "--device", type=str, default=None, help="TS2Vec device (e.g., cpu, cuda)."
    )
    parser.add_argument(
        "--output_dims", type=int, default=320, help="TS2Vec output embedding dimension."
    )
    parser.add_argument(
        "--hidden_dims", type=int, default=64, help="TS2Vec hidden dimension."
    )
    parser.add_argument("--depth", type=int, default=10, help="TS2Vec encoder depth.")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="TS2Vec fit batch size."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="TS2Vec fit learning rate (if supported)."
    )
    parser.add_argument(
        "--max_train_length",
        type=int,
        default=None,
        help="TS2Vec max_train_length (if supported).",
    )
    parser.add_argument(
        "--temporal_unit",
        type=int,
        default=0,
        help="TS2Vec temporal_unit (if supported).",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=50, help="Number of TS2Vec training epochs."
    )

    parser.add_argument(
        "--ts2vec_init_json",
        type=str,
        default=None,
        help="Extra JSON object of kwargs for TS2Vec(...) init. Overrides common args on key conflict.",
    )
    parser.add_argument(
        "--ts2vec_fit_json",
        type=str,
        default=None,
        help="Extra JSON object of kwargs for model.fit(...). Overrides common args on key conflict.",
    )
    parser.add_argument(
        "--ts2vec_encode_json",
        type=str,
        default=None,
        help="Extra JSON object of kwargs for model.encode(...).",
    )

    parser.add_argument(
        "--encoding_window",
        type=str,
        default="full_series",
        help="TS2Vec encode encoding_window. Use integer string for numeric window or full_series.",
    )
    parser.add_argument(
        "--save_timestep_embeddings",
        type=str2bool,
        default=False,
        help="Also save per-timestep embeddings (can be large).",
    )
    parser.add_argument(
        "--save_model",
        type=str2bool,
        default=True,
        help="Try to save the TS2Vec model if the package exposes model.save(...).",
    )

    return parser


def instantiate_ts2vec(
    TS2Vec: Any, init_kwargs: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Try a few progressively smaller init signatures to tolerate TS2Vec package variants.
    Returns (model, effective_init_kwargs).
    """
    candidate_kwargs = [dict(init_kwargs)]
    optional_drop_order = ["temporal_unit", "max_train_length", "device"]
    current = dict(init_kwargs)
    for key in optional_drop_order:
        if key in current:
            current = dict(current)
            current.pop(key, None)
            candidate_kwargs.append(current)

    last_error = None
    for kwargs in candidate_kwargs:
        try:
            return TS2Vec(**kwargs), kwargs
        except TypeError as e:
            last_error = e
            continue

    raise TypeError(
        f"Could not instantiate TS2Vec with tested kwargs variants: {last_error}"
    )


def fit_ts2vec_adaptive(
    model: Any, X_ts2vec: np.ndarray, fit_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Call TS2Vec.fit using only kwargs supported by the installed implementation.
    Returns the effective kwargs that were actually used.
    """
    try:
        sig = inspect.signature(model.fit)
        params = sig.parameters
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_var_kw:
            _ = model.fit(X_ts2vec, **fit_kwargs)
            return dict(fit_kwargs)

        allowed = {
            name
            for name, p in params.items()
            if name != "self"
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        effective_fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in allowed}
        _ = model.fit(X_ts2vec, **effective_fit_kwargs)
        return effective_fit_kwargs
    except (TypeError, ValueError):
        # Some wrapped methods don't expose a reliable signature.
        last_error = None
        candidate_kwarg_sets = [
            dict(fit_kwargs),
            {k: v for k, v in fit_kwargs.items() if k != "batch_size"},
            {k: v for k, v in fit_kwargs.items() if k not in {"batch_size", "lr"}},
            {k: v for k, v in fit_kwargs.items() if k == "n_epochs"},
            {},
        ]

        seen = set()
        for kw in candidate_kwarg_sets:
            key = tuple(sorted((str(k), str(v)) for k, v in kw.items()))
            if key in seen:
                continue
            seen.add(key)
            try:
                _ = model.fit(X_ts2vec, **kw)
                return kw
            except TypeError as e:
                last_error = e
                continue

        raise TypeError(
            f"Could not call TS2Vec.fit with tested kwargs variants: {last_error}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    multi_dataset_info, main_root, datasets, tasks_to_include = load_multi_dataset_spec(
        args.multi_dataset_info, args.simul_or_real
    )

    if args.datasets:
        datasets = [d for d in datasets if d in set(args.datasets)]
    task_filter = set(args.tasks) if args.tasks else set(tasks_to_include)
    session_filter = (
        {normalize_optional_token(x) for x in args.sessions} if args.sessions else None
    )
    run_filter = {normalize_optional_token(x) for x in args.runs} if args.runs else None

    if args.output_dir is None:
        output_root = f"{multi_dataset_info['output_root']}/TS2Vec/{args.simul_or_real}"
    else:
        output_root = args.output_dir
    Path(output_root).mkdir(parents=True, exist_ok=True)

    # group key -> payload
    grouped_sequences: Dict[Tuple[int, str, int], List[np.ndarray]] = {}
    grouped_targets: Dict[Tuple[int, str, int], List[np.ndarray]] = {}
    grouped_meta: Dict[Tuple[int, str, int], List[Dict[str, Any]]] = {}
    skipped_records: List[Dict[str, Any]] = []

    total_loaded = 0
    print(f"Datasets to process: {datasets}")
    for dataset in datasets:
        dataset_info_file = f"{main_root}/{dataset}/codes/dataset_info.json"
        roi_root = f"{main_root}/{dataset}/derivatives/ROI_timeseries"
        dFC_root = f"{main_root}/{dataset}/derivatives/dFC_assessed"

        if not os.path.exists(dataset_info_file):
            print(
                f"Skipping dataset {dataset}: dataset_info.json not found at {dataset_info_file}"
            )
            continue

        sessions, tasks, runs_map = load_dataset_info(dataset_info_file)
        if session_filter is not None:
            sessions = [s for s in sessions if s in session_filter]

        for session in sessions:
            for task in tasks:
                if task not in task_filter:
                    continue
                runs = runs_map.get(task, [None])
                if run_filter is not None:
                    runs = [r for r in runs if r in run_filter]

                for run in runs:
                    for dFC_id in args.dFC_ids:
                        try:
                            subjects = find_available_subjects(
                                dFC_root=dFC_root,
                                task=task,
                                run=run,
                                session=session,
                                dFC_id=dFC_id,
                            )
                        except FileNotFoundError:
                            print(f"Skipping missing dFC directory: {dFC_root}")
                            continue

                        if len(subjects) == 0:
                            continue
                        subjects = choose_subjects(
                            subjects=subjects,
                            max_subjects_per_scan=args.max_subjects_per_scan,
                            rng=rng,
                        )

                        print(
                            "Loading "
                            f"dataset={dataset} session={session} task={task} run={run} "
                            f"dFC_id={dFC_id} n_subjects={len(subjects)}"
                        )
                        for subj in subjects:
                            try:
                                dFC = load_dFC(
                                    dFC_root=dFC_root,
                                    subj=subj,
                                    task=task,
                                    dFC_id=dFC_id,
                                    run=run,
                                    session=session,
                                )
                                task_data = load_task_data(
                                    roi_root=roi_root,
                                    subj=subj,
                                    task=task,
                                    run=run,
                                    session=session,
                                )
                                X_subj, y_subj = dFC_feature_extraction_subj_lvl(
                                    dFC=dFC,
                                    task_data=task_data,
                                    dynamic_pred=args.dynamic_pred,
                                    normalize_dFC=args.normalize_dFC,
                                    FCS_proba_for_SB=args.FCS_proba_for_SB,
                                )
                            except Exception as e:
                                skipped_records.append(
                                    {
                                        "dataset": dataset,
                                        "session": session,
                                        "task": task,
                                        "run": run,
                                        "dFC_id": dFC_id,
                                        "subject": subj,
                                        "reason": f"{type(e).__name__}: {e}",
                                    }
                                )
                                continue

                            if X_subj.shape[0] < args.min_seq_len:
                                skipped_records.append(
                                    {
                                        "dataset": dataset,
                                        "session": session,
                                        "task": task,
                                        "run": run,
                                        "dFC_id": dFC_id,
                                        "subject": subj,
                                        "reason": f"seq_too_short({X_subj.shape[0]}<{args.min_seq_len})",
                                    }
                                )
                                continue

                            measure_name = dFC.measure.measure_name
                            group_key = (dFC_id, measure_name, int(X_subj.shape[1]))
                            grouped_sequences.setdefault(group_key, []).append(X_subj)
                            grouped_targets.setdefault(group_key, []).append(y_subj)
                            grouped_meta.setdefault(group_key, []).append(
                                {
                                    "dataset": dataset,
                                    "session": session,
                                    "task": task,
                                    "run": run,
                                    "dFC_id": dFC_id,
                                    "subject": subj,
                                    "measure_name": measure_name,
                                    "seq_len_raw": int(X_subj.shape[0]),
                                    "feature_dim": int(X_subj.shape[1]),
                                    "task_presence_mean": float(np.mean(y_subj)),
                                }
                            )
                            total_loaded += 1

    print(f"Loaded sequences: {total_loaded}")
    if total_loaded == 0:
        raise RuntimeError("No sequences were loaded. Check filters/paths/dFC_ids.")

    # Lazy import to avoid making this script unusable when ts2vec is not installed.
    try:
        from ts2vec import TS2Vec  # type: ignore
    except ImportError as e:
        raise ImportError(
            "TS2Vec package is not installed. Install a compatible implementation "
            "(commonly `pip install ts2vec`) and rerun."
        ) from e

    ts2vec_init_extra = parse_json_arg(args.ts2vec_init_json)
    ts2vec_fit_extra = parse_json_arg(args.ts2vec_fit_json)
    ts2vec_encode_extra = parse_json_arg(args.ts2vec_encode_json)

    encoding_window: Any = args.encoding_window
    if isinstance(encoding_window, str) and encoding_window.isdigit():
        encoding_window = int(encoding_window)

    run_summaries: List[Dict[str, Any]] = []

    for group_key in sorted(grouped_sequences.keys(), key=lambda x: (x[0], x[1], x[2])):
        dFC_id, measure_name, feature_dim = group_key
        sequences = grouped_sequences[group_key]
        targets = grouped_targets[group_key]
        meta_rows = grouped_meta[group_key]

        if (
            args.max_total_sequences is not None
            and len(sequences) > args.max_total_sequences
        ):
            idx = rng.choice(len(sequences), size=args.max_total_sequences, replace=False)
            idx = np.sort(idx)
            sequences = [sequences[i] for i in idx]
            targets = [targets[i] for i in idx]
            meta_rows = [meta_rows[i] for i in idx]

        if len(sequences) < 2:
            print(
                f"Skipping group dFC_id={dFC_id}, measure={measure_name}, feat_dim={feature_dim}: "
                "need at least 2 sequences for training."
            )
            continue

        X_ts2vec, raw_lengths = prepare_ts2vec_input(
            sequences=sequences,
            seq_len_mode=args.seq_len_mode,
            pad_value=args.pad_value,
            target_seq_len=args.target_seq_len,
        )

        feature_mean = None
        feature_std = None
        if args.standardize_features:
            X_ts2vec, feature_mean, feature_std = standardize_ts2vec_input(X_ts2vec)

        print(
            f"Training TS2Vec on group dFC_id={dFC_id}, measure={measure_name}, "
            f"X.shape={X_ts2vec.shape}"
        )

        init_kwargs: Dict[str, Any] = {
            "input_dims": int(feature_dim),
            "output_dims": int(args.output_dims),
            "hidden_dims": int(args.hidden_dims),
            "depth": int(args.depth),
        }
        if args.device is not None:
            init_kwargs["device"] = args.device
        if args.max_train_length is not None:
            init_kwargs["max_train_length"] = int(args.max_train_length)
        if args.temporal_unit is not None:
            init_kwargs["temporal_unit"] = int(args.temporal_unit)
        init_kwargs.update(ts2vec_init_extra)

        fit_kwargs: Dict[str, Any] = {
            "n_epochs": int(args.n_epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
        }
        fit_kwargs.update(ts2vec_fit_extra)

        model, effective_init_kwargs = instantiate_ts2vec(TS2Vec, init_kwargs)
        fit_kwargs = fit_ts2vec_adaptive(model, X_ts2vec, fit_kwargs)

        encode_kwargs: Dict[str, Any] = {"encoding_window": encoding_window}
        encode_kwargs.update(ts2vec_encode_extra)
        full_series_embeddings = model.encode(X_ts2vec, **encode_kwargs)

        timestep_embeddings = None
        if args.save_timestep_embeddings:
            timestep_embeddings = model.encode(X_ts2vec)

        safe_measure = str(measure_name).replace(" ", "_")
        group_dir = Path(output_root) / f"dFC_{dFC_id}_{safe_measure}_feat{feature_dim}"
        group_dir.mkdir(parents=True, exist_ok=True)

        meta_df = pd.DataFrame(meta_rows).copy()
        meta_df["seq_len_used"] = int(X_ts2vec.shape[1])
        meta_df["seq_len_raw"] = raw_lengths
        meta_df.to_csv(group_dir / "sequence_metadata.csv", index=False)

        np.save(
            group_dir / "full_series_embeddings.npy", np.asarray(full_series_embeddings)
        )
        np.save(group_dir / "train_sequences_input.npy", X_ts2vec)
        np.save(
            group_dir / "task_presence_labels.npy",
            np.array(targets, dtype=object),
            allow_pickle=True,
        )
        if timestep_embeddings is not None:
            np.save(
                group_dir / "timestep_embeddings.npy", np.asarray(timestep_embeddings)
            )
        if feature_mean is not None and feature_std is not None:
            np.save(group_dir / "feature_mean.npy", np.asarray(feature_mean))
            np.save(group_dir / "feature_std.npy", np.asarray(feature_std))

        model_saved = False
        if args.save_model and hasattr(model, "save"):
            try:
                model.save(str(group_dir / "ts2vec_model"))
                model_saved = True
            except Exception as e:
                print(f"Could not save TS2Vec model for group {group_key}: {e}")

        config_to_save = {
            "group_key": {
                "dFC_id": int(dFC_id),
                "measure_name": str(measure_name),
                "feature_dim": int(feature_dim),
            },
            "data": {
                "n_sequences": int(len(sequences)),
                "seq_len_mode": args.seq_len_mode,
                "target_seq_len": int(X_ts2vec.shape[1]),
                "raw_seq_len_min": int(raw_lengths.min()),
                "raw_seq_len_max": int(raw_lengths.max()),
                "standardize_features": bool(args.standardize_features),
            },
            "loader_params": {
                "simul_or_real": args.simul_or_real,
                "datasets": datasets,
                "task_filter": sorted(list(task_filter)),
                "session_filter": (
                    None if session_filter is None else sorted(list(session_filter))
                ),
                "run_filter": None if run_filter is None else sorted(list(run_filter)),
                "dFC_ids": [int(x) for x in args.dFC_ids],
                "dynamic_pred": args.dynamic_pred,
                "normalize_dFC": bool(args.normalize_dFC),
                "FCS_proba_for_SB": bool(args.FCS_proba_for_SB),
                "min_seq_len": int(args.min_seq_len),
                "max_subjects_per_scan": args.max_subjects_per_scan,
                "max_total_sequences": args.max_total_sequences,
                "seed": int(args.seed),
            },
            "ts2vec": {
                "init_kwargs": effective_init_kwargs,
                "fit_kwargs": fit_kwargs,
                "encode_kwargs": encode_kwargs,
                "model_saved": model_saved,
            },
        }
        with open(group_dir / "run_config.json", "w") as f:
            json.dump(config_to_save, f, indent=2)

        run_summaries.append(
            {
                "dFC_id": int(dFC_id),
                "measure_name": str(measure_name),
                "feature_dim": int(feature_dim),
                "n_sequences": int(len(sequences)),
                "seq_len_used": int(X_ts2vec.shape[1]),
                "embedding_shape": list(np.asarray(full_series_embeddings).shape),
                "output_dir": str(group_dir),
            }
        )

        # Avoid holding multiple large arrays/models longer than needed.
        del model

    if skipped_records:
        pd.DataFrame(skipped_records).to_csv(
            Path(output_root) / "skipped_records.csv", index=False
        )
    with open(Path(output_root) / "run_summary.json", "w") as f:
        json.dump(run_summaries, f, indent=2)

    print(f"Finished. Outputs written to: {output_root}")


if __name__ == "__main__":
    main()
