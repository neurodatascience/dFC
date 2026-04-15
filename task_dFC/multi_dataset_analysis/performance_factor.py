import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_functions import (  # pyright: ignore[reportMissingImports]
    RDoC_MAP,
    canon_task,
    savefig_pub,
    setup_pub_style,
)

LEVEL = "group_lvl"
GROUP = "test"

CLASSIFIER_METRIC_MAP = {
    "Logistic regression": "Logistic regression balanced accuracy",
    "SVM": "SVM balanced accuracy",
}

TIMING_FEATURES = [
    "task_ratio_avg",
    "transition_freq_avg",
    "OI_median",
    "rest_durations_median",
    "task_durations_median",
    "rest_durations_iqr",
    "task_durations_iqr",
]

COHEN_FEATURES = [
    "CohensD_max",
    "CohensD_mean",
]

TSNR_FEATURES = [
    "median_tsnr_avg_over_subjects",
]

CORR_EXCLUDE_COLUMNS = {
    "RDoC",
    "task",
    "run",
    "dFC assessment method",
    "classifier model",
    "embedding",
    "classification_balanced_accuracy",
}

TOP_BOTTOM_QUANTILE = 0.2
PERFORMANCE_GROUP_LABELS = ["Low", "Medium", "High"]

DEFAULT_FACTOR_LABEL_MAP = {
    "task_ratio_avg": "average task ratio",
    "task_durations_iqr": "task duration IQR",
    "task_durations_median": "task duration median",
    "rest_durations_iqr": "rest duration IQR",
    "rest_durations_median": "rest duration median",
    "OI_median": "median OI",
    "CohensD_mean": "mean Cohen's d",
    "CohensD_max": "max Cohen's d",
    "transition_freq_avg": "average transition frequency",
    "median_tsnr_avg_over_subjects": "median tSNR averaged over subjects",
}


def get_domain_axis_label(simul_or_real):
    return "RDoC domain" if simul_or_real == "real" else "Simulation design category"


def parse_args():
    helptext = """
    Build a unified run-level dataframe linking ML performance to task factors.
    """
    parser = argparse.ArgumentParser(description=helptext)
    parser.add_argument(
        "--multi_dataset_info",
        type=str,
        required=True,
        help="path to multi-dataset info file",
    )
    parser.add_argument(
        "--simul_or_real",
        type=str,
        required=True,
        choices=["simulated", "real"],
        help="Specify 'simulated' or 'real' data",
    )
    return parser.parse_args()


def read_json(json_file):
    with open(json_file, "r") as file_obj:
        return json.load(file_obj)


def load_npy_dict(path, label):
    assert os.path.exists(path), f"{label} file does not exist: {path}"
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray):
        loaded = loaded.item()
    assert isinstance(loaded, dict), f"{label} must be a dictionary. Got {type(loaded)}"
    return loaded


def assert_required_keys(data_dict, required_keys, label):
    missing = [key for key in required_keys if key not in data_dict]
    assert not missing, f"Missing required keys in {label}: {missing}"


def dict_to_df(data_dict, label):
    lengths = {key: len(value) for key, value in data_dict.items()}
    unique_lengths = set(lengths.values())
    assert len(unique_lengths) == 1, (
        f"Inconsistent column lengths in {label}: {lengths}. "
        "All arrays/lists must have equal length."
    )
    return pd.DataFrame.from_dict(data_dict)


def normalize_run(value):
    if value is None:
        return "none"
    if isinstance(value, float) and np.isnan(value):
        return "none"
    # TSV empty cells are read by pandas as NaN (float) handled above,
    # but guard against empty strings too (e.g. after manual editing).
    if str(value).strip() == "":
        return "none"
    return str(value).strip().lower()


def add_join_keys(df):
    assert "task" in df.columns, "Expected column 'task'"
    assert "run" in df.columns, "Expected column 'run'"
    df = df.copy()
    df["task_key"] = df["task"].astype(str).map(canon_task)
    df["run_key"] = df["run"].map(normalize_run)
    assert (df["task_key"].str.len() > 0).all(), "Found empty normalized task key"
    return df


def get_paths(multi_dataset_info, simul_or_real):
    output_root = multi_dataset_info["output_root"]
    return {
        "ml": f"{output_root}/ML_results/ALL_ML_SCORES_{simul_or_real}.npy",
        "timing": f"{output_root}/task_timing_stats/{simul_or_real}/task_timing_stats_{simul_or_real}.npy",
        "cohensd": f"{output_root}/CohensD/{simul_or_real}/CohensD_ML_{simul_or_real}.npy",
        "tsnr": f"{output_root}/t-SNR/tsnr_summary_grouped.tsv",
        "out_dir": f"{output_root}/performance_factor/{simul_or_real}",
    }


def prepare_ml_df(ml_scores_all):
    ml_scores = ml_scores_all

    required_keys = [
        "task",
        "run",
        "embedding",
        "dFC method",
        "group",
        *CLASSIFIER_METRIC_MAP.values(),
    ]
    assert_required_keys(ml_scores, required_keys, "ALL_ML_SCORES")

    df_ml_wide = dict_to_df(ml_scores, "ALL_ML_SCORES")
    df_ml_wide = df_ml_wide[df_ml_wide["group"] == GROUP].copy()
    assert not df_ml_wide.empty, f"No ML rows found for group='{GROUP}'"

    if "dataset" in df_ml_wide.columns:
        id_cols = ["dataset", "task", "run", "embedding", "dFC method", "group"]
    else:
        id_cols = ["task", "run", "embedding", "dFC method", "group"]

    classifier_frames = []
    for classifier, metric_key in CLASSIFIER_METRIC_MAP.items():
        frame = df_ml_wide[id_cols + [metric_key]].copy()
        frame["classifier model"] = classifier
        frame = frame.rename(columns={metric_key: "classification_balanced_accuracy"})
        classifier_frames.append(frame)

    df_ml = pd.concat(classifier_frames, ignore_index=True)
    df_ml = df_ml.rename(columns={"dFC method": "dFC assessment method"})

    score = df_ml["classification_balanced_accuracy"].astype(float)
    assert np.isfinite(score).all(), "ML performance contains NaN/Inf values"
    assert ((score >= 0.0) & (score <= 1.0)).all(), (
        "Expected balanced accuracy in [0, 1]. "
        f"Observed min={score.min()}, max={score.max()}"
    )

    return add_join_keys(df_ml)


def prepare_timing_df(timing_dict):
    required_keys = ["task", "run", *TIMING_FEATURES]
    assert_required_keys(timing_dict, required_keys, "task_timing_stats")
    df_timing = dict_to_df(timing_dict, "task_timing_stats")

    keep_cols = ["task", "run", *TIMING_FEATURES]
    if "dataset" in df_timing.columns:
        keep_cols = ["dataset", *keep_cols]
    df_timing = df_timing[keep_cols].copy()

    for col in TIMING_FEATURES:
        values = df_timing[col].astype(float)
        assert np.isfinite(values).all(), f"Timing feature '{col}' contains NaN/Inf"

    return add_join_keys(df_timing)


def prepare_cohensd_df(cohensd_dict):
    required_keys = ["task", "run", *COHEN_FEATURES]
    assert_required_keys(cohensd_dict, required_keys, "CohensD_ML")
    df_cohensd = dict_to_df(cohensd_dict, "CohensD_ML")

    keep_cols = ["task", "run", *COHEN_FEATURES]
    if "dataset" in df_cohensd.columns:
        keep_cols = ["dataset", *keep_cols]
    df_cohensd = df_cohensd[keep_cols].copy()

    for col in COHEN_FEATURES:
        values = df_cohensd[col].astype(float)
        assert np.isfinite(values).all(), f"Cohen's D feature '{col}' contains NaN/Inf"

    return add_join_keys(df_cohensd)


def prepare_tsnr_df(tsnr_path):
    assert os.path.exists(tsnr_path), f"tSNR file does not exist: {tsnr_path}"
    df_tsnr = pd.read_csv(tsnr_path, sep="\t")

    required_cols = ["dataset", "task", "run", *TSNR_FEATURES]
    missing = [col for col in required_cols if col not in df_tsnr.columns]
    assert not missing, f"Missing required columns in tsnr_summary_grouped.tsv: {missing}"

    df_tsnr = df_tsnr[["dataset", "task", "run", *TSNR_FEATURES]].copy()

    # Validate tSNR values (allow NaN — runs with no data are left empty as specified)
    tsnr_vals = df_tsnr["median_tsnr_avg_over_subjects"].astype(float)
    assert (
        tsnr_vals.dropna() > 0
    ).all(), (
        "median_tsnr_avg_over_subjects contains non-positive values where data is present"
    )

    return add_join_keys(df_tsnr)


def choose_join_keys(df_ml, df_timing, df_cohensd, df_tsnr=None):
    sources = [df_ml, df_timing, df_cohensd]
    if df_tsnr is not None:
        sources.append(df_tsnr)

    has_dataset_everywhere = all("dataset" in df.columns for df in sources)

    base_keys = ["task_key", "run_key"]
    dataset_keys = ["dataset", *base_keys]

    timing_dupes_base = df_timing.duplicated(subset=base_keys).sum()
    cohensd_dupes_base = df_cohensd.duplicated(subset=base_keys).sum()
    tsnr_dupes_base = (
        df_tsnr.duplicated(subset=base_keys).sum() if df_tsnr is not None else 0
    )

    if timing_dupes_base == 0 and cohensd_dupes_base == 0 and tsnr_dupes_base == 0:
        return base_keys

    if has_dataset_everywhere:
        timing_dupes_dataset = df_timing.duplicated(subset=dataset_keys).sum()
        cohensd_dupes_dataset = df_cohensd.duplicated(subset=dataset_keys).sum()
        tsnr_dupes_dataset = (
            df_tsnr.duplicated(subset=dataset_keys).sum() if df_tsnr is not None else 0
        )
        assert timing_dupes_dataset == 0, (
            "task_timing_stats still has duplicate rows per dataset/task/run after "
            f"normalization. duplicate_count={timing_dupes_dataset}"
        )
        assert cohensd_dupes_dataset == 0, (
            "CohensD_ML still has duplicate rows per dataset/task/run after "
            f"normalization. duplicate_count={cohensd_dupes_dataset}"
        )
        if df_tsnr is not None:
            assert tsnr_dupes_dataset == 0, (
                "tsnr_summary_grouped still has duplicate rows per dataset/task/run after "
                f"normalization. duplicate_count={tsnr_dupes_dataset}"
            )
        return dataset_keys

    raise AssertionError(
        "Ambiguous join on task/run (duplicates found), and dataset is not available "
        "in all sources to disambiguate."
    )


def merge_with_checks(df_ml, df_timing, df_cohensd, join_keys, df_tsnr=None):
    timing_cols = join_keys + TIMING_FEATURES
    cohensd_cols = join_keys + COHEN_FEATURES

    df_merged = df_ml.merge(
        df_timing[timing_cols],
        on=join_keys,
        how="left",
        validate="many_to_one",
        indicator="timing_merge",
    )
    timing_unmatched = (df_merged["timing_merge"] != "both").sum()
    assert (
        timing_unmatched == 0
    ), f"Could not match timing stats for {timing_unmatched} ML rows using keys {join_keys}"
    df_merged = df_merged.drop(columns=["timing_merge"])

    df_merged = df_merged.merge(
        df_cohensd[cohensd_cols],
        on=join_keys,
        how="left",
        validate="many_to_one",
        indicator="cohensd_merge",
    )
    cohensd_unmatched = (df_merged["cohensd_merge"] != "both").sum()
    assert (
        cohensd_unmatched == 0
    ), f"Could not match Cohen's D stats for {cohensd_unmatched} ML rows using keys {join_keys}"
    df_merged = df_merged.drop(columns=["cohensd_merge"])

    if df_tsnr is not None:
        tsnr_cols = join_keys + TSNR_FEATURES
        # tSNR: left join — rows with no tSNR data (e.g. None-run datasets not in file)
        # will have NaN, which is acceptable as specified.
        df_merged = df_merged.merge(
            df_tsnr[tsnr_cols],
            on=join_keys,
            how="left",
            validate="many_to_one",
        )

    return df_merged


def add_rdoc(df, simul_or_real):
    task_to_domain = RDoC_MAP[simul_or_real]["TASK2DOMAIN"]
    df = df.copy()
    df["RDoC"] = df["task_key"].map(task_to_domain)

    missing_mask = df["RDoC"].isna()
    if missing_mask.any():
        missing_tasks = sorted(df.loc[missing_mask, "task"].astype(str).unique())
        raise AssertionError(
            "Missing RDoC mapping for tasks (after canonicalization): "
            f"{missing_tasks}. Update helper_functions.RDoC_MAP if needed."
        )

    return df


def finalize_columns(df):
    cols = []
    if "dataset" in df.columns:
        cols.append("dataset")
    cols += [
        "task",
        "run",
        "RDoC",
        *TIMING_FEATURES,
        *COHEN_FEATURES,
        "dFC assessment method",
        "classifier model",
        "embedding",
        "classification_balanced_accuracy",
    ]

    if all(col in df.columns for col in TSNR_FEATURES):
        insert_at = cols.index("dFC assessment method")
        cols[insert_at:insert_at] = TSNR_FEATURES

    missing = [col for col in cols if col not in df.columns]
    assert not missing, f"Missing expected final columns: {missing}"

    out = df[cols].copy()
    sort_cols = ["task", "run", "dFC assessment method", "classifier model", "embedding"]
    if "dataset" in out.columns:
        sort_cols = ["dataset", *sort_cols]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def save_outputs(df, out_dir, simul_or_real):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = f"{out_dir}/performance_factor_{simul_or_real}.csv"
    pkl_path = f"{out_dir}/performance_factor_{simul_or_real}.pkl"
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    return csv_path, pkl_path


def build_correlation_table(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    factor_cols = [
        col
        for col in numeric_cols
        if col not in CORR_EXCLUDE_COLUMNS and col != "classification_balanced_accuracy"
    ]
    assert factor_cols, "No numeric factor columns available for correlation analysis"

    rows = []
    for method, group_df in df.groupby("dFC assessment method", observed=True):
        for factor in factor_cols:
            pair_df = group_df[[factor, "classification_balanced_accuracy"]].dropna()
            n_samples = len(pair_df)

            if (
                n_samples < 3
                or pair_df[factor].nunique(dropna=True) < 2
                or pair_df["classification_balanced_accuracy"].nunique(dropna=True) < 2
            ):
                corr = np.nan
            else:
                corr = pair_df[factor].corr(
                    pair_df["classification_balanced_accuracy"], method="pearson"
                )

            rows.append(
                {
                    "factor": factor,
                    "dFC assessment method": method,
                    "correlation": corr,
                    "n_samples": n_samples,
                }
            )

    corr_df = pd.DataFrame(rows)
    corr_df["factor"] = pd.Categorical(
        corr_df["factor"], categories=factor_cols, ordered=True
    )
    corr_df = corr_df.sort_values(["factor", "dFC assessment method"]).reset_index(
        drop=True
    )
    return corr_df


def get_numeric_factor_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    factor_cols = [
        col
        for col in numeric_cols
        if col not in CORR_EXCLUDE_COLUMNS and col != "classification_balanced_accuracy"
    ]
    assert factor_cols, "No numeric factor columns available for analysis"
    return factor_cols


def plot_factor_correlation_pointplot(corr_df, out_dir, simul_or_real):
    valid_df = corr_df.dropna(subset=["correlation"]).copy()
    assert (
        not valid_df.empty
    ), "All factor correlations are NaN; cannot generate correlation pointplot"

    n_factors = valid_df["factor"].nunique()
    width = max(10, 0.75 * n_factors)
    height = 7.0

    figure, ax = plt.subplots(figsize=(width, height))
    sns.pointplot(
        data=valid_df,
        x="factor",
        y="correlation",
        hue="dFC assessment method",
        dodge=0.4,
        errorbar=None,
        markers="o",
        linestyles="",
        ax=ax,
    )

    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Factor")
    ax.set_ylabel("Corr. with balanced accuracy")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))
    ax.grid(True, axis="y", which="major", linestyle="-", alpha=0.4)
    ax.grid(True, axis="y", which="minor", linestyle="--", alpha=0.22)

    ax.legend(title="dFC assessment method", frameon=True)
    sns.despine(ax=ax, top=True, right=True)
    figure.tight_layout()

    fig_path = f"{out_dir}/performance_factor_correlation_pointplot_{simul_or_real}.png"
    savefig_pub(fig_path)
    plt.close(figure)
    return fig_path


def build_top_bottom_profile_table(df, quantile=TOP_BOTTOM_QUANTILE):
    assert 0 < quantile < 0.5, "quantile must be in (0, 0.5)"

    factor_cols = get_numeric_factor_columns(df)

    rows = []
    for method, method_df in df.groupby("dFC assessment method", observed=True):
        score = method_df["classification_balanced_accuracy"].astype(float)
        low_thr = score.quantile(quantile)
        high_thr = score.quantile(1 - quantile)

        bottom_df = method_df[score <= low_thr].copy()
        top_df = method_df[score >= high_thr].copy()

        if len(top_df) < 3 or len(bottom_df) < 3:
            print(
                f"[TopBottom] Skipping method '{method}' due to too few samples "
                f"(top={len(top_df)}, bottom={len(bottom_df)})."
            )
            continue

        for factor in factor_cols:
            top_vals = top_df[factor].astype(float).dropna()
            bottom_vals = bottom_df[factor].astype(float).dropna()
            n_top = len(top_vals)
            n_bottom = len(bottom_vals)

            mean_top = np.nan if n_top == 0 else float(top_vals.mean())
            mean_bottom = np.nan if n_bottom == 0 else float(bottom_vals.mean())
            mean_diff = mean_top - mean_bottom if (n_top > 0 and n_bottom > 0) else np.nan

            cohens_d = np.nan
            if n_top >= 2 and n_bottom >= 2:
                var_top = float(np.var(top_vals, ddof=1))
                var_bottom = float(np.var(bottom_vals, ddof=1))
                pooled_num = ((n_top - 1) * var_top) + ((n_bottom - 1) * var_bottom)
                pooled_den = n_top + n_bottom - 2
                if pooled_den > 0:
                    pooled_std = np.sqrt(pooled_num / pooled_den)
                    if np.isfinite(pooled_std) and pooled_std > 0:
                        cohens_d = mean_diff / pooled_std

            rows.append(
                {
                    "factor": factor,
                    "dFC assessment method": method,
                    "mean_top": mean_top,
                    "mean_bottom": mean_bottom,
                    "mean_diff": mean_diff,
                    "cohens_d": cohens_d,
                    "n_top": n_top,
                    "n_bottom": n_bottom,
                    "low_threshold": float(low_thr),
                    "high_threshold": float(high_thr),
                    "n_method_total": int(len(method_df)),
                }
            )

    assert rows, "No method had enough samples for top-vs-bottom profile analysis"

    profile_df = pd.DataFrame(rows)
    profile_df["abs_cohens_d"] = profile_df["cohens_d"].abs()
    profile_df = profile_df.sort_values(
        ["abs_cohens_d", "factor", "dFC assessment method"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    return profile_df


def plot_top_bottom_profile(profile_df, out_dir, simul_or_real, factor_label_map=None):
    valid_df = profile_df.dropna(subset=["cohens_d"]).copy()
    assert (
        not valid_df.empty
    ), "No valid Cohen's d values available for top-vs-bottom profile plot"

    factor_label_map = factor_label_map or DEFAULT_FACTOR_LABEL_MAP
    valid_df["factor_display"] = (
        valid_df["factor"].map(factor_label_map).fillna(valid_df["factor"].astype(str))
    )

    factor_order = (
        valid_df.groupby("factor", observed=True)["abs_cohens_d"]
        .max()
        .sort_values(ascending=True)
        .index.tolist()
    )
    valid_df = valid_df.sort_values(["factor", "dFC assessment method"])
    factor_display_order = [
        factor_label_map.get(factor, factor) for factor in factor_order
    ]
    valid_df["factor_display"] = pd.Categorical(
        valid_df["factor_display"], categories=factor_display_order, ordered=True
    )

    method_order = sorted(valid_df["dFC assessment method"].astype(str).unique())
    vivid_palette = sns.color_palette("tab10", n_colors=len(method_order))
    method_palette = {m: vivid_palette[i] for i, m in enumerate(method_order)}

    # More generous height keeps rows legible when many factors are shown.
    height = max(7.0, 0.72 * len(factor_order))
    width = 16.5
    figure, ax = plt.subplots(figsize=(width, height))

    # Alternating row bands make it easier to track each factor across methods.
    for idx, factor in enumerate(factor_order):
        if idx % 2 == 0:
            ax.axhspan(idx - 0.5, idx + 0.5, color="#F4F6FA", alpha=0.75, zorder=0)

    sns.stripplot(
        data=valid_df,
        x="cohens_d",
        y="factor_display",
        hue="dFC assessment method",
        order=factor_display_order,
        hue_order=method_order,
        palette=method_palette,
        dodge=True,
        jitter=0.08,
        size=8.2,
        linewidth=0.85,
        edgecolor="white",
        alpha=0.98,
        ax=ax,
    )

    max_abs = float(np.nanmax(np.abs(valid_df["cohens_d"].values)))
    x_pad = max(0.15, 0.12 * max_abs)
    x_lim = max_abs + x_pad

    ax.axvline(0.0, color="#1F1F1F", linestyle="--", linewidth=1.5, zorder=3)
    ax.set_xlim(-x_lim, x_lim)
    ax.set_xlabel("Effect size (Cohen's d): Top 20% vs Bottom 20% within method")
    ax.set_ylabel("Factor")

    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # Keep tick labels readable but not too sparse using a "nice number" step.
    span = 2.0 * x_lim
    target_ticks = 11  # aim for ~11 major ticks across full span
    raw_major_step = span / (target_ticks - 1)
    if raw_major_step <= 0:
        major_step = 0.5
    else:
        exponent = np.floor(np.log10(raw_major_step))
        base = 10.0**exponent
        fraction = raw_major_step / base
        if fraction <= 1.0:
            nice_fraction = 1.0
        elif fraction <= 2.0:
            nice_fraction = 2.0
        elif fraction <= 2.5:
            nice_fraction = 2.5
        elif fraction <= 5.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
        major_step = nice_fraction * base

    minor_step = major_step / 2.0
    ax.xaxis.set_major_locator(MultipleLocator(major_step))
    ax.xaxis.set_minor_locator(MultipleLocator(minor_step))
    ax.grid(True, axis="x", which="major", linestyle="-", linewidth=1.0, alpha=0.35)
    ax.grid(True, axis="x", which="minor", linestyle="--", linewidth=0.8, alpha=0.2)

    legend = ax.legend(
        title="dFC assessment method",
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    if legend is not None:
        legend.get_title().set_fontsize(12)
        for txt in legend.get_texts():
            txt.set_fontsize(11)

    sns.despine(ax=ax, top=True, right=True)
    figure.tight_layout(rect=[0.18, 0, 0.83, 1])

    fig_path = f"{out_dir}/performance_top_bottom_profile_{simul_or_real}.png"
    savefig_pub(fig_path)
    plt.close(figure)
    return fig_path


def _get_present_rdoc_order(df, simul_or_real):
    domain_order = RDoC_MAP[simul_or_real]["DOMAIN_ORDER"]
    present = set(df["RDoC"].dropna().astype(str).unique())
    ordered = [domain for domain in domain_order if domain in present]
    remaining = sorted([domain for domain in present if domain not in ordered])
    return ordered + remaining


def add_performance_group(df):
    df = df.copy()
    score = df["classification_balanced_accuracy"].astype(float)
    low_thr = score.quantile(0.25)
    high_thr = score.quantile(0.75)
    assert (
        low_thr < high_thr
    ), "Performance-group thresholds collapsed; cannot form 25/50/25 groups"

    df["performance_group"] = pd.cut(
        score,
        bins=[-np.inf, low_thr, high_thr, np.inf],
        labels=PERFORMANCE_GROUP_LABELS,
        include_lowest=True,
    )
    assert df["performance_group"].notna().all(), "Failed to assign performance groups"
    return df, float(low_thr), float(high_thr)


def build_rdoc_performance_group_table(df, simul_or_real):
    df_grouped, low_thr, high_thr = add_performance_group(df)
    rdoc_order = _get_present_rdoc_order(df_grouped, simul_or_real)
    assert rdoc_order, "No RDoC values found for RDoC-performance grouping"

    count_table = (
        df_grouped.groupby(["RDoC", "performance_group"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(index=rdoc_order, columns=PERFORMANCE_GROUP_LABELS, fill_value=0)
    )
    proportion_table = count_table.div(count_table.sum(axis=1), axis=0)
    assert np.isclose(
        proportion_table.sum(axis=1), 1.0
    ).all(), "RDoC performance-group proportions do not sum to 1"

    summary_long = (
        count_table.stack()
        .rename("count")
        .reset_index()
        .rename(columns={"level_1": "performance_group"})
    )
    summary_long["proportion"] = [
        proportion_table.loc[row.RDoC, row.performance_group]
        for row in summary_long.itertuples(index=False)
    ]
    summary_long["low_threshold"] = low_thr
    summary_long["high_threshold"] = high_thr
    return summary_long, count_table, proportion_table


def plot_rdoc_performance_group_stacked_bar(
    proportion_table, out_dir, simul_or_real, x_label="RDoC domain"
):
    width = max(10.0, 1.6 * len(proportion_table.index))
    figure, ax = plt.subplots(figsize=(width, 7.2))

    palette = {
        "Low": "#D1495B",
        "Medium": "#F4D35E",
        "High": "#2A9D8F",
    }
    proportion_pct = proportion_table.mul(100.0)
    bottom = np.zeros(len(proportion_pct.index))

    for label in PERFORMANCE_GROUP_LABELS:
        values = proportion_pct[label].to_numpy()
        ax.bar(
            proportion_pct.index,
            values,
            bottom=bottom,
            label=label,
            color=palette[label],
            edgecolor="white",
            linewidth=1.0,
        )
        bottom += values

    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_horizontalalignment("right")
        label.set_fontsize(12)
        label.set_fontweight("bold")

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("Samples (%)", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(True, axis="y", which="major", linestyle="-", alpha=0.34)
    ax.grid(True, axis="y", which="minor", linestyle="--", alpha=0.16)
    ax.tick_params(axis="y", labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    legend = ax.legend(
        title="Performance group", frameon=True, fontsize=11, title_fontsize=12
    )
    if legend is not None:
        legend.get_title().set_fontweight("bold")
        for txt in legend.get_texts():
            txt.set_fontweight("bold")
    sns.despine(ax=ax, top=True, right=True)
    figure.tight_layout()

    fig_path = f"{out_dir}/performance_group_by_rdoc_stacked_{simul_or_real}.png"
    savefig_pub(fig_path)
    plt.close(figure)
    return fig_path


def plot_rdoc_performance_group_heatmap(
    proportion_table, out_dir, simul_or_real, x_label="Performance group"
):
    annot_table = proportion_table.mul(100.0).applymap(lambda value: f"{value:.1f}%")

    figure, ax = plt.subplots(figsize=(8.6, max(5.2, 0.82 * len(proportion_table.index))))
    heatmap = sns.heatmap(
        proportion_table.loc[:, PERFORMANCE_GROUP_LABELS],
        cmap="crest",
        vmin=0.0,
        vmax=1.0,
        annot=annot_table.loc[:, PERFORMANCE_GROUP_LABELS],
        fmt="",
        linewidths=0.9,
        linecolor="white",
        cbar_kws={"shrink": 0.84, "pad": 0.03},
        ax=ax,
    )
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label("Proportion", fontweight="bold", fontsize=12)

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("RDoC domain", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=12, fontweight="bold")
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=12, fontweight="bold")
    ax.set_title("RDoC composition by performance group", pad=12, fontweight="bold")
    figure.tight_layout()

    fig_path = f"{out_dir}/performance_group_by_rdoc_heatmap_{simul_or_real}.png"
    savefig_pub(fig_path)
    plt.close(figure)
    return fig_path


def plot_rdoc_overall_distribution(df, out_dir, simul_or_real, x_label="RDoC domain"):
    rdoc_order = _get_present_rdoc_order(df, simul_or_real)
    assert rdoc_order, "No RDoC values found for plotting"

    width = max(12.0, 1.55 * len(rdoc_order))
    height = 7.0
    figure, ax = plt.subplots(figsize=(width, height))

    palette = sns.color_palette("Spectral", n_colors=len(rdoc_order))
    palette_map = {rdoc: palette[i] for i, rdoc in enumerate(rdoc_order)}

    sns.boxplot(
        data=df,
        x="RDoC",
        y="classification_balanced_accuracy",
        order=rdoc_order,
        showfliers=False,
        width=0.58,
        palette=palette_map,
        linewidth=1.2,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="RDoC",
        y="classification_balanced_accuracy",
        order=rdoc_order,
        palette=palette_map,
        alpha=0.45,
        size=3.1,
        jitter=0.2,
        ax=ax,
    )

    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel("Balanced accuracy", fontweight="bold")
    ax.set_ylim(0.45, 1.02)
    plt.setp(
        ax.get_xticklabels(), rotation=25, ha="right", fontsize=12, fontweight="bold"
    )
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.025))
    ax.tick_params(axis="y", labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(True, axis="y", which="major", linestyle="-", alpha=0.34)
    ax.grid(True, axis="y", which="minor", linestyle="--", alpha=0.18)
    sns.despine(ax=ax, top=True, right=True)
    figure.tight_layout()

    fig_path = f"{out_dir}/performance_by_rdoc_overall_{simul_or_real}.png"
    savefig_pub(fig_path)
    plt.close(figure)
    return fig_path


def plot_rdoc_faceted_distribution(df, out_dir, simul_or_real, x_label="RDoC domain"):
    rdoc_order = _get_present_rdoc_order(df, simul_or_real)
    assert rdoc_order, "No RDoC values found for plotting"

    combo_df = (
        df[["classifier model", "embedding"]]
        .drop_duplicates()
        .sort_values(["classifier model", "embedding"])
    )
    assert not combo_df.empty, "No classifier/embedding combinations found for plotting"

    n_methods = df["dFC assessment method"].nunique()
    # Generous per-domain width so boxes never feel cramped
    n_domains = len(rdoc_order)
    # Each domain gets ~3.1 in; minimum figure width 20 in
    axes_width = max(20.0, 3.1 * n_domains)
    # Reserve more room for the legend column
    legend_width = 4.2
    total_width = axes_width + legend_width
    # Height: keep panels open and readable
    height = max(8.5, 0.42 * n_methods + 6.8)

    fig_paths = []
    for _, combo in combo_df.iterrows():
        classifier = combo["classifier model"]
        embedding = combo["embedding"]

        sub_df = df[
            (df["classifier model"] == classifier) & (df["embedding"] == embedding)
        ].copy()
        if sub_df.empty:
            continue

        figure, ax = plt.subplots(figsize=(total_width, height))

        palette = sns.color_palette("tab10", n_colors=n_methods)
        hue_order = sorted(sub_df["dFC assessment method"].dropna().astype(str).unique())
        method_palette = {method: palette[i] for i, method in enumerate(hue_order)}

        sns.boxplot(
            data=sub_df,
            x="RDoC",
            y="classification_balanced_accuracy",
            hue="dFC assessment method",
            order=rdoc_order,
            showfliers=False,
            width=0.72,
            linewidth=1.35,
            palette=method_palette,
            hue_order=hue_order,
            ax=ax,
        )

        ax.set_ylim(0.45, 1.02)
        ax.set_xlabel(x_label, labelpad=12, fontsize=14, fontweight="bold")
        ax.set_ylabel("Balanced accuracy", labelpad=12, fontsize=14, fontweight="bold")
        ax.set_title(
            f"{classifier}  |  {embedding}",
            fontweight="bold",
            pad=14,
            fontsize=15,
        )
        ax.tick_params(axis="both", labelsize=12)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(MultipleLocator(0.025))
        ax.grid(True, axis="y", which="major", linestyle="-", alpha=0.36)
        ax.grid(True, axis="y", which="minor", linestyle="--", alpha=0.20)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
            label.set_fontsize(13)
            label.set_fontweight("bold")
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.get_legend().remove()
            figure.legend(
                handles,
                labels,
                title="dFC assessment method",
                title_fontsize=12,
                fontsize=11,
                frameon=True,
                loc="center left",
                bbox_to_anchor=(axes_width / total_width + 0.01, 0.5),
            )
            if figure.legends:
                for legend in figure.legends:
                    legend.get_title().set_fontweight("bold")
                    for txt in legend.get_texts():
                        txt.set_fontweight("bold")

        sns.despine(ax=ax, top=True, right=True)
        # Leave right margin for the figure-level legend
        figure.tight_layout(rect=[0, 0, axes_width / total_width, 1])

        classifier_key = str(classifier).replace(" ", "_").replace("/", "-")
        embedding_key = str(embedding).replace(" ", "_").replace("/", "-")
        fig_path = (
            f"{out_dir}/performance_by_rdoc_{classifier_key}"
            f"_{embedding_key}_{simul_or_real}.png"
        )
        plt.savefig(fig_path, bbox_inches="tight", dpi=150)
        plt.close(figure)
        fig_paths.append(fig_path)

    assert fig_paths, "No RDoC per-combination figures were generated"
    return fig_paths


def main():
    args = parse_args()
    setup_pub_style()

    multi_dataset_info = read_json(args.multi_dataset_info)
    paths = get_paths(multi_dataset_info, args.simul_or_real)

    ml_scores_all = load_npy_dict(paths["ml"], "ALL_ML_SCORES")
    timing_dict = load_npy_dict(paths["timing"], "task_timing_stats")
    cohensd_dict = load_npy_dict(paths["cohensd"], "CohensD_ML")

    df_ml = prepare_ml_df(ml_scores_all)
    df_timing = prepare_timing_df(timing_dict)
    df_cohensd = prepare_cohensd_df(cohensd_dict)
    df_tsnr = None
    if args.simul_or_real == "real":
        df_tsnr = prepare_tsnr_df(paths["tsnr"])

    join_keys = choose_join_keys(df_ml, df_timing, df_cohensd, df_tsnr)
    print(f"Using join keys: {join_keys}")

    df = merge_with_checks(df_ml, df_timing, df_cohensd, join_keys, df_tsnr)
    df = add_rdoc(df, args.simul_or_real)
    df = finalize_columns(df)

    csv_path, pkl_path = save_outputs(df, paths["out_dir"], args.simul_or_real)

    corr_df = build_correlation_table(df)
    corr_csv_path = (
        f"{paths['out_dir']}/performance_factor_correlations_{args.simul_or_real}.csv"
    )
    corr_df.to_csv(corr_csv_path, index=False)
    corr_fig_path = plot_factor_correlation_pointplot(
        corr_df, paths["out_dir"], args.simul_or_real
    )

    profile_df = build_top_bottom_profile_table(df, quantile=TOP_BOTTOM_QUANTILE)
    profile_csv_path = (
        f"{paths['out_dir']}/performance_top_bottom_profile_{args.simul_or_real}.csv"
    )
    profile_df.to_csv(profile_csv_path, index=False)
    profile_fig_path = plot_top_bottom_profile(
        profile_df,
        paths["out_dir"],
        args.simul_or_real,
        factor_label_map=DEFAULT_FACTOR_LABEL_MAP,
    )

    domain_x_label = get_domain_axis_label(args.simul_or_real)

    rdoc_overall_path = plot_rdoc_overall_distribution(
        df,
        paths["out_dir"],
        args.simul_or_real,
        x_label=domain_x_label,
    )
    rdoc_faceted_paths = plot_rdoc_faceted_distribution(
        df,
        paths["out_dir"],
        args.simul_or_real,
        x_label=domain_x_label,
    )
    rdoc_group_long_df, rdoc_group_count_table, rdoc_group_prop_table = (
        build_rdoc_performance_group_table(df, args.simul_or_real)
    )
    rdoc_group_csv_path = (
        f"{paths['out_dir']}/performance_group_by_rdoc_{args.simul_or_real}.csv"
    )
    rdoc_group_long_df.to_csv(rdoc_group_csv_path, index=False)
    rdoc_group_bar_path = plot_rdoc_performance_group_stacked_bar(
        rdoc_group_prop_table,
        paths["out_dir"],
        args.simul_or_real,
        x_label=domain_x_label,
    )
    rdoc_group_heatmap_path = plot_rdoc_performance_group_heatmap(
        rdoc_group_prop_table, paths["out_dir"], args.simul_or_real
    )

    print(f"Saved dataframe with shape: {df.shape}")
    print(f"CSV: {csv_path}")
    print(f"PKL: {pkl_path}")
    print(f"Correlation CSV: {corr_csv_path}")
    print(f"Correlation figure: {corr_fig_path}")
    print(f"Top-bottom profile CSV: {profile_csv_path}")
    print(f"Top-bottom profile figure: {profile_fig_path}")
    print(f"RDoC overall figure: {rdoc_overall_path}")
    print(f"RDoC per-combination figures: {len(rdoc_faceted_paths)} files")
    print(f"RDoC performance-group CSV: {rdoc_group_csv_path}")
    print(f"RDoC performance-group stacked bar: {rdoc_group_bar_path}")
    print(f"RDoC performance-group heatmap: {rdoc_group_heatmap_path}")


if __name__ == "__main__":
    main()
