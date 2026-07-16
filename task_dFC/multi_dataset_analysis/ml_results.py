import argparse
import json
import os
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.ticker import PercentFormatter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_functions import (  # pyright: ignore[reportMissingImports]
    boldify_axes,
    build_experiment_display_info,
    relabel_heatmap_rows,
    savefig_pub,
    setup_pub_style,
)

LEVEL = "group_lvl"
KEYS_NOT_TO_INCLUDE = [
    "Logistic regression permutation p_value",
    "Logistic regression permutation score mean",
    "Logistic regression permutation score std",
    "SVM permutation p_value",
    "SVM permutation score mean",
    "SVM permutation score std",
]
GROUP = "test"
TARGETS = [
    ("PCA", "Logistic regression balanced accuracy"),
    ("PLS", "Logistic regression balanced accuracy"),
    ("PCA", "SVM balanced accuracy"),
    ("PLS", "SVM balanced accuracy"),
    ("PCA", "SI"),
    ("PLS", "SI"),
]
TOP_EXPERIMENT_SHAPES = 3
TOP_EXPERIMENT_MARKERS = ["*"]  # star for all top experiments
COLOR_THRESHOLD = 60.0
PER_METHOD_LABEL_SCORE_THRESHOLD = 55.0
SIMULATED_METHOD_MEDIAN_ANNOTATION_THRESHOLD = 80.0
NEUTRAL_COLOR = "#D49B9B"

NON_AIGM_METHODS = frozenset(
    [
        "SlidingWindow",
        "Time-Freq",
        "CAP",
        "ContinuousHMM",
        "Windowless",
        "Clustering",
        "DiscreteHMM",
    ]
)

HYBRID_METHODS = frozenset(
    [
        "AdaptiveDccRandom_hybrid_ensemble",
        "AdaptiveEdgeKalman_hybrid",
        "AdaptiveQuantumReservoir_hybrid",
        "AdaptiveRandomSparse_hybrid",
        "AdaptiveRobustDifferential_hybrid_ensemble",
        "ChangepointMultiscaleExp_hybrid_ensemble",
        "ChangepointRobustExp_hybrid",
        "DccChangepointSparse_hybrid",
        "DccEdgeExponential_hybrid",
        "DccKalmanVolatility_hybrid_ensemble",
        "DifferentialEdgeKalman_hybrid",
        "EdgeKalmanExp_hybrid_ensemble",
        "EdgeRandomReservoir_hybrid_ensemble",
        "EdgeStftQuantum_hybrid",
        "KalmanReservoirMultiscale_hybrid",
        "MultiscaleDccKalman_hybrid",
        "QuantumMultiscaleKalman_hybrid_ensemble",
        "QuantumRandomExp_hybrid",
        "RandomFourierKalman_hybrid",
        "ReservoirChangepointRobust_hybrid_ensemble",
        "ReservoirEdgeDcc_hybrid",
        "RobustDifferentialStft_hybrid",
        "RobustSparseEdge_hybrid",
        "RobustVolatilitySliding_hybrid_ensemble",
        "SlidingVolatilityDcc_hybrid",
        "SparseDccExp_hybrid_ensemble",
        "StftEdgeAdaptive_hybrid_ensemble",
        "StftExpKalman_hybrid",
        "StftQuantumSparse_hybrid_ensemble",
        "VolatilityAdaptiveEdge_hybrid"
    ]
)

_AIGM_COLOR = "#0077B6"
_HYBRID_COLOR = "#1A759F"
_NON_AIGM_COLOR = "#E63946"
_NON_AIGM_LABEL_COLOR = "#D4721A"
_HYBRID_LABEL_COLOR = "#1A759F"
_METRIC_SHORT = {
    "Logistic regression balanced accuracy": "LogReg BA",
    "SVM balanced accuracy": "SVM BA",
    "SI": "SI",
}


def parse_args():
    helptext = """
    Script to make figures/tables from multi-dataset ML results.
    """
    parser = argparse.ArgumentParser(description=helptext)
    parser.add_argument(
        "--multi_dataset_info", type=str, help="path to multi-dataset info file"
    )
    parser.add_argument(
        "--simul_or_real", type=str, help="Specify 'simulated' or 'real' data"
    )
    return parser.parse_args()


def read_json(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def get_analysis_config(multi_dataset_info, simul_or_real):
    if simul_or_real == "real":
        return multi_dataset_info["real_data"]
    if simul_or_real == "simulated":
        return multi_dataset_info["simulated_data"]
    raise ValueError(f"Invalid simul_or_real: {simul_or_real}")


def get_classification_input_dir(ml_root, dataset_info):
    sessions = dataset_info.get("SESSIONS") or [None]
    session = sessions[0]
    if session is None:
        return f"{ml_root}/classification"
    return f"{ml_root}/classification/{session}"


def filter_ml_scores(ml_scores_new, tasks_to_include):
    filtered_scores = {
        key: [] for key in ml_scores_new[LEVEL].keys() if key not in KEYS_NOT_TO_INCLUDE
    }

    for index, task in enumerate(ml_scores_new[LEVEL]["task"]):
        if task not in tasks_to_include:
            continue
        for key in filtered_scores:
            filtered_scores[key].append(ml_scores_new[LEVEL][key][index])

    return filtered_scores


def merge_ml_scores(all_ml_scores, ml_scores_new_updated):
    if all_ml_scores is None:
        return ml_scores_new_updated

    for key, values in ml_scores_new_updated.items():
        if key in all_ml_scores:
            all_ml_scores[key].extend(values)
    return all_ml_scores


def collect_all_ml_scores(main_root, datasets, tasks_to_include):
    all_ml_scores = None

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")
        dataset_info_file = f"{main_root}/{dataset}/codes/dataset_info.json"
        ml_root = f"{main_root}/{dataset}/derivatives/ML"
        dataset_info = read_json(dataset_info_file)
        input_dir = get_classification_input_dir(ml_root, dataset_info)

        if not os.path.exists(input_dir):
            print(
                f"Input directory {input_dir} does not exist. Skipping dataset {dataset}."
            )
            continue

        all_ml_scores_files = [
            filename
            for filename in os.listdir(input_dir)
            if "ML_scores_classify_" in filename
        ]

        for filename in all_ml_scores_files:
            try:
                ml_scores_new = np.load(
                    f"{input_dir}/{filename}", allow_pickle=True
                ).item()
                filtered_scores = filter_ml_scores(ml_scores_new, tasks_to_include)
                all_ml_scores = merge_ml_scores(all_ml_scores, filtered_scores)
            except Exception as error:
                print(f"Error loading {filename}: {error}")
                continue

    return all_ml_scores


def validate_score_lengths(all_ml_scores):
    if all_ml_scores is None:
        return

    lengths = [len(values) for values in all_ml_scores.values()]
    if len(set(lengths)) != 1:
        print(
            "Warning: Not all keys have the same length in ALL_ML_SCORES. "
            f"key and length pairs: {dict(zip(all_ml_scores.keys(), lengths))}"
        )


def save_all_ml_scores(all_ml_scores, output_root, simul_or_real):
    os.makedirs(output_root, exist_ok=True)
    np.save(f"{output_root}/ALL_ML_SCORES_{simul_or_real}.npy", all_ml_scores)


def prepare_metric_dataframe(
    all_ml_scores, tasks_to_include, embedding, metric, simul_or_real
):
    df = pd.DataFrame.from_dict(all_ml_scores)
    df = df[df["task"].isin(tasks_to_include)]
    df = df[(df["embedding"] == embedding) & (df["group"] == GROUP)].copy()

    method_order = sorted(df["dFC method"].unique(), key=lambda method: method.lower())
    df["dFC method"] = pd.Categorical(
        df["dFC method"], categories=method_order, ordered=True
    )

    task_order, task_to_experiment, experiment_order, experiment_palette = (
        build_experiment_display_info(
            df["task"].unique(),
            task_reference_order=tasks_to_include,
            simul_or_real=simul_or_real,
        )
    )
    df["experiment"] = df["task"].map(task_to_experiment)

    return (
        df,
        method_order,
        task_order,
        task_to_experiment,
        experiment_order,
        experiment_palette,
    )


def build_best_and_multi_tables(df, metric):
    counts_task = df.groupby("task")["run"].nunique()
    multi_tasks = counts_task[counts_task > 1].index
    df_multi = df[df["task"].isin(multi_tasks)].copy()

    df_best = (
        df.sort_values(["task", "dFC method", metric], ascending=[True, True, False])
        .drop_duplicates(subset=["task", "dFC method"], keep="first")
        .rename(columns={metric: "score"})
    )

    return df_best, df_multi


def get_pointplot_limits(metric):
    if metric == "SI":
        return -1.0, 1.0
    return 0.5, 1.0


def convert_threshold_to_score_scale(threshold, metric):
    if metric != "SI" and threshold > 1.0:
        return threshold / 100.0
    return threshold


def get_heatmap_limits(metric):
    if metric == "SI":
        return None, 1.0, 0.0
    return 0.5 - 1e-6, 1.0, 0.5


def style_boxplot(ax, box_edge):
    for artist in ax.artists:
        artist.set_edgecolor(box_edge)
        facecolor = artist.get_facecolor()
        artist.set_facecolor((facecolor[0], facecolor[1], facecolor[2], 0.12))
    for line in ax.lines:
        line.set_color(box_edge)
        line.set_alpha(0.5)
        line.set_zorder(1)


def overlay_method_means(ax, df_best, lower, upper, lw=2.4, halfwidth=0.25):
    means = df_best.groupby("dFC method", observed=True)["score"].mean()
    yticks = ax.get_yticks()
    yticklabels = [tick.get_text() for tick in ax.get_yticklabels()]
    y_positions = {label: yticks[index] for index, label in enumerate(yticklabels)}

    for method, mean_score in means.items():
        if method not in y_positions or pd.isna(mean_score):
            continue
        mean_score = min(upper, max(lower, mean_score))
        y_pos = y_positions[method]
        ax.vlines(
            mean_score,
            y_pos - halfwidth,
            y_pos + halfwidth,
            colors="#050505",
            lw=lw,
            zorder=3,
        )


def finalize_marker_edges(ax):
    for line in ax.lines:
        try:
            line.set_markeredgecolor("#222222")
            line.set_markeredgewidth(0.8)
        except Exception:
            pass


def get_colored_experiment_mask(df_best, color_threshold=COLOR_THRESHOLD):
    """Return set of experiments with max score >= color_threshold across all methods."""
    max_scores = df_best.groupby("experiment", observed=True)["score"].max()
    return set(max_scores[max_scores >= color_threshold].index)


def get_top_experiments_by_mean(df_best, top_experiment_shapes=TOP_EXPERIMENT_SHAPES):
    if top_experiment_shapes <= 0:
        return []
    return (
        df_best.groupby("experiment", observed=True)["score"]
        .mean()
        .sort_values(ascending=False)
        .head(top_experiment_shapes)
        .index.tolist()
    )


def create_neutral_palette(experiment_order, colored_experiments, vibrant_palette):
    """Create palette: neutral for non-colored experiments, vibrant for colored ones."""
    palette = {}
    for exp in experiment_order:
        if exp in colored_experiments:
            palette[exp] = vibrant_palette[exp]
        else:
            palette[exp] = NEUTRAL_COLOR
    return palette


def extract_pointplot_coordinates(ax, method_order, experiment_order, experiment_palette):
    candidate_lines = []
    for line in ax.lines:
        x_data = np.asarray(line.get_xdata(), dtype=float)
        y_data = np.asarray(line.get_ydata(), dtype=float)
        marker = line.get_marker()
        if marker in {None, "", "None", " "}:
            continue
        if x_data.size != len(method_order) or y_data.size != len(method_order):
            continue
        candidate_lines.append(line)

    assigned_lines = {
        experiment: candidate_lines[idx]
        for idx, experiment in enumerate(experiment_order)
        if idx < len(candidate_lines)
    }

    coordinates = {}
    for experiment, line in assigned_lines.items():
        x_data = np.asarray(line.get_xdata(), dtype=float)
        y_data = np.asarray(line.get_ydata(), dtype=float)
        coordinates[experiment] = {}
        for method_index, method in enumerate(method_order):
            x_value = x_data[method_index]
            y_value = y_data[method_index]
            if np.isnan(x_value) or np.isnan(y_value):
                continue
            coordinates[experiment][method] = (x_value, y_value)
    return coordinates


def resize_colored_markers(
    ax,
    experiment_order,
    colored_experiments,
    method_order,
    base_size=5,
    colored_size=8,
):
    """Make circles for colored (high-performing) experiments slightly bigger."""
    candidate_lines = []
    for line in ax.lines:
        x_data = np.asarray(line.get_xdata(), dtype=float)
        y_data = np.asarray(line.get_ydata(), dtype=float)
        marker = line.get_marker()
        if marker in {None, "", "None", " "}:
            continue
        if x_data.size != len(method_order) or y_data.size != len(method_order):
            continue
        candidate_lines.append(line)

    for idx, experiment in enumerate(experiment_order):
        if idx >= len(candidate_lines):
            break
        size = colored_size if experiment in colored_experiments else base_size
        candidate_lines[idx].set_markersize(size)


def overlay_top_experiment_shapes(
    ax,
    df_best,
    point_coordinates,
    shape_palette,
    top_experiment_shapes,
):
    if top_experiment_shapes <= 0:
        return

    top_experiments = get_top_experiments_by_mean(df_best, top_experiment_shapes)

    for rank, experiment in enumerate(top_experiments):
        if experiment not in point_coordinates:
            continue
        marker = TOP_EXPERIMENT_MARKERS[rank % len(TOP_EXPERIMENT_MARKERS)]
        points = list(point_coordinates[experiment].values())
        if not points:
            continue
        x_vals = [pt[0] for pt in points]
        y_vals = [pt[1] for pt in points]
        ax.scatter(
            x_vals,
            y_vals,
            marker=marker,
            s=250,
            c=shape_palette[experiment],
            edgecolors="#111111",
            linewidths=1.0,
            zorder=8,
        )


def annotate_per_method_quartile(
    ax,
    df_best,
    point_coordinates,
    method_order,
    colored_experiments,
    metric,
    simul_or_real,
    score_threshold=PER_METHOD_LABEL_SCORE_THRESHOLD,
):
    """
    Default behavior:
    - annotate colored experiments in top quartile and above score_threshold.

    Simulated + non-SI override:
    - if a method median is above SIMULATED_METHOD_MEDIAN_ANNOTATION_THRESHOLD,
      annotate all experiments for that method.
    """
    simulated_non_si = simul_or_real == "simulated" and metric != "SI"
    if not colored_experiments and not simulated_non_si:
        return

    simulated_median_threshold = convert_threshold_to_score_scale(
        SIMULATED_METHOD_MEDIAN_ANNOTATION_THRESHOLD, metric
    )

    yticks = ax.get_yticks()
    yticklabels = [t.get_text() for t in ax.get_yticklabels()]
    method_positions = {lab: yticks[i] for i, lab in enumerate(yticklabels)}

    for method in method_order:
        method_df = df_best[df_best["dFC method"] == method]
        if method_df.empty:
            continue

        scores = method_df["score"].values
        quartile_threshold = np.percentile(scores, 75)

        if simulated_non_si and np.nanmedian(scores) > simulated_median_threshold:
            qualify_rows = method_df
        else:
            qualify_rows = method_df[
                method_df["experiment"].isin(colored_experiments)
                & (method_df["score"] > score_threshold)
                & (method_df["score"] >= quartile_threshold)
            ]

        method_center = method_positions.get(method)
        if method_center is None:
            continue

        for _, row in qualify_rows.iterrows():
            experiment = row["experiment"]
            if experiment not in point_coordinates:
                continue
            if method not in point_coordinates[experiment]:
                continue

            x_value, y_value = point_coordinates[experiment][method]

            # Annotate to the right; position text above/below based on dodge offset
            va_align = "bottom" if y_value > method_center else "top"
            y_offset = 3 if y_value > method_center else -3

            ax.annotate(
                experiment,
                xy=(x_value, y_value),
                xytext=(8, y_offset),
                textcoords="offset points",
                ha="left",
                va=va_align,
                fontsize=7,
                fontweight="bold",
                color="#1A1A1A",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
                zorder=9,
            )


def _highlight_nonaigm_labels(ax):
    """Color non-AIGM method y-tick labels in a muted orange."""
    for label in ax.get_yticklabels():
        if label.get_text() in NON_AIGM_METHODS:
            label.set_color(_NON_AIGM_LABEL_COLOR)
    for label in ax.get_yticklabels():
        if label.get_text() in HYBRID_METHODS:
            label.set_color(_HYBRID_LABEL_COLOR)


def _build_experiment_legend(
    ax, experiment_order, neutral_palette, colored_experiments, top_experiments
):
    """Add an experiment legend inside the bottom-right of the axis."""
    top_set = set(top_experiments)
    handles = []
    for exp in experiment_order:
        color = neutral_palette.get(exp, NEUTRAL_COLOR)
        marker = TOP_EXPERIMENT_MARKERS[0] if exp in top_set else "o"
        ms = 10 if exp in top_set else 7
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                marker=marker,
                linestyle="",
                markersize=ms,
                markeredgecolor="#222222",
                markeredgewidth=0.8,
                label=exp,
            )
        )
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=9,
        frameon=True,
        framealpha=0.92,
        title="Experiments",
        title_fontsize=10,
    )


def plot_best_pointplot(
    df_best,
    method_order,
    experiment_order,
    experiment_palette,
    output_root,
    embedding,
    metric,
    simul_or_real,
):
    # Sort methods by median score, best at top (ascending → last item at top of y-axis)
    method_medians = df_best.groupby("dFC method", observed=True)["score"].median()
    method_order_sorted = (
        method_medians.reindex(method_order).sort_values(ascending=True).index.tolist()
    )

    # Fixed width; height scales with number of methods to avoid pixel-limit errors
    plot_width = 10
    plot_height = max(8, 0.35 * len(method_order_sorted))
    figure, ax = plt.subplots(figsize=(plot_width, plot_height))

    color_threshold = convert_threshold_to_score_scale(COLOR_THRESHOLD, metric)
    label_threshold = convert_threshold_to_score_scale(
        PER_METHOD_LABEL_SCORE_THRESHOLD, metric
    )

    top_experiments = get_top_experiments_by_mean(df_best, TOP_EXPERIMENT_SHAPES)

    # SI policy: color/annotate only star experiments.
    if metric == "SI":
        colored_experiments = set(top_experiments)
        label_threshold = -np.inf
    else:
        colored_experiments = get_colored_experiment_mask(df_best, color_threshold)

    # Create neutral palette: vibrant for high performers, neutral for others
    neutral_palette = create_neutral_palette(
        experiment_order, colored_experiments, experiment_palette
    )

    box_face = to_rgba("#DE9995", 0.18)
    box_edge = "#730800"

    # Horizontal boxplot: score on x-axis, methods on y-axis
    sns.boxplot(
        data=df_best,
        x="score",
        y="dFC method",
        order=method_order_sorted,
        whis=(5, 95),
        fliersize=0,
        linewidth=1.0,
        width=0.5,
        color=box_face,
        ax=ax,
        zorder=1,
    )
    style_boxplot(ax, box_edge)

    lower, upper = get_pointplot_limits(metric)
    overlay_method_means(ax, df_best, lower, upper)

    # Horizontal pointplot: experiments as hue, dodged vertically
    sns.pointplot(
        data=df_best,
        x="score",
        y="dFC method",
        hue="experiment",
        order=method_order_sorted,
        hue_order=experiment_order,
        dodge=0.4,
        errorbar=None,
        linestyles="",
        markers="o",
        palette=neutral_palette,
        ax=ax,
        zorder=6,
    )
    finalize_marker_edges(ax)
    resize_colored_markers(ax, experiment_order, colored_experiments, method_order_sorted)

    point_coordinates = extract_pointplot_coordinates(
        ax,
        method_order_sorted,
        experiment_order,
        neutral_palette,
    )

    overlay_top_experiment_shapes(
        ax,
        df_best,
        point_coordinates,
        neutral_palette,
        top_experiment_shapes=TOP_EXPERIMENT_SHAPES,
    )

    annotate_per_method_quartile(
        ax,
        df_best,
        point_coordinates,
        method_order_sorted,
        colored_experiments=colored_experiments,
        metric=metric,
        simul_or_real=simul_or_real,
        score_threshold=label_threshold,
    )

    ax.set_ylabel("dFC method", fontsize=15, fontweight="bold")
    ax.set_xlabel(metric, fontsize=15, fontweight="bold")
    if metric == "SI":
        ax.set_xlim(right=1.02)
    else:
        ax.set_xlim(0.48, 1.02)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(-0.5, len(method_order_sorted) - 0.5)
    ax.grid(True, axis="x", color="#FFFFFF", alpha=0.85, linewidth=1.1)
    sns.despine(ax=ax, top=True, right=True)
    plt.setp(ax.get_yticklabels(), fontweight="bold", fontsize=13)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    _highlight_nonaigm_labels(ax)

    if ax.legend_:
        ax.legend_.remove()

    figure.tight_layout()

    savefig_pub(
        f"{output_root}/ML_scores_{embedding}_{metric}_{LEVEL}_{simul_or_real}_best.png"
    )
    plt.close(figure)


def plot_lollipop_pointplot(
    df_best,
    method_order,
    experiment_order,
    experiment_palette,
    output_root,
    embedding,
    metric,
    simul_or_real,
):
    method_means = df_best.groupby("dFC method", observed=True)["score"].mean()
    method_order_sorted = (
        method_means.reindex(method_order).sort_values(ascending=True).index.tolist()
    )

    plot_width = 10
    plot_height = max(8, 0.20 * len(method_order_sorted))
    figure, ax = plt.subplots(figsize=(plot_width, plot_height))

    color_threshold = convert_threshold_to_score_scale(COLOR_THRESHOLD, metric)
    top_experiments = get_top_experiments_by_mean(df_best, TOP_EXPERIMENT_SHAPES)

    if metric == "SI":
        colored_experiments = set(top_experiments)
    else:
        colored_experiments = get_colored_experiment_mask(df_best, color_threshold)

    box_edge = "#730800"

    # Lollipop: 5th–95th percentile range line + median dot per method
    method_groups = df_best.groupby("dFC method", observed=True)["score"]
    for i, method in enumerate(method_order_sorted):
        if method not in method_groups.groups:
            continue
        vals = method_groups.get_group(method).dropna().values
        if len(vals) == 0:
            continue
        lo, med, hi = np.nanpercentile(vals, [5, 50, 95])
        ax.hlines(i, lo, hi, colors=box_edge, lw=1.4, alpha=0.45, zorder=1)
        ax.scatter(med, i, color=box_edge, s=28, zorder=2, linewidths=0)

    lower, upper = get_pointplot_limits(metric)

    sns.pointplot(
        data=df_best,
        x="score",
        y="dFC method",
        hue="experiment",
        order=method_order_sorted,
        hue_order=experiment_order,
        dodge=0.35,
        errorbar=None,
        linestyles="",
        markers="o",
        palette=experiment_palette,
        ax=ax,
        zorder=6,
    )
    finalize_marker_edges(ax)
    # Only starred (top) experiments get large markers; all others stay small
    resize_colored_markers(
        ax, experiment_order, set(top_experiments), method_order_sorted
    )

    # Called after pointplot so yticks are populated
    overlay_method_means(ax, df_best, lower, upper, lw=2.8, halfwidth=0.30)

    point_coordinates = extract_pointplot_coordinates(
        ax, method_order_sorted, experiment_order, experiment_palette
    )
    overlay_top_experiment_shapes(
        ax,
        df_best,
        point_coordinates,
        experiment_palette,
        top_experiment_shapes=TOP_EXPERIMENT_SHAPES,
    )

    ax.set_ylabel("dFC method", fontsize=15, fontweight="bold")
    ax.set_xlabel(metric, fontsize=15, fontweight="bold")
    if metric == "SI":
        ax.set_xlim(right=1.02)
    else:
        ax.set_xlim(0.48, 1.02)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(-0.5, len(method_order_sorted) - 0.5)
    ax.grid(True, axis="x", color="#FFFFFF", alpha=0.85, linewidth=1.1)
    sns.despine(ax=ax, top=True, right=True)
    plt.setp(ax.get_yticklabels(), fontweight="bold", fontsize=11)
    plt.setp(ax.get_xticklabels(), fontsize=12)
    _highlight_nonaigm_labels(ax)
    _build_experiment_legend(
        ax, experiment_order, experiment_palette, colored_experiments, top_experiments
    )

    figure.tight_layout()
    savefig_pub(
        f"{output_root}/ML_scores_{embedding}_{metric}_{LEVEL}_{simul_or_real}_best_lollipop.png"
    )
    plt.close(figure)


def plot_best_heatmap(
    df_best,
    method_order,
    task_order,
    task_to_experiment,
    output_root,
    embedding,
    metric,
    simul_or_real,
):
    matrix_best = df_best.pivot(index="task", columns="dFC method", values="score")
    annot_best = df_best.assign(
        label=lambda df_plot: df_plot["score"].map(lambda value: f"{value:.2f}")
    ).pivot(index="task", columns="dFC method", values="label")

    matrix_best, annot_best, _ = relabel_heatmap_rows(
        matrix_best,
        annot_best,
        task_reference_order=task_order,
        task_to_experiment=task_to_experiment,
    )
    col_order = [method for method in method_order if method in matrix_best.columns]

    # Transpose: methods on rows, experiments on columns — keeps width bounded
    matrix_plot = matrix_best.loc[:, col_order].T
    annot_plot = annot_best.loc[:, col_order].T

    n_methods = len(matrix_plot.index)
    n_exps = len(matrix_plot.columns)
    width = max(8.0, 1.2 * n_exps)
    height = max(8.0, 0.30 * n_methods)

    figure, ax = plt.subplots(figsize=(width, height))
    vmin, vmax, center = get_heatmap_limits(metric)
    heatmap = sns.heatmap(
        matrix_plot,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cmap="coolwarm",
        annot=annot_plot,
        fmt="",
        annot_kws={"fontsize": 9, "fontweight": "bold", "linespacing": 1.15},
        cbar_kws={"shrink": 0.7, "pad": 0.02},
        ax=ax,
    )
    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label(metric, fontsize=10, fontweight="bold")
    colorbar.ax.tick_params(labelsize=9)

    boldify_axes(ax, xlabel="Experiment", ylabel="dFC method", rotate_xticks=35)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("dFC method")
    plt.setp(ax.get_xticklabels(), fontweight="bold", rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    savefig_pub(
        f"{output_root}/ML_scores_heatmap_{embedding}_{metric}_{LEVEL}_{simul_or_real}_best.png"
    )
    plt.close(figure)


def build_across_heatmap_data(df_multi, metric, task_order, task_to_experiment):
    summary = (
        df_multi.groupby(["task", "dFC method"], observed=True)[metric]
        .agg(n="count", med="median", vmin="min", vmax="max")
        .reset_index()
    )

    matrix_across = summary.pivot(index="task", columns="dFC method", values="med")
    annot_across = summary.assign(
        label=lambda df_plot: df_plot["vmin"].map(lambda value: f"{value:.2f}")
        + "\u2013"
        + df_plot["vmax"].map(lambda value: f"{value:.2f}")
        + "\n"
        + df_plot["n"].map(lambda value: f"n={value}")
    ).pivot(index="task", columns="dFC method", values="label")

    return relabel_heatmap_rows(
        matrix_across,
        annot_across,
        task_reference_order=task_order,
        task_to_experiment=task_to_experiment,
    )


def plot_across_heatmap(
    df_multi,
    method_order,
    task_order,
    task_to_experiment,
    output_root,
    embedding,
    metric,
    simul_or_real,
):
    if df_multi.empty:
        print(
            f"[ACROSS-RUN] No tasks with ≥2 runs for {embedding} / {metric} — skipping across-run figures."
        )
        return

    matrix_across, annot_across, _ = build_across_heatmap_data(
        df_multi,
        metric,
        task_order,
        task_to_experiment,
    )
    col_order = [method for method in method_order if method in matrix_across.columns]

    # Transpose: methods on rows, experiments on columns — keeps width bounded
    matrix_plot = matrix_across.loc[:, col_order].T
    annot_plot = annot_across.loc[:, col_order].T

    n_methods = len(matrix_plot.index)
    n_exps = len(matrix_plot.columns)
    width = max(8.0, 1.5 * n_exps)
    height = max(8.0, 0.35 * n_methods)

    figure, ax = plt.subplots(figsize=(width, height))
    vmin, vmax, center = get_heatmap_limits(metric)
    heatmap = sns.heatmap(
        matrix_plot,
        vmin=vmin,
        vmax=vmax,
        center=center,
        cmap="coolwarm",
        annot=annot_plot,
        fmt="",
        annot_kws={"fontsize": 9, "fontweight": "bold", "linespacing": 1.15},
        cbar_kws={"shrink": 0.7, "pad": 0.02},
        ax=ax,
    )

    colorbar = heatmap.collections[0].colorbar
    colorbar.set_label(metric, fontsize=10, fontweight="bold")
    boldify_axes(ax, xlabel="Experiment", ylabel="dFC method", rotate_xticks=35)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("dFC method")
    plt.setp(ax.get_xticklabels(), fontweight="bold", rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), fontweight="bold")
    sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()
    savefig_pub(
        f"{output_root}/ML_scores_heatmap_{embedding}_{metric}_{LEVEL}_{simul_or_real}_across.png"
    )
    plt.close(figure)


def plot_aigm_comparison(
    df_best,
    output_root,
    embedding,
    metric,
    simul_or_real,
):
    """
    Horizontal boxplot + scatter comparing AIGM vs non-AIGM method scores.
    One point per (method × experiment). Matches plot_best_pointplot style.
    """
    from scipy.stats import mannwhitneyu

    df_best = df_best.copy()
    df_best["group"] = df_best["dFC method"].apply(
        lambda m: "Non-AIGM" if m in NON_AIGM_METHODS else "Hybrid" if m in HYBRID_METHODS else "AIGM"
    )

    group_order = ["AIGM", "Non-AIGM", "Hybrid"]
    n_aigm = df_best[df_best["group"] == "AIGM"]["dFC method"].nunique()
    n_non_aigm = df_best[df_best["group"] == "Non-AIGM"]["dFC method"].nunique()
    n_hybrid = df_best[df_best["group"] == "Hybrid"]["dFC method"].nunique()
    group_labels = [f"AIGM\n(n={n_aigm} methods)", f"Non-AIGM\n(n={n_non_aigm} methods)", f"Hybrid\n(n={n_hybrid} methods)"]
    df_best["group_label"] = df_best["group"].map(dict(zip(group_order, group_labels)))

    fig, ax = plt.subplots(figsize=(9, 4))

    box_face = to_rgba("#DE9995", 0.18)
    box_edge = "#730800"

    sns.boxplot(
        data=df_best,
        x="score",
        y="group_label",
        order=group_labels,
        whis=(5, 95),
        fliersize=0,
        linewidth=1.0,
        width=0.5,
        color=box_face,
        ax=ax,
        zorder=1,
    )
    style_boxplot(ax, box_edge)

    # One point per (method × experiment), colored by group
    rng = np.random.default_rng(42)
    group_colors = {"AIGM": _AIGM_COLOR, "Non-AIGM": _NON_AIGM_COLOR, "Hybrid": _HYBRID_COLOR}
    for i, (group, label) in enumerate(zip(group_order, group_labels)):
        vals = df_best[df_best["group"] == group]["score"].dropna().values
        y_jit = i + rng.uniform(-0.18, 0.18, len(vals))
        ax.scatter(
            vals,
            y_jit,
            color=group_colors[group],
            alpha=0.35,
            s=18,
            linewidths=0.0,
            zorder=4,
        )

    # Mann-Whitney p-value
    aigm_vals = df_best[df_best["group"] == "AIGM"]["score"].dropna().values
    non_aigm_vals = df_best[df_best["group"] == "Non-AIGM"]["score"].dropna().values
    hybrid_vals = df_best[df_best["group"] == "Hybrid"]["score"].dropna().values
    if len(aigm_vals) >= 2 and len(non_aigm_vals) >= 2: #Ajouter version pour p-value AIGM vs Hybrid
        _, pval = mannwhitneyu(aigm_vals, non_aigm_vals, alternative="two-sided")
        pstr = "p<0.001" if pval < 0.001 else f"p={pval:.3f}"
        ax.text(
            0.97,
            0.97,
            pstr,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#BBBBBB", alpha=0.9),
        )

    lower, upper = get_pointplot_limits(metric)
    if metric == "SI":
        ax.set_xlim(right=1.02)
    else:
        ax.set_xlim(lower, 1.02)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))

    ax.set_xlabel(metric, fontsize=15, fontweight="bold")
    ax.set_ylabel("", fontsize=15)
    ax.set_ylim(-0.5, len(group_order) - 0.5)
    ax.grid(True, axis="x", color="#FFFFFF", alpha=0.85, linewidth=1.1)
    sns.despine(ax=ax, top=True, right=True)
    plt.setp(ax.get_yticklabels(), fontweight="bold", fontsize=13)
    plt.setp(ax.get_xticklabels(), fontsize=12)

    fig.tight_layout()
    savefig_pub(
        f"{output_root}/ML_scores_AIGM_vs_nonAIGM_{embedding}_{metric}_{LEVEL}_{simul_or_real}.png"
    )
    plt.close(fig)


def generate_all_plots(all_ml_scores, tasks_to_include, output_root, simul_or_real):
    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.2})
    sns.set_style("darkgrid")

    for embedding, metric in TARGETS:
        (
            df,
            method_order,
            task_order,
            task_to_experiment,
            experiment_order,
            experiment_palette,
        ) = prepare_metric_dataframe(
            all_ml_scores,
            tasks_to_include,
            embedding,
            metric,
            simul_or_real,
        )
        df_best, df_multi = build_best_and_multi_tables(df, metric)

        plot_best_pointplot(
            df_best,
            method_order,
            experiment_order,
            experiment_palette,
            output_root,
            embedding,
            metric,
            simul_or_real,
        )
        plot_best_heatmap(
            df_best,
            method_order,
            task_order,
            task_to_experiment,
            output_root,
            embedding,
            metric,
            simul_or_real,
        )
        plot_across_heatmap(
            df_multi,
            method_order,
            task_order,
            task_to_experiment,
            output_root,
            embedding,
            metric,
            simul_or_real,
        )
        plot_aigm_comparison(
            df_best,
            output_root,
            embedding,
            metric,
            simul_or_real,
        )
        plot_lollipop_pointplot(
            df_best,
            method_order,
            experiment_order,
            experiment_palette,
            output_root,
            embedding,
            metric,
            simul_or_real,
        )


def main():
    args = parse_args()
    setup_pub_style()

    multi_dataset_info = read_json(args.multi_dataset_info)
    analysis_config = get_analysis_config(multi_dataset_info, args.simul_or_real)
    main_root = analysis_config["main_root"]
    datasets = analysis_config["DATASETS"]
    tasks_to_include = analysis_config["TASKS_to_include"]
    output_root = f"{multi_dataset_info['output_root']}/ML_results"

    all_ml_scores = collect_all_ml_scores(main_root, datasets, tasks_to_include)
    validate_score_lengths(all_ml_scores)
    save_all_ml_scores(all_ml_scores, output_root, args.simul_or_real)
    generate_all_plots(
        all_ml_scores,
        tasks_to_include,
        output_root,
        args.simul_or_real,
    )


if __name__ == "__main__":
    main()
