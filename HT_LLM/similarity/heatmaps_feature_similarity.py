# %%
# Note: if similarity.pkl file is too large (>5 GB), need to request more memory on a compute node via:
# $ salloc --account=def-<supervisor_name> --mem=128G --cpus-per-task=8 --time=4:00:00
# or submit a batch job

import csv

# %%
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

# %%
# Filter methods in heatmaps by performance threshold
threshold = 0.6  # must exist; else, run filter_methods_by_performance.py with the desired threshold first

eligible_methods_path = os.path.join(
    "sample_data", f"threshold_{int(threshold * 100)}", "filtered_methods.npy"
)
eligible_methods = set(
    np.load(eligible_methods_path, allow_pickle=True).astype(str).tolist()
)

print(f"Loaded {len(eligible_methods)} eligible methods")
print("Examples:", sorted(eligible_methods)[:10])

# %%
DEFAULT_OUTPUT_DIR = Path(
    f"HT_LLM/similarity/feature_similarity_results/threshold_{int(threshold * 100)}"
)
OUTPUT_DIR_PDF = DEFAULT_OUTPUT_DIR / "pdf"
OUTPUT_DIR_PNG = DEFAULT_OUTPUT_DIR / "png"
OUTPUT_DIR_PDF.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_PNG.mkdir(parents=True, exist_ok=True)

NON_AIGM_SET = {
    "CAP",
    "Windowless",
    "Clustering",
    "DiscreteHMM",
    "ContinuousHMM",
    "Time-Freq",
    "SlidingWindow",
}
NON_AIGM_COLOR = "darkorange"


# %%
root = "/home/kinichen/scratch/data/pydfc_validator/similarity_assessments_complete"

with open(f"{root}/similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

print(
    "Datasets:", similarity.keys()
)  # layer 1 of hierarchy is datasets, then subjects, sessions, runs, tasks, etc. (pydFC objects)


# %%
dataset_id = "ds003465"
subject_id = "sub-f1027ao"
session_id = "ses-wave1bas"
run_id = "run-2"
task_id = "task-Stroop"

sim_ex = similarity[dataset_id][subject_id][session_id][run_id][task_id]
measures = sim_ex["matrix"]["measure_lst"]
methods = [
    method.MEASURE_NAME for method in measures
]  # extract method names from the pydfc dfc_methods objects
methods = sorted(methods)  # sort alphabetically for consistent ordering
print("Example methods:", methods[:5])

matrix_ex = sim_ex["matrix"]["all"]["spearman"]  # similarity matrix for all methods
print("Example matrix shape:", matrix_ex.shape)


# %%
######### Helper functions to collect and aggregate similarity matrices based on filters
# for various levels (dataset, subject, session, run, task),
# plot heatmaps, and save method pairs for scatterplot #########


def make_pair_key(method_a, method_b):
    """Stable key for joining method-pair outputs across scripts for AS vs FS scatterplot."""
    return " x ".join(sorted([method_a, method_b]))


def collect_similarity_matrices(
    similarity: dict,
    dataset_id=None,
    subject_id=None,
    session_id=None,
    run_id=None,
    task_id=None,
    similarity_key="all",
    metric="spearman",
):
    """
    Collect all similarity matrices matching the specified filters. If a filter is None,
    it matches all values for that level and aggregates over/across it.

    Returns:
        matrices: list of np.ndarray of shape (1, n_methods, n_methods)
    """

    matrices = []

    for ds, ds_data in similarity.items():

        if dataset_id is not None and ds != dataset_id:
            continue

        for sub, sub_data in ds_data.items():

            if subject_id is not None and sub != subject_id:
                continue

            for ses, ses_data in sub_data.items():

                if session_id is not None and ses != session_id:
                    continue

                for run, run_data in ses_data.items():

                    if run_id is not None and run != run_id:
                        continue

                    for task, task_data in run_data.items():

                        if task_id is not None and task != task_id:
                            continue

                        matrices.append(task_data["matrix"][similarity_key][metric])

    return matrices


def aggregate_similarity_matrices(matrices, aggregation="mean"):
    """
    Parameters
    ----------
    matrices : list of arrays
        Each array has shape (1, n_methods, n_methods)

    Returns
    -------
    aggregated_matrix : np.ndarray
        Shape (n_methods, n_methods)

    n_matrices : int
        Number of matrices contributing to the aggregation (sample size)
    """

    if len(matrices) == 0:
        raise ValueError("No matrices found.")

    arr = np.concatenate(matrices, axis=0)

    if aggregation == "mean":
        aggregated = np.mean(arr, axis=0)

    elif aggregation == "median":
        aggregated = np.median(arr, axis=0)

    elif aggregation == "std":
        aggregated = np.std(arr, axis=0)

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return aggregated, len(matrices)


def save_feature_similarity_outputs(
    matrix,
    method_names,
    aggregation_size,
    output_name,
    aggregation="mean",
    similarity_key="all",
    metric="spearman",
):
    """
    Save feature similarity outputs in both matrix and tidy pairwise formats.

    The pairwise CSV is designed to be merged with algorithm similarity outputs
    using pair_key in the AS vs FS scatterplot.
    """

    matrix = np.squeeze(matrix)
    method_names = list(method_names)

    # Note: Saved matrix is in the original methods order, not the reordered version for plotting
    np.save(DEFAULT_OUTPUT_DIR / f"{output_name}_matrix.npy", matrix)
    np.save(
        DEFAULT_OUTPUT_DIR / f"{output_name}_method_names.npy",
        np.array(method_names, dtype=object),
    )
    pairwise_path = DEFAULT_OUTPUT_DIR / f"{output_name}_pairs.csv"

    rows = []

    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            method_a = method_names[i]
            method_b = method_names[j]

            rows.append(
                {
                    "pair_key": make_pair_key(method_a, method_b),
                    "method_a": method_a,
                    "method_b": method_b,
                    "feature_similarity": matrix[i, j],
                    "aggregation": aggregation,
                    "aggregation_size": aggregation_size,
                    "similarity_key": similarity_key,
                    "metric": metric,
                    "output_name": output_name,
                }
            )

    with open(pairwise_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "pair_key",
            "method_a",
            "method_b",
            "feature_similarity",
            "aggregation",
            "aggregation_size",
            "similarity_key",
            "metric",
            "output_name",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_similarity_heatmap(
    matrix,
    aggregation_size=None,
    method_names=methods,
    ordered_method_names=None,
    title="Similarity Heatmap",
    annot=False,
    figsize=(10, 8),
    cluster=False,
    cluster_method="average",
    vmin=None,
    vmax=None,
):
    method_names = list(method_names)
    matrix = np.squeeze(matrix)

    # Use given ordering of methods, i.e., do not do hierarchical clustering
    if ordered_method_names is not None:
        if cluster:
            raise ValueError(
                "When using a given methods order, it doesn't make sense to also do clustering."
            )
        ordered_method_names = list(ordered_method_names)
        order = [method_names.index(name) for name in ordered_method_names]

        # Reorder matrix and labels
        matrix = matrix[np.ix_(order, order)]
        method_names = [method_names[i] for i in order]

    # Highlight non-AIGM names in a different color
    highlight_color = NON_AIGM_COLOR
    highlight_method_names = NON_AIGM_SET
    matrix = np.squeeze(matrix)

    # Optional hierarchical clustering to reorder methods based on similarity to each other
    if cluster:

        # Convert similarity to distance
        distance = (1 - matrix) / 2

        # Ensure exact symmetry and diagonal is 0
        distance = (distance + distance.T) / 2
        np.fill_diagonal(distance, 0)

        # Convert to condensed format required by scipy
        condensed = squareform(distance)

        # Hierarchical clustering
        Z = linkage(condensed, method=cluster_method)

        # Obtain reordered indices
        order = leaves_list(Z)

        # Reorder matrix
        matrix = matrix[np.ix_(order, order)]

        # Reorder labels
        method_names = [method_names[i] for i in order]

    plotted_methods_order = list(method_names)

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        matrix,
        annot=annot,
        xticklabels=method_names,
        yticklabels=method_names,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    if aggregation_size is not None:
        title += f" (n={aggregation_size})"

    plt.title(title, fontsize=12)
    plt.xlabel("dFC Method", fontsize=11)
    plt.ylabel("dFC Method", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(rotation=0, fontsize=6)

    # Highlight selected method labels in a different color.
    for tick_label in ax.get_xticklabels():
        tick_label.set_color(
            highlight_color
            if tick_label.get_text() in highlight_method_names
            else "black"
        )

    for tick_label in ax.get_yticklabels():
        tick_label.set_color(
            highlight_color
            if tick_label.get_text() in highlight_method_names
            else "black"
        )

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR_PDF}/{title}.pdf", bbox_inches="tight")
    plt.savefig(f"{OUTPUT_DIR_PNG}/{title}.png", dpi=600, bbox_inches="tight")
    plt.close()

    return plotted_methods_order


"""
# OUTDATED: Individual heatmap for each experiment/dataset. UPDATED version below
# is a subplot of all experiments/datasets together

# %%
### Average over everything (all subjects, sessions, runs, and datasets) for a specific TASK ###

task_ids = sorted(
    {
        task
        for ds_data in similarity.values()
        for sub_data in ds_data.values()
        for ses_data in sub_data.values()
        for run_data in ses_data.values()
        for task in run_data.keys()
    }
)

for task_id in task_ids:

    matrices = collect_similarity_matrices(similarity, task_id=task_id)

    aggregated, aggregation_size = aggregate_similarity_matrices(
        matrices, aggregation="mean"
    )

    plot_similarity_heatmap(aggregated, aggregation_size, title=f"{task_id}")


# %%
### Average over everything for a specific DATASET ###

dataset_ids = sorted(similarity.keys())

for dataset_id in dataset_ids:

    matrices = collect_similarity_matrices(similarity, dataset_id=dataset_id)

    aggregated, aggregation_size = aggregate_similarity_matrices(
        matrices, aggregation="mean"
    )

    plot_similarity_heatmap(aggregated, aggregation_size, title=f"{dataset_id}")

"""

# %%
### Average over EVERYTHING ###

matrices = collect_similarity_matrices(similarity)

aggregated, aggregation_size = aggregate_similarity_matrices(matrices, aggregation="mean")

# Filter methods based on performance threshold
keep_idx = [i for i, method in enumerate(methods) if method in eligible_methods]
filtered_matrix = aggregated[np.ix_(keep_idx, keep_idx)]
filtered_methods = [methods[i] for i in keep_idx]

filtered_methods_order = plot_similarity_heatmap(
    matrix=filtered_matrix,
    aggregation_size=aggregation_size,
    method_names=filtered_methods,
    title="Mean dFC feature similarity between methods",
    cluster=True,
    vmin=-0.2,
    vmax=1.0,
)

save_feature_similarity_outputs(
    matrix=filtered_matrix,
    method_names=filtered_methods,  # not reordered since aggregated did not go through hierarchical clustering in plotting function
    aggregation_size=aggregation_size,
    output_name="FS",
    aggregation="mean",
    similarity_key="all",
    metric="spearman",
)


# %%
### Standard deviation over EVERYTHING ###

# Measures which method pairs are more stable vs. more variable across filters

matrices = collect_similarity_matrices(similarity)

aggregated, aggregation_size = aggregate_similarity_matrices(matrices, aggregation="std")

# Filter methods based on performance threshold
keep_idx = [i for i, method in enumerate(methods) if method in eligible_methods]
filtered_std_matrix = aggregated[np.ix_(keep_idx, keep_idx)]

plot_similarity_heatmap(
    filtered_std_matrix,
    aggregation_size,
    method_names=filtered_methods,
    ordered_method_names=filtered_methods_order,
    title="Standard deviation of dFC feature similarity between methods",
)


# %%
### 3 x 3 subplot heatmap: mean similarity for each TASK ###


def plot_task_similarity_heatmap_grid(
    similarity,
    task_ids,
    ordered_method_names,
    original_method_names=methods,
    aggregation="mean",
    title="Task-Specific Mean dFC Feature Similarity",
    figsize=(14, 12),
    nrows=3,
    ncols=3,
    tick_fontsize=3,
    title_fontsize=12,
    vmin=None,
    vmax=None,
):
    """Plot one heatmap per task using a shared method ordering and colorbar."""

    ordered_method_names = list(ordered_method_names)
    original_method_names = list(original_method_names)

    # Filter methods based on performance threshold
    keep_idx = [
        i for i, method in enumerate(original_method_names) if method in eligible_methods
    ]
    original_method_names = [original_method_names[i] for i in keep_idx]

    # Reorder based on same methods order from average heatmap
    order = [original_method_names.index(name) for name in ordered_method_names]

    if len(task_ids) > nrows * ncols:
        raise ValueError(f"Expected at most {nrows * ncols} tasks, got {len(task_ids)}.")

    highlight_color = NON_AIGM_COLOR
    highlight_method_names = NON_AIGM_SET

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=False)
    axes = np.ravel(axes)

    # Dedicated colorbar axis placed off to the right of the subplot grid.
    cbar_ax = fig.add_axes([0.92, 0.18, 0.02, 0.64])

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(task_ids):
            ax.axis("off")
            continue

        task_id = task_ids[ax_idx]
        matrices = collect_similarity_matrices(similarity, task_id=task_id)
        aggregated, aggregation_size = aggregate_similarity_matrices(
            matrices,
            aggregation=aggregation,
        )

        matrix = np.squeeze(aggregated)
        matrix = matrix[
            np.ix_(keep_idx, keep_idx)
        ]  # Filter methods based on performance threshold first
        matrix = matrix[
            np.ix_(order, order)
        ]  # Reorder based on same methods order from average heatmap

        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=ordered_method_names,
            yticklabels=ordered_method_names,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=ax_idx == 0,
            cbar_ax=cbar_ax if ax_idx == 0 else None,
            square=True,
        )

        ax.set_title(f"{task_id} (n={aggregation_size})", fontsize=title_fontsize)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelrotation=45, labelsize=tick_fontsize)
        ax.tick_params(axis="y", labelrotation=0, labelsize=tick_fontsize)

        for tick_label in ax.get_xticklabels():
            tick_label.set_horizontalalignment("right")
            tick_label.set_color(
                highlight_color
                if tick_label.get_text() in highlight_method_names
                else "black"
            )

        for tick_label in ax.get_yticklabels():
            tick_label.set_color(
                highlight_color
                if tick_label.get_text() in highlight_method_names
                else "black"
            )

    fig.suptitle(title, fontsize=14, y=0.98)
    # fig.supxlabel("dFC Method", fontsize=11, y=0.01)
    # fig.supylabel("dFC Method", fontsize=11, x=0.01)
    fig.subplots_adjust(
        left=0.07,
        right=0.90,
        bottom=0.08,
        top=0.93,
        wspace=0.25,
        hspace=0.35,
    )

    fig.savefig(f"{OUTPUT_DIR_PDF}/{title}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUTPUT_DIR_PNG}/{title}.png", dpi=600, bbox_inches="tight")
    plt.close()


# Make task- (aka experiment-) specific subplot

task_ids = sorted(
    {
        task
        for ds_data in similarity.values()
        for sub_data in ds_data.values()
        for ses_data in sub_data.values()
        for run_data in ses_data.values()
        for task in run_data.keys()
    }
)

plot_task_similarity_heatmap_grid(
    similarity=similarity,
    task_ids=task_ids,
    ordered_method_names=filtered_methods_order,
    vmin=-0.2,
    vmax=1.0,
)


# %%
print(f"Complete! Figures saved to {DEFAULT_OUTPUT_DIR}")
