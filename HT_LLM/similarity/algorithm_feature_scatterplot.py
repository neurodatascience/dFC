### Scatter plot of algorithm similarity vs. feature similarity, coloured by difference in performance, between method pairs ###

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

threshold = 0.6  # minimum performance threshold for a method to be included
print(f"Method Filtering Threshold: {threshold}")
OUTPUT_DIR = (
    Path("HT_LLM/similarity/scatterplot_algorithm_feature_results")
    / f"threshold_{int(threshold * 100)}"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

algorithm_pairs = pd.read_csv(
    f"HT_LLM/similarity/algorithm_similarity_results/threshold_{int(threshold * 100)}/AS_BOO_weighted_jaccard_pairs.csv"
)

feature_pairs = pd.read_csv(
    f"HT_LLM/similarity/feature_similarity_results/threshold_{int(threshold * 100)}/FS_pairs.csv"
)

scatter_df = feature_pairs.merge(
    algorithm_pairs[["pair_key", "algorithm_similarity"]],
    on="pair_key",
    how="inner",
)

### Colour by difference in performance between the two methods in each pair
# Note: only methods that passed the performance threshold were saved and loaded above,
# so performance difference is only coloured for those methods automatically.
performance_diff_table_path = (
    Path("sample_data")
    / f"threshold_{int(threshold * 100)}"
    / "filtered_performance_differences.csv"
)

pair_performance_table = pd.read_csv(performance_diff_table_path)

scatter_df = scatter_df.merge(
    pair_performance_table[
        [
            "pair_key",
            "method_a_performance",
            "method_b_performance",
            "absolute_performance_difference",
        ]
    ],
    on="pair_key",
    how="left",
)

# Use a separate cutoff for categorizing whether each method in a pair is high-performing.
# If this equals `threshold`, most pairs may be classified as high-high because the
# upstream files are already filtered using `threshold`
# Note, however, that upstream files are filtered for at least 1 experiment/task (more lenient), whereas
# this threshold is applied to the average performance across all experiments/tasks for each method.
high_performance_threshold = threshold

scatter_df["performance_pair_type"] = np.select(
    [
        (scatter_df["method_a_performance"] >= high_performance_threshold)
        & (scatter_df["method_b_performance"] >= high_performance_threshold),
        (scatter_df["method_a_performance"] >= high_performance_threshold)
        | (scatter_df["method_b_performance"] >= high_performance_threshold),
    ],
    [
        "both high performance",
        "mixed high/low performance",
    ],
    default="both low performance",
)

print(
    "Pairs missing performance info:",
    scatter_df["absolute_performance_difference"].isna().sum(),
)


### Plot
fig, ax = plt.subplots(figsize=(12, 8))

performance_pair_palette = {
    "both high performance": "#D0021B",
    "mixed high/low performance": "#8E9AA8",
    "both low performance": "#4A90E2",
}

sns.scatterplot(
    data=scatter_df,
    x="algorithm_similarity",
    y="feature_similarity",
    hue="performance_pair_type",
    size="absolute_performance_difference",
    sizes=(30, 180),
    palette=performance_pair_palette,
    alpha=0.75,
    ax=ax,
)

# Label certain points by their method pair name for identification
high_as_threshold = 0.6  # For vertical line; Only label points with algorithm similarity above this threshold
high_fs_threshold = 0.6  # For horizontal line
counter = 0
for idx, row in scatter_df.iterrows():
    # Only label points satisfying this condition
    if row["algorithm_similarity"] > high_as_threshold:
        ax.text(
            row["algorithm_similarity"] + 0.005,  # Add a slight x-offset manually
            row["feature_similarity"] + 0.005,  # Add a slight y-offset manually
            row["pair_key"],  # The labelled text is the method pair's name
            color=performance_pair_palette[row["performance_pair_type"]],
            weight="bold",
            size=7,
        )
    counter += 1

# Sanity checks for correct number of method pairs (not missing any)
print("Feature pairs:", len(feature_pairs))
print("Algorithm pairs:", len(algorithm_pairs))
print("Merged pairs:", len(scatter_df))
print("Plotted points:", counter)

ax.set_xlim(0, 1)
ax.set_ylim(-0.2, 1)
ax.axvline(x=high_as_threshold, color="grey", linestyle="--")
ax.axhline(y=high_fs_threshold, color="grey", linestyle="--")

legend_handles, legend_labels = ax.get_legend_handles_labels()
size_title_idx = legend_labels.index("absolute_performance_difference")
for label_idx in range(size_title_idx + 1, len(legend_labels)):
    try:
        legend_labels[label_idx] = f"{float(legend_labels[label_idx]):.0%}"
    except ValueError:
        pass
legend_handles.insert(size_title_idx, Line2D([], [], linestyle="none"))
legend_labels.insert(size_title_idx, "")

legend = ax.legend(
    title="SVM Balanced Accuracy Performance\n",
    handles=legend_handles,
    labels=legend_labels,
    loc="lower left",
    bbox_to_anchor=(1.02, 0),
    fontsize=8,
    title_fontsize=9,
)
# legend.get_title().set_fontweight("bold")
for legend_text in legend.get_texts():
    if legend_text.get_text() in [
        "performance_pair_type",
        "absolute_performance_difference",
    ]:
        legend_text.set_fontweight("bold")

ax.minorticks_on()
ax.set_xlabel("Algorithm similarity")
ax.set_ylabel("dFC feature similarity")
ax.set_title(
    "Algorithm Similarity vs. dFC Feature Similarity vs. Performance Difference for Method Pairs",
    fontsize=12,
    y=1.02,
)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "algorithm_vs_feature_similarity_scatter.png",
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    OUTPUT_DIR / "algorithm_vs_feature_similarity_scatter.pdf", bbox_inches="tight"
)
plt.show()
