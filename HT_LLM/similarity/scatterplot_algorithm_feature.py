### Scatter plot of algorithm similarity vs. feature similarity, coloured by difference in performance, between method pairs ###

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.lines import Line2D

threshold = 0.0  # minimum performance threshold for a method to be included
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
# upstream files are already filtered for performance using `threshold`
# Note, however, that upstream files are filtered for at least 1 experiment/task (more lenient), whereas
# this threshold is applied to the average performance across all experiments/tasks for each method.
high_performance_threshold = 0.60

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


##### Plotting #####
fig, ax = plt.subplots(figsize=(14, 8))

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

# Add a linear trendline and report the model fit between the two plotted axes.
# i.e., assumes {feature_similarity} = beta_0 + beta_1 * {algorithm_similarity} + epsilon
trend_df = scatter_df[["algorithm_similarity", "feature_similarity"]].dropna()
if len(trend_df) >= 2:
    trend_x = trend_df["algorithm_similarity"].to_numpy(dtype=float)
    trend_y = trend_df["feature_similarity"].to_numpy(dtype=float)

    trend_model = sm.OLS(trend_y, sm.add_constant(trend_x, has_constant="add")).fit()

    # Plot trendline using the fitted model
    trendline_x = np.linspace(trend_x.min(), trend_x.max(), 100)
    trendline_y = trend_model.predict(sm.add_constant(trendline_x, has_constant="add"))
    ax.plot(
        trendline_x,
        trendline_y,
        color="darkgreen",
        linestyle="-",
        linewidth=1.5,
        alpha=0.8,
        label="_nolegend_",
    )

    # R^2 = proportion of variance in y explained by a linear relationship with x
    # p-value = probability of observing a slope as extreme as the fitted slope if \
    # the null hypothesis (slope = 0) is true (i.e., prob. of observing by chance)
    # Note: pvalues[1] is the slope (beta_1) p-value. Tests H_0: beta_1 = 0 vs. H_A: beta_1 != 0
    slope_p_value = trend_model.pvalues[1]

    ax.text(
        0.98,
        0.03,
        f"$R^2$ = {trend_model.rsquared:.3f}\nOLS slope $p$ = {slope_p_value:.3g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="darkgreen",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "grey",
            "alpha": 0.85,
        },
    )
    print(f"OLS slope p-value: {slope_p_value}")

# Label certain points by their method pair name for identification
high_as_threshold = 0.7  # For vertical line; Only label points with algorithm similarity above this threshold
high_fs_threshold = 0.7  # For horizontal line
labelled_pairs = (
    scatter_df.loc[scatter_df["algorithm_similarity"] > high_as_threshold, "pair_key"]
    .dropna()
    .astype(str)
    .tolist()
)
labelled_methods = sorted(
    {method for pair_key in labelled_pairs for method in str(pair_key).split(" x ")}
)
method_id_by_name = {
    method: method_id for method_id, method in enumerate(labelled_methods, start=1)
}


def encode_pair_key(pair_key):
    """Replace method names in a pair key with their numeric method IDs to avoid overlapping labels in plot."""
    return " x ".join(
        str(method_id_by_name[method]) for method in str(pair_key).split(" x ")
    )


for idx, row in scatter_df.iterrows():
    # Only label points satisfying this condition
    if row["algorithm_similarity"] > high_as_threshold:
        ax.text(
            row["algorithm_similarity"] + 0.005,  # Add a slight x-offset manually
            row["feature_similarity"] + 0.005,  # Add a slight y-offset manually
            encode_pair_key(row["pair_key"]),  # Label using compact numeric method IDs
            color=performance_pair_palette[row["performance_pair_type"]],
            weight="bold",
            size=7,
        )

# Sanity checks for correct number of method pairs (not missing any)
print("Feature pairs:", len(feature_pairs))
print("Algorithm pairs:", len(algorithm_pairs))
print("Merged pairs:", len(scatter_df))

ax.set_xlim(0, 1)
ax.set_ylim(-0.2, 1)
ax.axvline(x=high_as_threshold, color="grey", linestyle="--")
ax.axhline(y=high_fs_threshold, color="grey", linestyle="--")

# Performance legend
legend_handles, legend_labels = ax.get_legend_handles_labels()
legend_labels = [
    "performance_difference" if label == "absolute_performance_difference" else label
    for label in legend_labels
]
size_title_idx = legend_labels.index("performance_difference")
for label_idx in range(size_title_idx + 1, len(legend_labels)):
    try:
        legend_labels[label_idx] = f"{float(legend_labels[label_idx]):.0%}"
    except ValueError:
        pass
legend_handles.insert(size_title_idx, Line2D([], [], linestyle="none"))
legend_labels.insert(size_title_idx, "")

perf_legend = ax.legend(
    title="SVM Balanced Accuracy\n",
    handles=legend_handles,
    labels=legend_labels,
    loc="lower left",
    bbox_to_anchor=(1.01, 0),
    fontsize=8,
    title_fontsize=9,
)
# perf_legend.get_title().set_fontweight("bold")
for legend_text in perf_legend.get_texts():
    if legend_text.get_text() in [
        "performance_pair_type",
        "performance_difference",
    ]:
        legend_text.set_fontweight("bold")

ax.add_artist(
    perf_legend
)  # Re-add the first legend back to the axes (don't overwrite it)


# Method IDs legend
if method_id_by_name:
    method_handles = [Line2D([], [], linestyle="none") for _ in method_id_by_name]
    method_labels = [
        f"{method_id}: {method}" for method, method_id in method_id_by_name.items()
    ]
    method_legend = ax.legend(
        title="Labelled Methods",
        handles=method_handles,
        labels=method_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        fontsize=8,
        title_fontsize=9,
        handlelength=0,
        handletextpad=0,
    )
    # method_legend.get_title().set_fontweight("bold")

ax.minorticks_on()
ax.set_xlabel("Algorithm similarity")
ax.set_ylabel("dFC feature similarity")
ax.set_title(
    "Algorithm Similarity vs. dFC Feature Similarity vs. Performance Difference for Method Pairs",
    fontsize=12,
    y=1.02,
)

# Make the full figure wider and reserve the extra width for legends on the right.
# The scatterplot axes stay about as wide as before; the added figure width becomes
# the outside legend area.
fig.subplots_adjust(right=0.75)

plt.savefig(
    OUTPUT_DIR / "algorithm_vs_feature_similarity_scatter.png",
    dpi=600,
    bbox_inches="tight",
)
plt.savefig(
    OUTPUT_DIR / "algorithm_vs_feature_similarity_scatter.pdf", bbox_inches="tight"
)
plt.show()
