### NEW: To filter methods in heatmaps by performance threshold ###
# Build a set of dFC methods with performance >= threshold in at least one task.
# If several runs exist for the same (task, dFC method), keep the best run.

import csv
import itertools
import os
from pathlib import Path

import numpy as np

threshold = 0.0  # minimum performance threshold in at least one task=experiment
# and one run for a method to be included

output_root = Path("sample_data") / f"threshold_{int(threshold * 100)}"
output_root.mkdir(parents=True, exist_ok=True)
output_path = output_root / "filtered_methods.npy"

performances = np.load("sample_data/ALL_ML_SCORES_real.npy", allow_pickle=True).item()
# print(performances.keys())

metric = "SVM balanced accuracy"
embedding = "PLS"
group = "test"

tasks = performances["task"]
methods = performances["dFC method"]
runs = performances["run"]

all_embeddings = performances["embedding"]
all_groups = performances["group"]
all_metric_scores = performances[metric]

# Filter for scores that have a specific embedding and group using zip,
# and keep the corresponding task, dFC method, and run labels for each score
filtered_rows = [
    (task, method, run, score)
    for emb, grp, task, method, run, score in zip(
        all_embeddings, all_groups, tasks, methods, runs, all_metric_scores
    )
    if emb == embedding and grp == group
]
# print(filtered_rows)

# If several runs exist for the same (task, dFC method), keep the best run's score.
best_score_by_task_method = {}
for tuple_row in filtered_rows:
    task, method, run, score = tuple_row
    if np.isnan(score):
        continue

    key = (str(task), str(method))  # unique key for each (task, dFC method) combination
    if key not in best_score_by_task_method or score > best_score_by_task_method[key]:
        best_score_by_task_method[key] = float(score)

# Check
test_scores = np.asarray(list(best_score_by_task_method.values()), dtype=float)
print(
    f"Test scores - Min: {np.min(test_scores)}, Max: {np.max(test_scores)}, Mean: {np.mean(test_scores)}"
)


# Keep methods that have at least one task with best-run score >= threshold.
eligible_methods = {
    method
    for (task, method), best_score in best_score_by_task_method.items()
    if best_score >= threshold
}

# Save as a sorted numpy array for easy loading in heatmap scripts.
eligible_methods_sorted = np.array(sorted(eligible_methods), dtype=object)
np.save(output_path, eligible_methods_sorted, allow_pickle=True)

print(f"Metric: {metric}")
print(f"Threshold: {threshold}")
print(f"Eligible methods: {len(eligible_methods_sorted)}")
print(f"Saved to: {output_path}")
print(eligible_methods_sorted)


### Save difference in performance between filtered method pairs for later use in scatterplot colouring ###

# For each filtered method, use its AVERAGE score across tasks after already taking the
# best run for each (task, dFC method) combination above. This creates one scalar
# performance per method, suitable for colouring task-agnostic FS vs AS pair points.
avg_score_by_method = {}
for (task, method), best_score in best_score_by_task_method.items():
    if method not in eligible_methods:
        continue

    if method not in avg_score_by_method:
        avg_score_by_method[method] = []
    avg_score_by_method[method].append(best_score)

# Calculate the average score for each method
for method in avg_score_by_method:
    avg_score_by_method[method] = np.mean(avg_score_by_method[method])


def make_pair_key(method_a, method_b):
    """Stable key matching heatmaps_feature_similarity.py pair keys."""
    return " x ".join(sorted([method_a, method_b]))


pair_performance_difference = {}
pair_rows = []

for method_a, method_b in itertools.combinations(eligible_methods_sorted.tolist(), 2):
    method_a = str(method_a)
    method_b = str(method_b)
    method_a_score = avg_score_by_method[method_a]
    method_b_score = avg_score_by_method[method_b]
    signed_difference = method_a_score - method_b_score
    absolute_difference = abs(signed_difference)
    pair_key = make_pair_key(method_a, method_b)

    pair_performance_difference[pair_key] = absolute_difference
    pair_rows.append(
        {
            "pair_key": pair_key,
            "method_a": method_a,
            "method_b": method_b,
            "method_a_performance": method_a_score,
            "method_b_performance": method_b_score,
            "signed_performance_difference": signed_difference,
            "absolute_performance_difference": absolute_difference,
            "metric": metric,
            "threshold": threshold,
        }
    )

pair_diff_path = output_root / "filtered_performance_differences.npy"
pair_diff_csv_path = pair_diff_path.with_suffix(".csv")

np.save(pair_diff_path, pair_performance_difference, allow_pickle=True)
with open(pair_diff_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
    writer.writeheader()
    writer.writerows(pair_rows)

print(f"Saved pair performance differences to: {pair_diff_path}")
print(f"Saved pair performance differences table to: {pair_diff_csv_path}")
print(f"Method pairs saved: {len(pair_rows)}")
