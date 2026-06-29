"""
Algorithm Similarity (AS): Bag of Operations (BOO) with operation counts.

Idea
----
Parse each dFC method's .py file into an Abstract Syntax Tree (AST), extract every
function/class call that touches a known numerical/scientific library (numpy, scipy,
sklearn, hmmlearn, statsmodels, ...), resolve it to a fully-qualified name
(e.g. "np.corrcoef" -> "numpy.corrcoef"), and represent the method as a BAG
(multiset) of operations.

Unlike a plain set-based Jaccard score, this implementation keeps the number of
times each operation appears. Pairwise AS is the WEIGHTED Jaccard similarity:

    sum(min(count_a[op], count_b[op])) / sum(max(count_a[op], count_b[op]))

This metric asks: "how similar are the methods' scripts in both the operations they use and how
often they use them?"

Usage
-----
python BOO_algorithm_similarity.py /path/to/dfc_methods/*.py
"""

import ast
import csv
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

# Only calls whose resolved root module starts with one of these are kept as
# "operations". Everything else (self.*, local helper functions, plain
# built-ins like zip/len/range) is treated as implementation glue and dropped.
LIBRARY_PREFIXES = (
    "numpy",
    "scipy",
    "sklearn",
    "hmmlearn",
    "statsmodels",
    "ksvd",
    "pycwt",
)

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

DEFAULT_OUTPUT_DIR = "HT_LLM/similarity/algorithm_similarity_results"
METRIC_NAME = "BOO_weighted_jaccard"
EXCLUDED_METHOD_FILES = {"__init__.py", "base_dfc_method.py"}


def _build_import_map(tree):
    """Map local alias -> fully-qualified module/object path, from this file's imports."""
    import_map = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]
                import_map[local_name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for alias in node.names:
                local_name = alias.asname or alias.name
                import_map[local_name] = f"{node.module}.{alias.name}"
    return import_map


def _dotted_name(node):
    """Best-effort reconstruction of a dotted attribute chain, e.g. Attribute(Attribute(Name)) -> 'a.b.c'."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None  # call target is something we can't statically resolve (e.g. a subscript)


def _resolve_call_name(raw_name, import_map):
    """Resolve an imported alias in a call name to its fully-qualified name."""
    head, *rest = raw_name.split(".")
    if head in import_map:
        return ".".join([import_map[head]] + rest)
    return raw_name


def _is_library_operation(resolved_name):
    """Return True when a resolved call belongs to one of the tracked libraries."""
    return resolved_name.split(".")[0] in LIBRARY_PREFIXES


def extract_operation_counts(filepath, verbose=False):
    """Return Counter({operation_name: count}) for resolved library operations."""
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    import_map = _build_import_map(tree)

    operation_counts = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            raw = func.id
        elif isinstance(func, ast.Attribute):
            raw = _dotted_name(func)
        else:
            continue
        if raw is None:
            continue

        resolved = _resolve_call_name(raw, import_map)
        if _is_library_operation(resolved):
            operation_counts[resolved] += 1
        # else: skip self.*, locally-defined helpers, and plain built-ins

    if verbose:
        total_calls = sum(operation_counts.values())
        print(
            f"\n{filepath} -> {len(operation_counts)} distinct operations, "
            f"{total_calls} counted calls:"
        )
        for op, count in sorted(operation_counts.items()):
            print(f"    {op}: {count}")
    return operation_counts


def extract_operations(filepath, verbose=False):
    """Return the set of distinct operations. Kept for backward compatibility."""
    return set(extract_operation_counts(filepath, verbose=verbose))


def weighted_jaccard_similarity(counts_a, counts_b):
    """Compute weighted Jaccard similarity between two operation-count bags."""
    operations = set(counts_a) | set(counts_b)  # complete set of all operations

    overlap = sum(min(counts_a[op], counts_b[op]) for op in operations)
    union = sum(max(counts_a[op], counts_b[op]) for op in operations)

    if not operations:  # neither script captured any operations from tracked libraries
        similarity = 0.0
        
    else:
        similarity = overlap / union

    return overlap, union, similarity


def _to_readable_method_name(filepath):
    """Read the method display name from the script's class-level MEASURE_NAME."""
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(filepath))

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        for statement in node.body:
            if isinstance(statement, ast.Assign):
                targets = statement.targets
                value = statement.value
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
                value = statement.value
            else:
                continue

            has_measure_name = any(
                isinstance(target, ast.Name) and target.id == "MEASURE_NAME"
                for target in targets
            )

            if has_measure_name:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
                raise ValueError(
                    f"MEASURE_NAME in {filepath} must be a string literal to be used as a label."
                )

    raise ValueError(f"No class-level MEASURE_NAME string found in {filepath}.")


def _make_unique_labels(filepaths):
    """Generate unique labels from each method script's MEASURE_NAME."""
    counts = {}
    labels = []
    for filepath in filepaths:
        base = _to_readable_method_name(filepath)
        counts[base] = counts.get(base, 0) + 1
        labels.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return labels

def make_pair_key(method_a, method_b):
    """Stable key for joining method-pair outputs across scripts for AS vs FS scatterplot."""
    return "+".join(sorted([method_a, method_b]))


def _hierarchical_cluster_order(matrix, cluster_method="average"):
    """Return indices that order similar methods next to each other."""

    if matrix.shape[0] < 2:
        return np.arange(matrix.shape[0])

    distance = (1 - matrix) / 2
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    condensed = squareform(distance)

    linkage_matrix = linkage(condensed, method=cluster_method)
    return leaves_list(linkage_matrix)


def plot_similarity_heatmap(
    matrix,
    labels,
    title="Algorithm Similarity (Bag of Operations with Weighted Jaccard)",
    annot=False,
    figsize=(10, 8),
    cluster=True,
    cluster_method="average",
):
    """Return a heatmap figure for a saved AS matrix.

    Note: The ordering is controlled by this script's own hierarchical clustering,
    while the visual formatting mirrors the feature similarity's heatmap style.
    """
    labels = list(labels)
    matrix = np.squeeze(matrix)

    highlight_color = NON_AIGM_COLOR
    highlight_method_names = NON_AIGM_SET

    if cluster:
        order = _hierarchical_cluster_order(matrix, cluster_method=cluster_method)
        matrix = matrix[np.ix_(order, order)]
        labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=annot,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        ax=ax,
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("dFC Method", fontsize=11)
    ax.set_ylabel("dFC Method", fontsize=11)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, rotation=0, fontsize=6)

    # Highlight selected method labels in orange, leave all other labels black.
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
    return fig, ax


def save_similarity_outputs(output_dir, labels, matrix, table):
    """Save the matrix, ordered labels, pairwise table, and heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"AS_{METRIC_NAME}_matrix.npy", matrix)
    np.save(
        output_dir / f"AS_{METRIC_NAME}_method_names.npy", np.array(labels, dtype=object)
    )

    with open(
        output_dir / f"AS_{METRIC_NAME}_pairs.csv", "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "pair_key",
            "method_a",
            "method_b",
            "source_a",
            "source_b",
            "algorithm_similarity",
            "weighted_overlap",
            "weighted_union",
            "n_shared_distinct",
            "n_distinct_ops_a",
            "n_distinct_ops_b",
            "n_total_ops_a",
            "n_total_ops_b",
            "shared_operation_counts",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)

    try:
        fig, _ = plot_similarity_heatmap(matrix, labels, cluster=True)
    except ImportError:
        print(
            "Skipping heatmap export because matplotlib is not available in this environment."
        )
    else:
        fig.savefig(
            str(output_dir / f"AS_{METRIC_NAME}_heatmap.png"),
            dpi=600,
            bbox_inches="tight",
        )
        fig.savefig(
            str(output_dir / f"AS_{METRIC_NAME}_heatmap.pdf"),
            bbox_inches="tight",
        )

        plt.close(fig)


def load_similarity_outputs(output_dir):
    """Load the saved AS matrix, ordered method names/labels, and searchable pairwise table."""
    output_dir = Path(output_dir)
    matrix = np.load(output_dir / f"AS_{METRIC_NAME}_matrix.npy")
    labels = np.load(
        output_dir / f"AS_{METRIC_NAME}_method_names.npy", allow_pickle=True
    ).tolist()
    with open(
        output_dir / f"AS_{METRIC_NAME}_pairs.csv", "r", newline="", encoding="utf-8"
    ) as f:
        table = list(csv.DictReader(f))
    return labels, matrix, table


def main(filepaths):
    source_paths = [
        str(Path(fp)) for fp in filepaths if Path(fp).name not in EXCLUDED_METHOD_FILES
    ]

    if not source_paths:
        raise ValueError(
            "No concrete dFC method files provided. Pass method scripts such as "
            "pydfc/dfc_methods/*.py; base_dfc_method.py and __init__.py are skipped."
        )

    labels = _make_unique_labels(source_paths)
    operation_bags = {}
    for label, path in zip(labels, source_paths):
        operation_bags[label] = extract_operation_counts(path, verbose=True)

    print("\nPairwise Algorithm Similarity (weighted Jaccard over operation counts):")
    names = list(operation_bags.keys())

    # Initialize with zeros so the main diagonal stays 0.0 for simple visualization.
    alg_sim = np.zeros((len(names), len(names)), dtype=float)

    pairwise_rows = []

    for i, j in itertools.combinations(range(len(names)), 2):  # only off diagonal pairs
        method_a = names[i]
        method_b = names[j]
        counts_a = operation_bags[method_a]
        counts_b = operation_bags[method_b]
        weighted_overlap, weighted_union, similarity = weighted_jaccard_similarity(
            counts_a, counts_b
        )
        alg_sim[i, j] = similarity
        alg_sim[j, i] = similarity

        shared = sorted(set(counts_a) & set(counts_b))
        shared_counts = {
            op: {"method_a": counts_a[op], "method_b": counts_b[op]} for op in shared
        }
        pairwise_rows.append(
            {
                "pair_key": make_pair_key(method_a, method_b),
                "method_a": method_a,
                "method_b": method_b,
                "source_a": source_paths[i],
                "source_b": source_paths[j],
                "algorithm_similarity": similarity,
                "weighted_overlap": weighted_overlap,
                "weighted_union": weighted_union,
                "n_shared_distinct": len(shared),
                "n_distinct_ops_a": len(counts_a),
                "n_distinct_ops_b": len(counts_b),
                "n_total_ops_a": sum(counts_a.values()),
                "n_total_ops_b": sum(counts_b.values()),
                "shared_operation_counts": json.dumps(shared_counts, sort_keys=True),
            }
        )

        print(
            f"{method_a:35s} vs {method_b:35s}  AS = {similarity:.3f}   "
            f"overlap/union = {weighted_overlap}/{weighted_union}   "
            f"shared = {shared}"
        )

    save_similarity_outputs(
        DEFAULT_OUTPUT_DIR,
        names,
        alg_sim,
        pairwise_rows,
    )
    print(f"Saved outputs to {DEFAULT_OUTPUT_DIR}/")


if __name__ == "__main__":
    main(sys.argv[1:])
