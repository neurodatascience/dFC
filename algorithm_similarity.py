"""
Algorithm Similarity (AS) metric, v1: "bag of mathematical operations" + Jaccard similarity.

Idea
----
Parse each dFC method's .py file into an Abstract Syntax Tree (AST), extract every
function/class call that touches a known numerical/scientific library (numpy, scipy,
sklearn, hmmlearn, statsmodels, joblib, ...), resolve it to a fully-qualified name
(e.g. "np.corrcoef" -> "numpy.corrcoef"), and represent the method as the SET of
distinct operations it uses. Pairwise AS = Jaccard similarity between two methods'
operation sets. This is invariant to variable names, comments, helper-function
decomposition, and loop-vs-vectorized style -- it only asks "which math/stats
primitives does this method call at all".

Usage
-----
python algorithm_similarity.py file1.py file2.py file3.py ...
OR
python algorithm_similarity.py /path/to/dfc_methods/*.py
"""

import ast
import csv
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

# Only calls whose resolved root module starts with one of these are kept as
# "operations". Everything else (self.*, local helper functions, plain
# built-ins like zip/len/range) is treated as implementation glue and dropped.
LIBRARY_PREFIXES = ("numpy", "scipy", "sklearn", "hmmlearn", "statsmodels", "joblib")


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


def extract_operations(filepath, verbose=False):
    """Return the set of distinct resolved library operations called in a method's file."""
    with open(filepath, "r") as f:
        source = f.read()
    tree = ast.parse(source)
    import_map = _build_import_map(tree)

    ops = set()
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

        head, *rest = raw.split(".")
        if head in import_map:
            resolved = ".".join([import_map[head]] + rest)
        else:
            resolved = raw

        if resolved.split(".")[0] in LIBRARY_PREFIXES:
            ops.add(resolved)
        # else: skip self.*, locally-defined helpers, and plain built-ins

    if verbose:
        print(f"\n{filepath} -> {len(ops)} operations:")
        for op in sorted(ops):
            print(f"    {op}")
    return ops


def jaccard_similarity(set_a, set_b):
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def _to_readable_method_name(filepath):
    """Convert a filepath into a readable CamelCase label."""
    stem = Path(filepath).stem
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", stem) if part]
    if not parts:
        return stem
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _make_unique_labels(filepaths):
    """Generate readable, unique labels in input order."""
    counts = {}
    labels = []
    for filepath in filepaths:
        base = _to_readable_method_name(filepath)
        counts[base] = counts.get(base, 0) + 1
        labels.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return labels


def _hierarchical_cluster_order(matrix, cluster_method="average"):
    """Return indices that order similar methods next to each other."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

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
    title="Algorithm Similarity (Jaccard)",
    cluster=True,
    cluster_method="average",
):
    """Return a heatmap figure for a saved AS matrix."""
    import matplotlib.pyplot as plt

    if cluster:
        order = _hierarchical_cluster_order(matrix, cluster_method=cluster_method)
        matrix = matrix[np.ix_(order, order)]
        labels = [labels[i] for i in order]

    fig, ax = plt.subplots(
        figsize=(max(8, 0.45 * len(labels)), max(6, 0.45 * len(labels)))
    )
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    fig.colorbar(image, ax=ax, label="AS")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("Method")
    ax.set_ylabel("Method")
    ax.tick_params(axis="y", labelrotation=0)
    plt.tight_layout()
    return fig, ax


def save_similarity_outputs(output_dir, labels, source_paths, matrix, table):
    """Persist the matrix, ordered labels, pairwise table, and heatmap to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "AS_jaccard.npy", matrix)
    np.save(output_dir / "AS_jaccard_names.npy", np.array(labels, dtype=object))
    with open(output_dir / "AS_jaccard_names.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    with open(output_dir / "AS_jaccard_source_paths.json", "w", encoding="utf-8") as f:
        json.dump(source_paths, f, indent=2)

    with open(
        output_dir / "AS_jaccard_pairs.csv", "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "method_a",
            "method_b",
            "source_a",
            "source_b",
            "similarity",
            "n_shared",
            "n_ops_a",
            "n_ops_b",
            "shared_operations",
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
            str(output_dir / "AS_jaccard_heatmap.png"), dpi=200, bbox_inches="tight"
        )
        import matplotlib.pyplot as plt

        plt.close(fig)


def load_similarity_outputs(output_dir):
    """Load the saved AS matrix, ordered labels, and searchable pairwise table."""
    output_dir = Path(output_dir)
    matrix = np.load(output_dir / "AS_jaccard.npy")
    labels = np.load(output_dir / "AS_jaccard_names.npy", allow_pickle=True).tolist()
    with open(output_dir / "AS_jaccard_source_paths.json", "r", encoding="utf-8") as f:
        source_paths = json.load(f)
    with open(
        output_dir / "AS_jaccard_pairs.csv", "r", newline="", encoding="utf-8"
    ) as f:
        table = list(csv.DictReader(f))
    return labels, source_paths, matrix, table


def main(filepaths):
    source_paths = [str(Path(fp)) for fp in filepaths if Path(fp).name != "__init__.py"]

    labels = _make_unique_labels(source_paths)
    op_sets = {}
    for label, path in zip(labels, source_paths):
        op_sets[label] = extract_operations(path, verbose=True)

    print("\nPairwise Algorithm Similarity (Jaccard over operation sets):")
    names = list(op_sets.keys())

    alg_sim = np.eye(len(names), dtype=float)
    pairwise_rows = []

    for i, j in itertools.combinations(range(len(names)), 2):
        method_a = names[i]
        method_b = names[j]
        sim = jaccard_similarity(op_sets[method_a], op_sets[method_b])
        alg_sim[i, j] = sim
        alg_sim[j, i] = sim

        shared = sorted(op_sets[method_a] & op_sets[method_b])
        pairwise_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "source_a": source_paths[i],
                "source_b": source_paths[j],
                "similarity": sim,
                "n_shared": len(shared),
                "n_ops_a": len(op_sets[method_a]),
                "n_ops_b": len(op_sets[method_b]),
                "shared_operations": "; ".join(shared),
            }
        )

        print(f"{method_a:35s} vs {method_b:35s}  AS = {sim:.3f}   shared = {shared}")

    save_similarity_outputs(
        "algorithm_similarity_results", names, source_paths, alg_sim, pairwise_rows
    )
    print("Saved outputs to algorithm_similarity_results/")


if __name__ == "__main__":
    main(sys.argv[1:])
