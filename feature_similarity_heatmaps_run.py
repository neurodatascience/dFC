# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


# %%
import os
os.makedirs("feature_similarity_results", exist_ok=True)
os.makedirs("feature_similarity_results/pdf", exist_ok=True)
os.makedirs("feature_similarity_results/jpg", exist_ok=True)


# %%
root = "/home/kinichen/scratch/data/pydfc_validator/similarity_assessments_complete"

with open(f"{root}/similarity.pkl", "rb") as f:
    similarity = pickle.load(f)
    
print(similarity.keys())    # layer 1 of hierarchy is datasets, then subjects, sessions, runs, tasks, etc. (pydFC objects)


# %%
dataset_id = "ds003465"
subject_id = "sub-f1027ao"
session_id = "ses-wave1bas"
run_id = "run-2"
task_id = "task-Stroop"

sim_ex = similarity[dataset_id][subject_id][session_id][run_id][task_id]
measures = sim_ex["matrix"]["measure_lst"]
methods = [method.MEASURE_NAME for method in measures]

matrix_ex = sim_ex["matrix"]["all"]["spearman"]  # similarity matrix for all methods
print("Example matrix shape:", matrix_ex.shape)


# %%
######### Helper functions to collect and aggregate similarity matrices based on filters 
# for various levels (dataset, subject, session, run, task) #########

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

                        matrices.append(
                            task_data["matrix"][similarity_key][metric]
                        )

    return matrices




def aggregate_similarity_matrices(
    matrices,
    aggregation="mean"
):
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
        raise ValueError(
            f"Unknown aggregation: {aggregation}"
        )

    return aggregated, len(matrices)



def plot_similarity_heatmap(
    matrix,
    aggregation_size=None,
    method_names=methods,
    title="Similarity Heatmap",
    annot=False,
    figsize=(10, 8),
    cmap="viridis",
    cluster=True,
    cluster_method="average"
):
    
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
        method_names = [
            method_names[i]
            for i in order
        ]
    
    
    plt.figure(figsize=figsize)

    sns.heatmap(
        matrix,
        annot=annot,
        xticklabels=method_names,
        yticklabels=method_names,
        cmap=cmap,
    )
    
    if aggregation_size is not None:
        title += f" (n={aggregation_size})"

    plt.title(title)

    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(rotation=0, fontsize=6)

    plt.tight_layout()
    
    # For running .py, save fig
    plt.savefig(f"feature_similarity/pdf/{title}.pdf", bbox_inches="tight")
    plt.savefig(f"feature_similarity/jpg/{title}.jpg", bbox_inches="tight")
    plt.close()




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
    
    matrices = collect_similarity_matrices(
        similarity,
        task_id=task_id
    )

    aggregated, aggregation_size = aggregate_similarity_matrices(
        matrices,
        aggregation="mean"
    )

    plot_similarity_heatmap(
        aggregated,
        aggregation_size,
        title=f"{task_id}"
    )



# %%
### Average over everything for a specific DATASET ###

dataset_ids = sorted(similarity.keys())

for dataset_id in dataset_ids:

    matrices = collect_similarity_matrices(
        similarity,
        dataset_id=dataset_id
    )

    aggregated, aggregation_size = aggregate_similarity_matrices(
        matrices,
        aggregation="mean"
    )

    plot_similarity_heatmap(
        aggregated,
        aggregation_size,
        title=f"{dataset_id}"
    )




# %%
### Average over EVERYTHING ###

matrices = collect_similarity_matrices(
    similarity
)

aggregated, aggregation_size = aggregate_similarity_matrices(
    matrices,
    aggregation="mean"
)

plot_similarity_heatmap(
    aggregated,
    aggregation_size,
    title=f"Mean dFC feature similarity between methods"
)



# %%
### Standard deviation over EVERYTHING ###

# Measures which method pairs are more stable vs. more variable across filters

matrices = collect_similarity_matrices(
    similarity
)

aggregated, aggregation_size = aggregate_similarity_matrices(
    matrices,
    aggregation="std"
)

plot_similarity_heatmap(
    aggregated,
    aggregation_size,
    title=f"Standard deviation of dFC feature similarity between methods"
)




print("Complete! Figures saved to feature_similarity_results/")