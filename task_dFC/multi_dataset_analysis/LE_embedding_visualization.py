import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from pydfc.ml_utils import (
    LE_transform,
    PLSEmbedder,
    dFC_feature_extraction,
    find_available_subjects,
    process_SB_features,
)

fig_dpi = 120
fig_bbox_inches = "tight"
fig_pad = 0.1
show_title = True
save_fig_format = "png"  # pdf, png,

normalize_dFC = False

#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to analyze and visualize LE-transformed features across multiple datasets.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument(
        "--multi_dataset_info", type=str, help="path to multi-dataset info file"
    )
    parser.add_argument(
        "--simul_or_real", type=str, help="Specify 'simulated' or 'real' data"
    )

    args = parser.parse_args()

    multi_dataset_info = args.multi_dataset_info
    simul_or_real = args.simul_or_real

    # Read dataset info
    with open(multi_dataset_info, "r") as f:
        multi_dataset_info = json.load(f)

    if simul_or_real == "real":
        main_root = multi_dataset_info["real_data"]["main_root"]
        DATASETS = multi_dataset_info["real_data"]["DATASETS"]
        TASKS_to_include = multi_dataset_info["real_data"]["TASKS_to_include"]
    elif simul_or_real == "simulated":
        main_root = multi_dataset_info["simulated_data"]["main_root"]
        DATASETS = multi_dataset_info["simulated_data"]["DATASETS"]
        TASKS_to_include = multi_dataset_info["simulated_data"]["TASKS_to_include"]

    output_root = f"{multi_dataset_info['output_root']}/LE_embed/{simul_or_real}"

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    for dataset in DATASETS:
        dataset_info_file = f"{main_root}/{dataset}/codes/dataset_info.json"
        roi_root = f"{main_root}/{dataset}/derivatives/ROI_timeseries"
        dFC_root = f"{main_root}/{dataset}/derivatives/dFC_assessed"

        # Read dataset info
        with open(dataset_info_file, "r") as f:
            dataset_info = json.load(f)

        if "SESSIONS" in dataset_info:
            SESSIONS = dataset_info["SESSIONS"]
        else:
            SESSIONS = None
        if SESSIONS is None:
            SESSIONS = [None]

        TASKS = dataset_info["TASKS"]

        if "RUNS" in dataset_info:
            RUNS = dataset_info["RUNS"]
        else:
            RUNS = None
        if RUNS is None:
            RUNS = {task: [None] for task in TASKS}

        for session in SESSIONS:
            for task_id, task in enumerate(TASKS):
                for run in RUNS[task][:1]:
                    for dFC_id in range(7):
                        try:
                            SUBJECTS = find_available_subjects(
                                dFC_root=dFC_root,
                                task=task,
                                dFC_id=dFC_id,
                                session=session,
                                run=run,
                            )
                            if len(SUBJECTS) == 0:
                                print(
                                    f"No subjects found for task {task}, dFC_id {dFC_id}, session {session}, run {run}."
                                )
                                continue
                            SUBJECTS = SUBJECTS[0:1]
                            print(f"Number of subjects: {len(SUBJECTS)}")

                            X, _, y, _, subj_label, _, measure_name = (
                                dFC_feature_extraction(
                                    task=task,
                                    train_subjects=SUBJECTS,
                                    test_subjects=[],
                                    dFC_id=dFC_id,
                                    roi_root=roi_root,
                                    dFC_root=dFC_root,
                                    run=run,
                                    session=session,
                                    dynamic_pred="no",
                                    normalize_dFC=normalize_dFC,
                                    FCS_proba_for_SB=True,
                                )
                            )

                            assert (
                                X.shape[0] == y.shape[0]
                            ), "Number of samples do not match."
                            assert (
                                X.shape[0] == subj_label.shape[0]
                            ), "Number of samples do not match."

                            if measure_name in [
                                "CAP",
                                "Clustering",
                                "ContinuousHMM",
                                "DiscreteHMM",
                                "Windowless",
                            ]:
                                X = process_SB_features(X=X, measure_name=measure_name)

                            print(f"Task: {task}")
                            print(measure_name)
                            print(X.shape, y.shape)
                            print(silhouette_score(X, y))

                            # embed the features
                            # n_components = "auto"
                            n_components = 3
                            for embedding_method in ["PCA", "PLS", "LE"]:
                                if embedding_method == "PCA":
                                    X_embedded = PCA(
                                        n_components=n_components,
                                        whiten=False,
                                        svd_solver="full",
                                        random_state=0,
                                    ).fit_transform(X)
                                elif embedding_method == "PLS":
                                    X_embedded = (
                                        PLSEmbedder(
                                            n_components=n_components, scale=False
                                        )
                                        .fit(X, y)
                                        .transform(X)
                                    )
                                elif embedding_method == "LE":
                                    X_embedded = LE_transform(
                                        X,
                                        n_components=n_components,
                                        n_neighbors=125,
                                        distance_metric="correlation",
                                    )

                                # X_embedded = TSNE(n_components=n_components, learning_rate='auto', init='random', perplexity=125, metric="correlation").fit_transform(X)
                                print(silhouette_score(X_embedded, y))
                                print(X_embedded.shape)

                                # plot
                                # ---- publication style (light touch) ----
                                mpl.rcParams.update(
                                    {
                                        "legend.fontsize": 10,
                                        "axes.linewidth": 0.9,
                                        "pdf.fonttype": 42,
                                        "ps.fonttype": 42,  # keep text as text in PDF/SVG
                                        "savefig.bbox": "tight",
                                        "savefig.dpi": 300,
                                        "figure.dpi": 150,
                                    }
                                )
                                fig = plt.figure(figsize=(7, 7))
                                ax = fig.add_subplot(111, projection="3d")

                                colors = ("#B1B1B1", "#2F5BD3")

                                for label in np.unique(y):
                                    ax.scatter(
                                        X_embedded[y == label, 0],
                                        X_embedded[y == label, 1],
                                        X_embedded[y == label, 2],
                                        label=["rest", "task"][label],
                                        s=50,
                                        c=[colors[label]],
                                        edgecolors="#202020",
                                        linewidths=0.25,
                                        depthshade=False,
                                    )
                                plt.legend()

                                # remove tick labels
                                ax.set_xticklabels([])
                                ax.set_yticklabels([])
                                ax.set_zticklabels([])

                                plt.savefig(
                                    f"{output_root}/{embedding_method}_embed_{task}_{measure_name}.png",
                                    dpi=fig_dpi,
                                    bbox_inches=fig_bbox_inches,
                                    pad_inches=fig_pad,
                                    format=save_fig_format,
                                )

                                plt.close()
                        except Exception as e:
                            print(
                                f"Error processing task {task}, dFC_id {dFC_id}, session {session}, run {run}: {e}"
                            )
                            continue
