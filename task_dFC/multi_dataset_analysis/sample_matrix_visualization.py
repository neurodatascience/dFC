import argparse
import json
import os
import sys

import numpy as np

from pydfc.ml_utils import (
    PLSEmbedder,
    dFC_feature_extraction,
    find_available_subjects,
    process_SB_features,
    select_num_components_binary_groupcv,
    subject_center,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_functions import (  # pyright: ignore[reportMissingImports]
    plot_samples_features,
    save_scalar_colorbar,
)

use_raw_features = False  # if True, use raw dFC features instead of embedded features
normalize_dFC = False
FCS_proba_for_SB = True
train_test_ratio = 0.8

if use_raw_features:
    raw_or_embedded = "_raw"
else:
    raw_or_embedded = ""

#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to visualize the feature-sample matrix for each dataset, task, and dFC measure.
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
        # TASKS_to_include = multi_dataset_info["real_data"]["TASKS_to_include"] # temporary !!!!
        TASKS_to_include = [
            "task-Axcpt",
            "task-CIC",
            "task-Cuedts",
            "task-feedback",
            "task-IHG",
            "task-matching",
            "task-motor",
            "task-Stern",
            "task-Stroop",
        ]
    elif simul_or_real == "simulated":
        main_root = multi_dataset_info["simulated_data"]["main_root"]
        DATASETS = multi_dataset_info["simulated_data"]["DATASETS"]
        TASKS_to_include = multi_dataset_info["simulated_data"]["TASKS_to_include"]

    output_root = f"{multi_dataset_info['output_root']}/feature-sample/{simul_or_real}"
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

        for dFC_id in range(0, 7):
            DATA = {}
            for session in SESSIONS[:1]:  # Only process the first session
                for task_id, task in enumerate(TASKS):
                    if task not in TASKS_to_include:
                        print(f"Skipping task: {task} as it is not in TASKS_to_include.")
                        continue
                    for run in RUNS[task][:1]:  # Only process the first run
                        print(
                            f"Processing dataset: {dataset}, task: {task}, run: {run}, session: {session}, dFC_id: {dFC_id}"
                        )

                        SUBJECTS = find_available_subjects(
                            dFC_root=dFC_root,
                            task=task,
                            run=run,
                            session=session,
                            dFC_id=dFC_id,
                        )

                        if task == "task-paingen":
                            # due to computational load, only use 100 subjects for this task
                            SUBJECTS = SUBJECTS[:100]

                        if (
                            task == "task-lowFreqLongRest"
                            or task == "task-lowFreqShortRest"
                            or task == "task-lowFreqShortTask"
                        ):
                            # due to computational load, only use 100 subjects for this task
                            SUBJECTS = SUBJECTS[:100]

                        # randomly select train_test_ratio of the subjects for training
                        # and rest for testing using numpy.random.choice
                        train_subjects = np.random.choice(
                            SUBJECTS, int(train_test_ratio * len(SUBJECTS)), replace=False
                        )
                        test_subjects = np.setdiff1d(SUBJECTS, train_subjects)
                        print(
                            f"Number of train subjects: {len(train_subjects)} and test subjects: {len(test_subjects)}"
                        )

                        (
                            X_train,
                            X_test,
                            y_train,
                            y_test,
                            subj_label_train,
                            subj_label_test,
                            measure_name,
                            measure_is_state_based,
                        ) = dFC_feature_extraction(
                            task=task,
                            train_subjects=train_subjects,
                            test_subjects=test_subjects,
                            dFC_id=dFC_id,
                            roi_root=roi_root,
                            dFC_root=dFC_root,
                            run=run,
                            session=session,
                            dynamic_pred="no",
                            normalize_dFC=normalize_dFC,
                            FCS_proba_for_SB=FCS_proba_for_SB,  # for state-based dFC features, we use FCS_proba
                        )

                        if measure_name is None:
                            print(
                                f"Skipping dataset: {dataset}, task: {task}, run: {run}, session: {session}, dFC_id: {dFC_id} due to no measure_name."
                            )
                            continue

                        if measure_is_state_based:
                            X_train = process_SB_features(
                                X=X_train, measure_name=measure_name
                            )
                            X_test = process_SB_features(
                                X=X_test, measure_name=measure_name
                            )
                        # center the data by subject before embedding to remove subject effects
                        # separately for train and test sets to avoid data leakage
                        # for both state-based and state-free methods
                        X_train_centered = subject_center(
                            X_train, subj_label_train, mode="demean"
                        )
                        X_test_centered = subject_center(
                            X_test, subj_label_test, mode="demean"
                        )
                        if not measure_is_state_based:
                            # embed dFC features using PLS regression, which is a supervised embedding method that finds the components that best explain the variance in the labels
                            best_n, _ = select_num_components_binary_groupcv(
                                X=X_train_centered,
                                y=y_train,
                                groups=subj_label_train,
                                embedding_method="PLS",
                                n_list=[
                                    2,
                                    3,
                                    4,
                                    5,
                                    10,
                                    15,
                                    20,
                                    25,
                                    30,
                                    40,
                                    50,
                                ],  # you can adjust this range based on your data
                                cv=5,  # more stable
                            )
                            pls = PLSEmbedder(n_components=best_n, scale=True)
                            # fit on train set
                            X_train_embedded = pls.fit_transform(
                                X_train_centered, y_train
                            )
                            assert (
                                X_train_embedded.shape[0] == y_train.shape[0]
                            ), "Number of samples do not match."
                            # only transform test set
                            if X_test is not None:
                                X_test_embedded = pls.transform(X_test_centered)
                                assert (
                                    X_test_embedded.shape[0] == y_test.shape[0]
                                ), "Number of samples do not match."
                            else:
                                X_test_embedded = None
                        else:
                            # for state-based measures, we skip the embedding step and just use the original features
                            X_train_embedded = X_train
                            X_test_embedded = X_test

                        assert (
                            task not in DATA
                        ), f"Task {task} already exists in DATA. Overwriting."
                        DATA[task] = {
                            "X_train": X_train,
                            "X_test": X_test,
                            "X_train_embedded": X_train_embedded,
                            "X_test_embedded": X_test_embedded,
                            "y_train": y_train,
                            "y_test": y_test,
                            "subj_label_train": subj_label_train,
                            "subj_label_test": subj_label_test,
                            "measure_name": measure_name,
                        }
            # save the data
            # save each task in a separate file and name the file as the task name, measure name, and dataset name
            for task in DATA.keys():
                if use_raw_features:
                    X_train = DATA[task]["X_train"]
                    X_test = DATA[task]["X_test"]
                else:
                    X_train = DATA[task]["X_train_embedded"]
                    X_test = DATA[task]["X_test_embedded"]
                y_train = DATA[task]["y_train"]
                y_test = DATA[task]["y_test"]
                subj_label_train = DATA[task]["subj_label_train"]
                subj_label_test = DATA[task]["subj_label_test"]
                measure_name = DATA[task]["measure_name"]

                if X_train is None or X_test is None:
                    print(f"Skipping task {task} due to embedding error.")
                    continue

                if not os.path.exists(f"{output_root}/processed_data"):
                    os.makedirs(f"{output_root}/processed_data")
                np.save(
                    f"{output_root}/processed_data/{dataset}_{task}_{measure_name}.npy",
                    DATA[task],
                )

                for group, X, y in zip(
                    ["train", "test"], [X_train, X_test], [y_train, y_test]
                ):
                    # if the folder does not exist, create it
                    if not os.path.exists(f"{output_root}/{measure_name}"):
                        os.makedirs(f"{output_root}/{measure_name}")

                    # A) Unsorted (your first vis, but rotated so time is horizontal)
                    plot_samples_features(
                        X,
                        y,
                        sample_order="original",
                        feature_order="original",
                        save_path=f"{output_root}/{measure_name}/feature-sample_{simul_or_real}_unsorted_{task}_{group}{raw_or_embedded}.png",
                        show=False,
                    )

                    # B) Label-sorted (your third vis)
                    plot_samples_features(
                        X,
                        y,
                        sample_order="label",
                        feature_order="original",
                        save_path=f"{output_root}/{measure_name}/feature-sample_{simul_or_real}_sorted-label_{task}_{group}{raw_or_embedded}.png",
                        show=False,
                    )

                    # C) clustering
                    plot_samples_features(
                        X,
                        y,
                        sample_order="cluster",
                        feature_order="original",
                        save_path=f"{output_root}/{measure_name}/feature-sample_{simul_or_real}_clustered-samples_{task}_{group}{raw_or_embedded}.png",
                        show=False,
                    )

                    save_scalar_colorbar(
                        cmap="coolwarm",
                        vmin=-1.6,
                        vmax=1.6,  # use the same V_RANGE you use in plots
                        label="z-scored feature value",
                        filename=f"{output_root}/zscore_colorbar.png",
                    )
