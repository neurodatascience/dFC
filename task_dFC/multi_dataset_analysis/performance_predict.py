import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pydfc import data_loader
from pydfc.data_loader import find_subj_list
from pydfc.ml_utils import find_available_subjects, load_task_data
from pydfc.task_utils import (
    calc_relative_task_on,
    calc_rest_duration,
    calc_task_duration,
    calc_transition_freq,
    cohen_d_bold,
    compute_optimality_index,
    compute_periodicity_index,
    extract_task_presence,
    periodicity_autocorr,
)

fig_bbox_inches = "tight"
fig_pad = 0.1
show_title = False
save_fig_format = "png"  # pdf, png,

level = "group_lvl"
keys_not_to_include = [
    "Logistic regression permutation p_value",
    "Logistic regression permutation score mean",
    "Logistic regression permutation score std",
    "SVM permutation p_value",
    "SVM permutation score mean",
    "SVM permutation score std",
]

#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to predict performance based on task design features and BOLD signals across multiple datasets.
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
    output_root = (
        f"{multi_dataset_info['output_root']}/performance_predictor/{simul_or_real}"
    )

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    task_ratio_all = {}
    transition_freq_all = {}
    rest_durations_original_all = {}
    task_durations_original_all = {}
    rest_durations_all = {}
    task_durations_all = {}
    PI_all = {}
    OI_all = {}
    PAC_all = {}
    for dataset in DATASETS:

        print(f"Processing dataset: {dataset}")
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
                if not task in TASKS_to_include:
                    continue
                for run in RUNS[task]:
                    SUBJECTS = find_subj_list(roi_root)
                    # print(f"Number of subjects: {len(SUBJECTS)}")

                    for subj in SUBJECTS:

                        try:
                            task_data = load_task_data(
                                roi_root=roi_root,
                                subj=subj,
                                task=task,
                                run=run,
                                session=session,
                            )
                        except FileNotFoundError:
                            continue

                        task_presence, indices = extract_task_presence(
                            event_labels=task_data["event_labels"],
                            TR_task=1 / task_data["Fs_task"],
                            TR_mri=task_data["TR_mri"],
                            binary=True,
                            binarizing_method="GMM",
                            no_hrf=False,
                        )

                        relative_task_on = calc_relative_task_on(task_presence[indices])
                        num_of_transitions, relative_transition_freq = (
                            calc_transition_freq(task_presence[indices])
                        )
                        # calculate rest and task durations based on original event labels
                        event_labels = np.multiply(task_data["event_labels"] != 0, 1)
                        rest_durations_original = calc_rest_duration(
                            event_labels, TR_mri=1 / task_data["Fs_task"]
                        )
                        task_durations_original = calc_task_duration(
                            event_labels, TR_mri=1 / task_data["Fs_task"]
                        )
                        # calculate rest and task durations based on binary task presence
                        rest_durations = calc_rest_duration(
                            task_presence[indices], TR_mri=task_data["TR_mri"]
                        )
                        task_durations = calc_task_duration(
                            task_presence[indices], TR_mri=task_data["TR_mri"]
                        )
                        # Periodicity Index (low entropy => high periodicity)
                        out = compute_periodicity_index(
                            event_labels=event_labels,
                            TR_task=1 / task_data["Fs_task"],
                            no_hrf=False,
                        )
                        PI = out["periodicity_index"]

                        # Optimality Index (how close the task design is to the optimal design)
                        out = compute_optimality_index(
                            event_labels=event_labels,
                            TR_task=1 / task_data["Fs_task"],
                            TR_mri=task_data["TR_mri"],
                        )
                        OI = out["OI_norm"]

                        # Periodicity via autocorrelation
                        out = periodicity_autocorr(
                            event_labels=event_labels,
                            TR_task=1 / task_data["Fs_task"],
                        )
                        PAC = out["periodicity"]

                        if not task in task_ratio_all:
                            task_ratio_all[task] = []
                        if not task in transition_freq_all:
                            transition_freq_all[task] = []
                        if not task in rest_durations_original_all:
                            rest_durations_original_all[task] = []
                        if not task in task_durations_original_all:
                            task_durations_original_all[task] = []
                        if not task in rest_durations_all:
                            rest_durations_all[task] = []
                        if not task in task_durations_all:
                            task_durations_all[task] = []
                        if not task in PI_all:
                            PI_all[task] = []
                        if not task in OI_all:
                            OI_all[task] = []
                        if not task in PAC_all:
                            PAC_all[task] = []
                        task_ratio_all[task].append(relative_task_on)
                        transition_freq_all[task].append(relative_transition_freq)
                        # rest_durations and task_durations are lists
                        rest_durations_original_all[task].extend(rest_durations_original)
                        task_durations_original_all[task].extend(task_durations_original)
                        rest_durations_all[task].extend(rest_durations)
                        task_durations_all[task].extend(task_durations)
                        PI_all[task].append(PI)
                        OI_all[task].append(OI)
                        PAC_all[task].append(PAC)

    task_design_features = {
        "task_ratio_all": task_ratio_all,
        "transition_freq_all": transition_freq_all,
        "rest_durations_original_all": rest_durations_original_all,
        "task_durations_original_all": task_durations_original_all,
        "rest_durations_all": rest_durations_all,
        "task_durations_all": task_durations_all,
        "PI_all": PI_all,
        "OI_all": OI_all,
        "PAC_all": PAC_all,
    }

    CohensD_across_task = {}
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}")
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

        for task in TASKS:
            if task not in TASKS_to_include:
                print(f"Skipping task {task} as it's not in the inclusion list.")
                continue
            d_values_all = []
            for session in SESSIONS:
                print(f"Processing task: {task}")
                SUBJECTS = find_available_subjects(
                    dFC_root=dFC_root,
                    task=task,
                    dFC_id=None,
                    session=session,
                )
                for subj in SUBJECTS:
                    for run in RUNS[task]:
                        try:
                            task_data = load_task_data(
                                roi_root=roi_root,
                                subj=subj,
                                task=task,
                                run=run,
                                session=session,
                            )
                        except:
                            continue

                        if run is None:
                            if session is None:
                                BOLD_file_name = "{subj_id}_{task}_time-series.npy"
                            else:
                                BOLD_file_name = (
                                    "{subj_id}_{session}_{task}_time-series.npy"
                                )
                        else:
                            if session is None:
                                BOLD_file_name = "{subj_id}_{task}_{run}_time-series.npy"
                            else:
                                BOLD_file_name = (
                                    "{subj_id}_{session}_{task}_{run}_time-series.npy"
                                )
                        try:
                            BOLD = data_loader.load_TS(
                                data_root=roi_root,
                                file_name=BOLD_file_name,
                                subj_id2load=subj,
                                task=task,
                                session=session,
                                run=run,
                            )
                        except Exception as e:
                            print(f"Error loading BOLD data: {e}")
                            continue
                        BOLD_data = BOLD.data  # np.ndarray (n_ROIs, n_TRs)

                        Fs_task = task_data["Fs_task"]
                        TR_task = 1 / Fs_task

                        TR_array = np.arange(0, BOLD_data.shape[1])
                        task_presence, indices = extract_task_presence(
                            event_labels=task_data["event_labels"],
                            TR_task=TR_task,
                            TR_mri=task_data["TR_mri"],
                            binary=True,
                            binarizing_method="GMM",
                            no_hrf=False,
                            TR_array=TR_array,
                        )

                        # if n_TRs do not match, align them
                        if BOLD_data.shape[1] != task_presence.shape[0]:
                            print(
                                f"Before alignment, shape of task_presence: {task_presence.shape}, shape of BOLD_data: {BOLD_data.shape}"
                            )
                            min_TRs = min(BOLD_data.shape[1], task_presence.shape[0])
                            task_presence = task_presence[:min_TRs]
                            BOLD_data = BOLD_data[:, :min_TRs]
                            print(
                                f"After alignment, shape of task_presence: {task_presence.shape}, shape of BOLD_data: {BOLD_data.shape}"
                            )
                            # also adjust indices
                            indices = [i for i in indices if i < min_TRs]
                        task_presence = task_presence[indices]  # (n_TRs,)
                        BOLD_data = BOLD_data[:, indices]  # (n_ROIs, n_TRs)

                        assert BOLD_data.shape[1] == task_presence.shape[0]

                        cohen_d = cohen_d_bold(X=BOLD_data.T, y=task_presence)
                        d_values_all.append(cohen_d)

            if len(d_values_all) == 0:
                print(f"No data found for task {task} in dataset {dataset}. Skipping.")
                continue
            d_values_all = np.array(d_values_all)  # (n_subjectsxrunsxsessions, n_ROIs)
            avg_d_values = np.nanmean(d_values_all, axis=0)  # (n_ROIs,)
            if not task in CohensD_across_task:
                CohensD_across_task[task] = []
            CohensD_across_task[task].extend(avg_d_values)

    ALL_ML_SCORES = None
    for dataset in DATASETS:
        print(f"Processing dataset: {dataset}")
        dataset_info_file = f"{main_root}/{dataset}/codes/dataset_info.json"
        ML_root = f"{main_root}/{dataset}/derivatives/ML"

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

        # find all ML_scores_classify_dFC-id.npy in the ML_root/classfication/ folder
        # for now we will only use the first session
        session = SESSIONS[0]
        if session is None:
            input_dir = f"{ML_root}/classification"
        else:
            input_dir = f"{ML_root}/classification/{session}"
        if not os.path.exists(input_dir):
            print(
                f"Input directory {input_dir} does not exist. Skipping dataset {dataset}."
            )
            continue
        ALL_ML_SCORES_FILES = os.listdir(input_dir)
        ALL_ML_SCORES_FILES = [
            f for f in ALL_ML_SCORES_FILES if "ML_scores_classify_" in f
        ]
        for f in ALL_ML_SCORES_FILES:
            try:
                ML_scores_new = np.load(f"{input_dir}/{f}", allow_pickle=True).item()
                # ML_scores_new_updated is a new dictionary with same keys as ML_scores_new but empty lists
                ML_scores_new_updated = {
                    key: []
                    for key in ML_scores_new[level].keys()
                    if key not in keys_not_to_include
                }
                for i in range(len(ML_scores_new[level]["task"])):
                    if task not in TASKS_to_include:
                        continue

                    for key in ML_scores_new_updated.keys():
                        ML_scores_new_updated[key].append(ML_scores_new[level][key][i])

                if ALL_ML_SCORES is None:
                    ALL_ML_SCORES = ML_scores_new_updated
                else:
                    for key in ML_scores_new_updated.keys():
                        if key in ALL_ML_SCORES:
                            ALL_ML_SCORES[key].extend(ML_scores_new_updated[key])
            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue

    # check that the lists in all keys have the same length
    if ALL_ML_SCORES is not None:
        lengths = [len(v) for v in ALL_ML_SCORES.values()]
        if len(set(lengths)) != 1:
            print(
                f"Warning: Not all keys have the same length in ALL_ML_SCORES. key and length pairs: {dict(zip(ALL_ML_SCORES.keys(), lengths))}"
            )

    embedding = "LE"
    metric = "SVM balanced accuracy"
    GROUP = "test"

    METHODS = set(ALL_ML_SCORES["dFC method"])
    all_scores = {method: {} for method in METHODS}
    for method in METHODS:
        for i in range(len(ALL_ML_SCORES["task"])):
            if (
                ALL_ML_SCORES["embedding"][i] == embedding
                and ALL_ML_SCORES["group"][i] == GROUP
                and ALL_ML_SCORES["dFC method"][i] == method
            ):
                if ALL_ML_SCORES["task"][i] not in all_scores[method]:
                    all_scores[method][ALL_ML_SCORES["task"][i]] = []
                all_scores[method][ALL_ML_SCORES["task"][i]].append(
                    ALL_ML_SCORES[metric][i]
                )

    # all_scores[<method>][<task>] is a list of scores across runs
    for method in all_scores:
        all_scores[method] = {k: np.array(v) for k, v in all_scores[method].items()}

    # we have task design features in task_design_features[task_ratio_all][task], task_design_features[transition_freq_all][task], task_design_features[rest_durations_all][task], task_design_features[task_durations_all][task]
    # we have CohensD in CohensD_across_task[task]
    # we have ML scores in all_scores[task]

    DATA = {
        "task": [],
        "task_ratio": [],
        "transition_freq": [],
        "rest_durations_original_mean": [],
        "task_durations_original_mean": [],
        "rest_durations_original_std": [],
        "task_durations_original_std": [],
        "rest_durations_mean": [],
        "task_durations_mean": [],
        "rest_durations_std": [],
        "task_durations_std": [],
        "PI_mean": [],
        "OI_mean": [],
        "PAC_mean": [],
        "cohen_d_max": [],
    }
    for task in TASKS_to_include:
        task_ratio = np.mean(task_design_features["task_ratio_all"][task])
        transition_freq = np.mean(task_design_features["transition_freq_all"][task])
        rest_durations_original_mean = np.mean(
            task_design_features["rest_durations_original_all"][task]
        )
        task_durations_original_mean = np.mean(
            task_design_features["task_durations_original_all"][task]
        )
        rest_durations_original_std = np.std(
            task_design_features["rest_durations_original_all"][task]
        )
        task_durations_original_std = np.std(
            task_design_features["task_durations_original_all"][task]
        )
        rest_durations_mean = np.mean(task_design_features["rest_durations_all"][task])
        task_durations_mean = np.mean(task_design_features["task_durations_all"][task])
        rest_durations_std = np.std(task_design_features["rest_durations_all"][task])
        task_durations_std = np.std(task_design_features["task_durations_all"][task])
        PI_mean = np.mean(PI_all[task])
        OI_mean = np.mean(OI_all[task])
        PAC_mean = np.mean(PAC_all[task])
        cohen_d_max = np.max(np.abs(CohensD_across_task[task]))

        DATA["task"].append(task)
        DATA["task_ratio"].append(task_ratio)
        DATA["transition_freq"].append(transition_freq)
        DATA["rest_durations_original_mean"].append(rest_durations_original_mean)
        DATA["task_durations_original_mean"].append(task_durations_original_mean)
        DATA["rest_durations_original_std"].append(rest_durations_original_std)
        DATA["task_durations_original_std"].append(task_durations_original_std)
        DATA["rest_durations_mean"].append(rest_durations_mean)
        DATA["task_durations_mean"].append(task_durations_mean)
        DATA["rest_durations_std"].append(rest_durations_std)
        DATA["task_durations_std"].append(task_durations_std)
        DATA["PI_mean"].append(PI_mean)
        DATA["OI_mean"].append(OI_mean)
        DATA["PAC_mean"].append(PAC_mean)
        DATA["cohen_d_max"].append(cohen_d_max)

        # Also add ML scores
        for method in all_scores:
            if f"classfication_score_{method}" not in DATA:
                DATA[f"classfication_score_{method}"] = []
            if task in all_scores[method]:
                score_mean = np.mean(all_scores[method][task])
            else:
                score_mean = np.nan
            DATA[f"classfication_score_{method}"].append(score_mean)

    # save DATA
    np.save(f"{output_root}/performance_predictor_data.npy", DATA)
