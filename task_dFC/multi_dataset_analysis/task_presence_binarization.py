import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from pydfc.ml_utils import find_available_subjects, load_task_data
from pydfc.task_utils import extract_task_presence

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_functions import (  # pyright: ignore[reportMissingImports]
    build_experiment_display_info,
)

#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to visualize task timing and binarization results for multiple datasets.
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

    print("Multi-Dataset Analysis started ...")

    if simul_or_real == "real":
        main_root = multi_dataset_info["real_data"]["main_root"]
        DATASETS = multi_dataset_info["real_data"]["DATASETS"]
        TASKS_to_include = multi_dataset_info["real_data"]["TASKS_to_include"]
    elif simul_or_real == "simulated":
        main_root = multi_dataset_info["simulated_data"]["main_root"]
        DATASETS = multi_dataset_info["simulated_data"]["DATASETS"]
        TASKS_to_include = multi_dataset_info["simulated_data"]["TASKS_to_include"]
    output_root = f"{multi_dataset_info['output_root']}/task_timing/{simul_or_real}"

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    _, task_to_experiment, _, _ = build_experiment_display_info(
        tasks_iterable=TASKS_to_include,
        task_reference_order=TASKS_to_include,
        simul_or_real=simul_or_real,
    )

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
            features_all = None
            for session in SESSIONS:
                print(f"Processing task: {task}")
                SUBJECTS = find_available_subjects(
                    dFC_root=dFC_root,
                    task=task,
                    dFC_id=None,
                    session=session,
                )
                subj = SUBJECTS[0]

                run = RUNS[task][0]
                try:
                    task_data = load_task_data(
                        roi_root=roi_root, subj=subj, task=task, run=run, session=session
                    )
                except:
                    continue

                stimulus_timing = np.multiply(task_data["event_labels"] != 0, 1)

                event_labels_all_task_hrf, _ = extract_task_presence(
                    event_labels=task_data["event_labels"],
                    TR_task=1 / task_data["Fs_task"],
                    TR_mri=1 / task_data["Fs_task"],
                    TR_array=None,
                    binary=False,
                    binarizing_method="GMM",
                    no_hrf=False,
                )

                task_presence, indices = extract_task_presence(
                    event_labels=task_data["event_labels"],
                    TR_task=1 / task_data["Fs_task"],
                    TR_mri=1 / task_data["Fs_task"],
                    TR_array=None,
                    binary=True,
                    binarizing_method="GMM",
                    no_hrf=False,
                )
                # plot event_labels_all_task_hrf
                plt.figure(figsize=(250, 10))
                print(
                    f"Fs_task: {task_data['Fs_task']}, TR_mri: {task_data['TR_mri']}, length of event_labels_all_task_hrf: {len(event_labels_all_task_hrf)}"
                )
                plt.plot(
                    stimulus_timing,
                    label="Stimulus Timing",
                    color="#B8AD6F",
                    linewidth=15,
                )
                # plt.plot(task_presence, label="Task Presence", color="blue", linewidth=3)
                plt.plot(
                    event_labels_all_task_hrf,
                    label="HRF Convolved",
                    color="#010101",
                    linewidth=8,
                )
                # plot a vertical dashed line at every TR_mri
                for i in range(
                    0,
                    len(event_labels_all_task_hrf),
                    int(task_data["TR_mri"] * task_data["Fs_task"]),
                ):
                    plt.axvline(x=i, color="#c20707", linestyle="--", linewidth=5.0)

                # on_indices are index in indices where task_presence=1
                on_indices = indices[task_presence[indices] == 1]
                # off_indices are index in indices where task_presence=0
                off_indices = indices[task_presence[indices] == 0]
                plt.scatter(
                    on_indices,
                    event_labels_all_task_hrf[on_indices],
                    color="#7ab3dc",
                    label="on_indices",
                    s=300,
                    zorder=10,
                )
                plt.scatter(
                    off_indices,
                    event_labels_all_task_hrf[off_indices],
                    color="#A8ACAD",
                    label="off_indices",
                    s=300,
                    zorder=10,
                )

                # remove all axis and spines, show only x axis
                plt.gca().spines["top"].set_visible(False)
                plt.gca().spines["right"].set_visible(False)
                plt.gca().spines["left"].set_visible(False)
                plt.gca().spines["bottom"].set_visible(True)
                # increase bottom spine width
                plt.gca().spines["bottom"].set_linewidth(5)
                plt.gca().yaxis.set_visible(False)
                plt.gca().xaxis.set_visible(True)

                # # set background color to lite pink
                # plt.gca().set_facecolor("#F7EFEF")

                # set x ticks to be every TR_mri
                step_factor = 1
                # if the length of event_labels_all_task_hrf > 6500, set step_factor to 5
                # to make the plot less crowded
                if len(event_labels_all_task_hrf) > 6500:
                    step_factor = (
                        np.ceil(len(event_labels_all_task_hrf) / 6500).astype(int) + 1
                    )
                step = int(
                    round(task_data["TR_mri"] * task_data["Fs_task"] * step_factor)
                )
                step = max(step, 1)  # avoid step=0

                ticks = np.arange(0, len(event_labels_all_task_hrf), step)
                plt.gca().set_xticks(ticks)

                TR_labels = np.arange(
                    len(ticks) * step_factor, step=step_factor
                )  # same length as ticks
                # label each tick as time in seconds, TR_labels*TR_mri
                time_labels = np.round(TR_labels * task_data["TR_mri"]).astype(int)
                plt.gca().set_xticklabels(time_labels, fontsize=50)
                plt.xlabel("Time (sec)", fontsize=60)

                experiment_label = task_to_experiment.get(task, task)
                experiment_key = str(experiment_label).replace(" ", "_").replace("/", "-")

                plt.savefig(
                    f"{output_root}/task_timing_{experiment_key}_{task}.png",
                    dpi=120,
                    bbox_inches="tight",
                    pad_inches=0.1,
                    format="png",
                )
                if task == "task-Localizer":
                    plt.savefig(
                        f"{output_root}/task_timing_{experiment_key}_{task}.svg",
                        dpi=120,
                        bbox_inches="tight",
                        pad_inches=0.1,
                        format="svg",
                    )

                plt.close()
