import argparse
import json
import os
import sys

from pydfc.dfc_utils import TR_intersection, rank_norm
from pydfc.ml_utils import find_available_subjects, load_dFC

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from helper_functions import (  # pyright: ignore[reportMissingImports]
    build_experiment_display_info,
    figure_dfc_matrices_window_png,
)

normalize_dFC = True

#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to make figures/tables from multi-dataset ML results.
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
    output_root = f"{multi_dataset_info['output_root']}/dFC/{simul_or_real}"

    if not os.path.exists(output_root):
        os.makedirs(output_root)

    _, task_to_experiment, _, _ = build_experiment_display_info(
        tasks_iterable=TASKS_to_include,
        task_reference_order=TASKS_to_include,
        simul_or_real=simul_or_real,
    )

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

        DATA = {}
        for dFC_id in range(0, 7):
            for session in SESSIONS[:1]:  # Only process the first session
                for task_id, task in enumerate(TASKS):
                    for run in RUNS[task][:1]:  # Only process the first run
                        print(
                            f"Processing dataset: {dataset}, task: {task}, run: {run}, session: {session}, dFC_id: {dFC_id}"
                        )

                        SUBJECTS = find_available_subjects(
                            dFC_root=dFC_root,
                            task=task,
                            dFC_id=dFC_id,
                            session=session,
                            run=run,
                        )
                        if len(SUBJECTS) == 0:
                            print(
                                f"No subjects found for dataset: {dataset}, task: {task}, run: {run}, session: {session}, dFC_id: {dFC_id}"
                            )
                            continue

                        subj = SUBJECTS[0]  # Only process the first subject

                        dFC = load_dFC(
                            dFC_root=dFC_root,
                            subj=subj,
                            task=task,
                            dFC_id=dFC_id,
                            run=run,
                            session=session,
                        )

                        if not task in DATA:
                            DATA[task] = {}
                        DATA[task][dFC.measure.measure_name] = dFC

        # visualize the dFC matrices for each task
        for task in DATA.keys():
            # first find common TRs across measures
            common_TRs = TR_intersection(
                [DATA[task][measure_name] for measure_name in DATA[task]]
            )

            dFC_mat_dict = {}
            for measure_name in DATA[task]:
                dFC = DATA[task][measure_name]
                dFC_mat = dFC.get_dFC_mat(TRs=common_TRs)
                if normalize_dFC:
                    dFC_mat = rank_norm(dFC_mat)
                dFC_mat_dict[measure_name] = dFC_mat
            figure_dfc_matrices_window_png(
                dFC_mat_dict,
                common_TRs,
                window_len=10,
                cmap="plasma",
                outfile=(
                    f"{output_root}/dFC_{dataset}_"
                    f"{task_to_experiment.get(task, task).replace(' ', '_').replace('/', '-')}_"
                    f"{task}_mid_10.png"
                ),
                dpi=600,
            )

        print(f"Saved data for dataset {dataset}")
