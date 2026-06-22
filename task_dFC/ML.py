import argparse
import json
import os
import traceback

import numpy as np
from joblib import Parallel, delayed

from pydfc.ml_utils import extract_task_features, task_presence_classification

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

#######################################################################################


def run_task_features_extraction(
    TASKS,
    RUNS,
    SESSIONS,
    roi_root,
    dFC_root,
    output_root,
):
    for session in SESSIONS:

        # Extract task features without HRF effect
        task_features = extract_task_features(
            TASKS=TASKS,
            RUNS=RUNS,
            session=session,
            roi_root=roi_root,
            dFC_root=dFC_root,
            no_hrf=True,
        )

        # Extract task features with HRF effect
        task_features_hrf = extract_task_features(
            TASKS=TASKS,
            RUNS=RUNS,
            session=session,
            roi_root=roi_root,
            dFC_root=dFC_root,
            no_hrf=False,
        )

        if session is None:
            folder = f"{output_root}/task_features"
        else:
            folder = f"{output_root}/task_features/{session}"
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except OSError as err:
            print(err)
        try:
            if not os.path.exists(f"{folder}/task_features.npy"):
                np.save(f"{folder}/task_features.npy", task_features)
            if not os.path.exists(f"{folder}/task_features_hrf.npy"):
                np.save(f"{folder}/task_features_hrf.npy", task_features_hrf)
        except OSError as err:
            print(err)


def classify_single_run(
    task, run, session, dFC_id, roi_root, dFC_root, dynamic_pred, normalize_dFC
):
    try:
        ML_scores_new = task_presence_classification(
            task=task,
            dFC_id=dFC_id,
            roi_root=roi_root,
            dFC_root=dFC_root,
            run=run,
            session=session,
            dynamic_pred=dynamic_pred,
            normalize_dFC=normalize_dFC,
        )
        return task, run, ML_scores_new
    except Exception as e:
        print(f"Error in task presence classification for {session} {task} {run}: {e}")
        traceback.print_exc()
        return task, run, None


def run_classification(
    dFC_id,
    TASKS,
    RUNS,
    SESSIONS,
    roi_root,
    dFC_root,
    output_root,
    dynamic_pred="no",
    normalize_dFC=False,
    n_jobs=-1,  # Number of parallel jobs; -1 = all available cores
):
    for session in SESSIONS:
        if session is not None:
            print(f"=================== {session} ===================")

        ML_scores = {
            "group_lvl": {},
            "subj_lvl": {},
        }

        # Parallel execution
        results = Parallel(n_jobs=n_jobs, verbose=0, backend="loky")(
            delayed(classify_single_run)(
                task,
                run,
                session,
                dFC_id,
                roi_root,
                dFC_root,
                dynamic_pred,
                normalize_dFC,
            )
            for task in TASKS
            for run in RUNS[task]
        )

        # Aggregate results
        for task, run, result in results:
            if result is None:
                continue
            for key in result["group_lvl"]:
                if key not in ML_scores["group_lvl"]:
                    ML_scores["group_lvl"][key] = []
                ML_scores["group_lvl"][key].extend(result["group_lvl"][key])
            for key in result["subj_lvl"]:
                if key not in ML_scores["subj_lvl"]:
                    ML_scores["subj_lvl"][key] = []
                ML_scores["subj_lvl"][key].extend(result["subj_lvl"][key])

        # Save output
        folder = (
            f"{output_root}/classification"
            if session is None
            else f"{output_root}/classification/{session}"
        )
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as err:
            print(err)

        np.save(f"{folder}/ML_scores_classify_{dFC_id}.npy", ML_scores)


#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to apply Machine Learning on dFC results to predict task presence.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument("--dataset_info", type=str, help="path to dataset info file")

    args = parser.parse_args()

    dataset_info_file = args.dataset_info

    # Read dataset info
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    print("Task presence prediction started ...")

    TASKS = dataset_info["TASKS"]
    if "RUNS" in dataset_info:
        RUNS = dataset_info["RUNS"]
    else:
        RUNS = None
    if RUNS is None:
        RUNS = {task: [None] for task in TASKS}

    if "SESSIONS" in dataset_info:
        SESSIONS = dataset_info["SESSIONS"]
    else:
        SESSIONS = None
    if SESSIONS is None:
        SESSIONS = [None]

    if "{dataset}" in dataset_info["main_root"]:
        main_root = dataset_info["main_root"].replace(
            "{dataset}", dataset_info["dataset"]
        )
    else:
        main_root = dataset_info["main_root"]

    if "{main_root}" in dataset_info["roi_root"]:
        roi_root = dataset_info["roi_root"].replace("{main_root}", main_root)
    else:
        roi_root = dataset_info["roi_root"]

    if "{main_root}" in dataset_info["dFC_root"]:
        dFC_root = dataset_info["dFC_root"].replace("{main_root}", main_root)
    else:
        dFC_root = dataset_info["dFC_root"]

    if "{main_root}" in dataset_info["ML_root"]:
        ML_root = dataset_info["ML_root"].replace("{main_root}", main_root)
    else:
        ML_root = dataset_info["ML_root"]

    # # The task feature extraction will be executed multiple times in parallel redundantly
    # try:
    #     run_task_features_extraction(
    #         TASKS=TASKS,
    #         RUNS=RUNS,
    #         SESSIONS=SESSIONS,
    #         roi_root=roi_root,
    #         dFC_root=dFC_root,
    #         output_root=ML_root,
    #     )
    # except Exception as e:
    #     print(f"Error in task features extraction: {e}")
    #     traceback.print_exc()
    # print("Task features extraction finished.")

    job_id = os.getenv("SGE_TASK_ID")  # for SGE
    if job_id is None:
        job_id = os.getenv("SLURM_ARRAY_TASK_ID")  # for SLURM
    job_id = int(job_id)
    dFC_id = job_id - 1  # TASK_ID starts from 1 not 0

    print(f"Task presence classification started for dFC ID {dFC_id}...")
    try:
        run_classification(
            dFC_id=dFC_id,
            TASKS=TASKS,
            RUNS=RUNS,
            SESSIONS=SESSIONS,
            roi_root=roi_root,
            dFC_root=dFC_root,
            output_root=ML_root,
            dynamic_pred="no",
            normalize_dFC=False,
            n_jobs=8,
        )
    except Exception as e:
        print(f"Error in classification for dFC ID {dFC_id}: {e}")
        traceback.print_exc()
    print(f"Task presence classification finished for dFC ID {dFC_id}.")

    print(f"Task presence prediction finished for dFC ID {dFC_id}.")

#######################################################################################
