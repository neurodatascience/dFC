import argparse
import json
import os
import time
import warnings

import numpy as np

from pydfc import data_loader, multi_analysis_utils

warnings.simplefilter("ignore")

os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"

################################# Functions #################################


def run_dFC_assess(
    subj_id,
    task,
    roi_root,
    fitted_measures_root,
    output_root,
    params_multi_analysis,
    session=None,
    run=None,
):
    if session is None:
        output_dir = f"{output_root}/{subj_id}"
        fitted_measures_dir = f"{fitted_measures_root}"
    else:
        output_dir = f"{output_root}/{subj_id}/{session}"
        fitted_measures_dir = f"{fitted_measures_root}/{session}"

    if run is None:
        if session is None:
            print(f"Subject-level dFC assessment started for TASK: {task} ...")
            input_root = f"{roi_root}/{subj_id}"
            BOLD_file_name = "{subj_id}_{task}_time-series.npy"
            file_suffix = f"{task}"
        else:
            print(
                f"Subject-level dFC assessment started for Session {session}, TASK: {task} ..."
            )
            input_root = f"{roi_root}/{subj_id}/{session}"
            BOLD_file_name = "{subj_id}_{session}_{task}_time-series.npy"
            file_suffix = f"{session}_{task}"
    else:
        if session is None:
            print(
                f"Subject-level dFC assessment started for TASK: {task}, RUN: {run} ..."
            )
            input_root = f"{roi_root}/{subj_id}"
            BOLD_file_name = "{subj_id}_{task}_{run}_time-series.npy"
            file_suffix = f"{task}_{run}"
        else:
            print(
                f"Subject-level dFC assessment started for Session {session}, TASK: {task}, RUN: {run} ..."
            )
            input_root = f"{roi_root}/{subj_id}/{session}"
            BOLD_file_name = "{subj_id}_{session}_{task}_{run}_time-series.npy"
            file_suffix = f"{session}_{task}_{run}"

    # check if the subject has this task in roi_root
    if not os.path.exists(input_root):
        print(f"{input_root} not found in {roi_root}")
        return

    ALL_ROI_FILES = os.listdir(f"{input_root}/")
    ALL_ROI_FILES = [
        roi_file
        for roi_file in ALL_ROI_FILES
        if ("_time-series.npy" in roi_file) and (f"_{task}_" in roi_file)
    ]
    if session is not None:
        ALL_ROI_FILES = [
            roi_file for roi_file in ALL_ROI_FILES if (f"_{session}_" in roi_file)
        ]
    if run is not None:
        ALL_ROI_FILES = [
            roi_file for roi_file in ALL_ROI_FILES if (f"_{run}_" in roi_file)
        ]
    ALL_ROI_FILES.sort()

    # if there are no files for this task, return
    if not len(ALL_ROI_FILES) >= 1:
        print(f"No time series files found for {subj_id} {file_suffix}")
        return
    ################################# LOAD FIT MEASURES #################################

    ALL_RECORDS = os.listdir(f"{fitted_measures_dir}/")
    ALL_RECORDS = [
        i for i in ALL_RECORDS if ("MEASURE" in i) and (f"_{file_suffix}_" in i)
    ]
    ALL_RECORDS.sort()
    MEASURES_fit_lst = list()
    for s in ALL_RECORDS:
        fit_measure = np.load(f"{fitted_measures_dir}/{s}", allow_pickle="TRUE").item()
        MEASURES_fit_lst.append(fit_measure)
    print("fitted MEASURES are loaded ...")

    ################################# LOAD DATA #################################

    BOLD = data_loader.load_TS(
        data_root=roi_root,
        file_name=BOLD_file_name,
        subj_id2load=subj_id,
        task=task,
        session=session,
        run=run,
    )

    ################################# dFC ASSESSMENT #################################

    tic = time.time()
    print("Measurement Started ...")

    print("dFC estimation started...")
    dFC_dict = multi_analysis_utils.subj_lvl_dFC_assess(
        time_series=BOLD,
        MEASURES_fit_lst=MEASURES_fit_lst,
        n_jobs=params_multi_analysis["n_jobs"],
        verbose=params_multi_analysis["verbose"],
        backend=params_multi_analysis["backend"],
    )
    print("dFC estimation done.")

    print(f"Measurement required {time.time() - tic:0.3f} seconds.")

    ################################# SAVE DATA #################################

    folder = f"{output_dir}/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for dFC_id, dFC in enumerate(dFC_dict["dFC_lst"]):

        # Optional: cast each dFC to float32 to save space
        dFC.FCSs_ = {
            key: value.astype(np.float32, copy=False) for key, value in dFC.FCSs_.items()
        }

        np.save(f"{folder}dFC_{file_suffix}_{dFC_id}.npy", dFC)


#######################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to assess dFC for a given participant.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument("--dataset_info", type=str, help="path to dataset info file")
    parser.add_argument("--methods_config", type=str, help="methods config file")
    parser.add_argument("--participant_id", type=str, help="participant id")

    args = parser.parse_args()

    dataset_info_file = args.dataset_info
    methods_config_file = args.methods_config
    participant_id = args.participant_id

    # Read dataset info
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    # Read methods config
    with open(methods_config_file, "r") as f:
        methods_config = json.load(f)

    print(
        f"subject-level dFC assessment CODE started running ... for subject: {participant_id} ..."
    )

    TASKS = dataset_info["TASKS"]

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

    if "{main_root}" in dataset_info["fitted_measures_root"]:
        fitted_measures_root = dataset_info["fitted_measures_root"].replace(
            "{main_root}", main_root
        )
    else:
        fitted_measures_root = dataset_info["fitted_measures_root"]

    if "{main_root}" in dataset_info["dFC_root"]:
        output_root = dataset_info["dFC_root"].replace("{main_root}", main_root)
    else:
        output_root = dataset_info["dFC_root"]

    if "SESSIONS" in dataset_info:
        SESSIONS = dataset_info["SESSIONS"]
    else:
        SESSIONS = None
    if SESSIONS is None:
        SESSIONS = [None]

    if "RUNS" in dataset_info:
        RUNS = dataset_info["RUNS"]
    else:
        RUNS = None
    if RUNS is None:
        RUNS = {task: [None] for task in TASKS}

    params_multi_analysis = methods_config["params_multi_analysis"]

    for session in SESSIONS:
        for task in TASKS:
            for run in RUNS[task]:
                try:
                    run_dFC_assess(
                        subj_id=participant_id,
                        task=task,
                        roi_root=roi_root,
                        fitted_measures_root=fitted_measures_root,
                        output_root=output_root,
                        params_multi_analysis=params_multi_analysis,
                        session=session,
                        run=run,
                    )
                except Exception as e:
                    print(
                        f"Error in dFC assessment for subject {participant_id}, task {task}, session {session}, run {run}: {e}"
                    )
                    continue

    print(
        f"subject-level dFC assessment CODE finished running for subject: {participant_id}"
    )

#######################################################################################
