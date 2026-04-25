import argparse
import json
import os
import time
import traceback
import warnings

import numpy as np

from pydfc import data_loader, multi_analysis_utils

warnings.simplefilter("ignore")

########################################################################################


def run_FCS_estimate(
    params_methods,
    MEASURES_name_lst,
    alter_hparams,
    params_multi_analysis,
    task,
    roi_root,
    output_root,
    session=None,
    run=None,
):
    if session is None:
        output_dir = f"{output_root}"
    else:
        output_dir = f"{output_root}/{session}"

    if run is None:
        print(f"TASK: {task} started ...")
        if session is None:
            BOLD_file_name = "{subj_id}_{task}_time-series.npy"
            file_suffix = f"{task}"
        else:
            BOLD_file_name = "{subj_id}_{session}_{task}_time-series.npy"
            file_suffix = f"{session}_{task}"
    else:
        print(f"TASK: {task}, RUN: {run} started ...")
        if session is None:
            BOLD_file_name = "{subj_id}_{task}_{run}_time-series.npy"
            file_suffix = f"{task}_{run}"
        else:
            BOLD_file_name = "{subj_id}_{session}_{task}_{run}_time-series.npy"
            file_suffix = f"{session}_{task}_{run}"
    ################################# LOAD DATA #################################
    BOLD = data_loader.load_TS(
        data_root=roi_root,
        file_name=BOLD_file_name,
        subj_id2load=None,
        task=task,
        session=session,
        run=run,
    )

    if BOLD is None:
        print(f"No BOLD data found for task: {task}, session: {session}, run: {run}.")
        return
    ################################ Measures of dFC #################################

    MEASURES_lst, hyper_param_info = multi_analysis_utils.measures_initializer(
        MEASURES_name_lst, params_methods, alter_hparams
    )

    # in this script we process only one measure
    # if alter_hparams is not empty, we need to change the naming of the output files
    # to differentiate between the measures
    if len(MEASURES_lst) == 1:
        only_one_measure = True
        n_jobs = None
    else:
        only_one_measure = False
        n_jobs = params_multi_analysis["n_jobs"]

    if not only_one_measure:
        # we assume only one hyperparameter is altered
        # alter_hparams is a dictionary with one key
        # ow change the naming of the output files
        assert len(alter_hparams) == 1, (
            "alter_hparams should have only one key, "
            "but got more than one. This script is designed to process only one hyperparameter."
        )
        hyper_param_name = [key for key in alter_hparams.keys()][0]

    tic = time.time()
    print("Measurement Started ...")

    ################################# estimate FCS #################################

    MEASURES_fit_lst = multi_analysis_utils.estimate_group_FCS(
        time_series=BOLD,
        MEASURES_lst=MEASURES_lst,
        n_jobs=n_jobs,
        verbose=params_multi_analysis["verbose"],
        backend=params_multi_analysis["backend"],
    )

    if only_one_measure:
        assert (
            len(MEASURES_fit_lst) == 1
        ), "Only one measure should be processed, but got more than one."

    # Save the fitted measures
    for measure in MEASURES_fit_lst:
        try:
            if not os.path.exists(f"{output_dir}"):
                os.makedirs(f"{output_dir}")
        except OSError as err:
            print(err)
        if only_one_measure:
            measure_name = measure.measure_name
        else:
            measure_name = f"{measure.measure_name}-{hyper_param_name}-{measure.params[hyper_param_name]}"
        np.save(f"{output_dir}/MEASURE_{file_suffix}_{measure_name}.npy", measure)

    print(f"Measurement required {time.time() - tic:0.3f} seconds.")


########################################################################################

if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to fit dFC methods for a given task.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument("--dataset_info", type=str, help="path to dataset info file")
    parser.add_argument("--methods_config", type=str, help="methods config file")

    args = parser.parse_args()

    dataset_info_file = args.dataset_info
    methods_config_file = args.methods_config

    # Read dataset info
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    # Read methods config
    with open(methods_config_file, "r") as f:
        methods_config = json.load(f)

    TASKS = dataset_info["TASKS"]

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

    # methods params
    params_methods = methods_config["params_methods"]
    MEASURES_name_lst = methods_config["MEASURES_name_lst"]
    alter_hparams = methods_config["alter_hparams"]
    params_multi_analysis = methods_config["params_multi_analysis"]

    # pick one method
    job_id = os.getenv("SGE_TASK_ID")  # for SGE
    if job_id is None:
        job_id = os.getenv("SLURM_ARRAY_TASK_ID")  # for SLURM
    job_id = int(job_id)
    MEASURE_id = job_id - 1  # job_id starts from 1 not 0
    if MEASURE_id >= len(MEASURES_name_lst):
        print("MEASURE_id out of MEASURES_name_lst range")
        exit()
    picked_measure_list = [MEASURES_name_lst[MEASURE_id]]  # pick one method but as a list

    print(
        f"FCS estimation CODE started running ... for measure: {picked_measure_list[0]} ..."
    )

    for session in SESSIONS:
        for task in TASKS:
            for run in RUNS[task]:
                try:
                    run_FCS_estimate(
                        params_methods=params_methods,
                        MEASURES_name_lst=picked_measure_list,
                        alter_hparams=alter_hparams,
                        params_multi_analysis=params_multi_analysis,
                        task=task,
                        roi_root=roi_root,
                        output_root=fitted_measures_root,
                        session=session,
                        run=run,
                    )
                except Exception as e:
                    print(
                        f"Error in run_FCS_estimate for task: {task}, session: {session}, run: {run}, measure: {picked_measure_list[0]}, error: {e}"
                    )
                    traceback.print_exc()

    print(
        f"FCS estimation CODE finished running ... for measure: {picked_measure_list[0]} ..."
    )
#################################################################################
