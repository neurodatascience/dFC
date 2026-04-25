# -*- coding: utf-8 -*-
"""
Created on Wed March 20 2024

@author: mte
"""
import argparse
import json
import os
import traceback
import warnings

import numpy as np
from tvb.simulator.lab import *

from pydfc import simul_utils

warnings.simplefilter("ignore")

os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"
################################# Parameters ####################################

# argparse
HELPTEXT = """
Script to simulate task-based data.
"""
parser = argparse.ArgumentParser(description=HELPTEXT)

parser.add_argument("--dataset_info", type=str, help="path to dataset info file")
parser.add_argument("--tasks_info", type=str, help="path to tasks info file")
parser.add_argument("--participant_id", type=str, help="participant id")

args = parser.parse_args()

dataset_info_file = args.dataset_info
tasks_info_file = args.tasks_info
participant_id = args.participant_id

# Read dataset info
with open(dataset_info_file, "r") as f:
    dataset_info = json.load(f)

if "{dataset}" in dataset_info["main_root"]:
    main_root = dataset_info["main_root"].replace("{dataset}", dataset_info["dataset"])
else:
    main_root = dataset_info["main_root"]

if "{main_root}" in dataset_info["roi_root"]:
    output_root = dataset_info["roi_root"].replace("{main_root}", main_root)
else:
    output_root = dataset_info["roi_root"]

# Read tasks info
with open(tasks_info_file, "r") as f:
    all_tasks_info = json.load(f)

print(f"subject-level simulation started running ... for subject: {participant_id} ...")

for task in all_tasks_info:

    # the task_data file might not exist for some subjects, so we use a try-except block
    try:
        time_series, task_data = simul_utils.simulate_task_data(
            participant_id, all_tasks_info[task]
        )
    except Exception as e:
        print(f"Error simulating task {task} for participant {participant_id}: {e}")
        # print traceback
        traceback.print_exc()
        continue

    # save the time series and task data
    output_file_prefix = f"{participant_id}_{task}"
    if not os.path.exists(f"{output_root}/{participant_id}/"):
        os.makedirs(f"{output_root}/{participant_id}/")
    np.save(
        f"{output_root}/{participant_id}/{output_file_prefix}_time-series.npy",
        time_series,
    )
    np.save(
        f"{output_root}/{participant_id}/{output_file_prefix}_task-data.npy", task_data
    )

print("****************** DONE ******************")
####################################################################################
