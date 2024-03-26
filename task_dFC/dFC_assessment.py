import os
import time
import warnings
from calendar import c

import numpy as np

from pydfc import MultiAnalysis, data_loader

warnings.simplefilter("ignore")

os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"

################################# Parameters #################################

# Data parameters
# main_root = '../../DATA/ds002785/' # for local
main_root = "../../../DATA/task-based/openneuro/ds002785/"  # for server

# subjects used for dFC assessment do not need to be the same as those used for FCS_estimate
# you can set the new roi root and data load parameters here:
roi_root = f"{main_root}/derivatives/ROI_timeseries"
fitted_measures_root = f"{main_root}/derivatives/fitted_MEASURES"
output_root = f"{main_root}/derivatives/dFC_assessed"

# for consistency we use 0 for resting state
TASKS = [
    "task-restingstate",
    "task-anticipation",
    "task-emomatching",
    "task-faces",
    "task-gstroop",
    "task-workingmemory",
]

# find all subjects across all tasks
SUBJECTS = data_loader.find_subj_list(data_root=roi_root, sessions=TASKS)

# job_id selects the subject
job_id = int(os.getenv("SGE_TASK_ID"))
if job_id > len(SUBJECTS):
    print("job_id > len(SUBJECTS)")
    exit()
subj_id = SUBJECTS[job_id - 1]  # SGE_TASK_ID starts from 1 not 0

for task in TASKS:

    MA = np.load(
        f"{fitted_measures_root}/{task}/multi_analysis.npy", allow_pickle="TRUE"
    ).item()

    # check if the subject has this task
    SUBJECTS_with_this_task = data_loader.find_subj_list(
        data_root=roi_root, sessions=[task]
    )
    if not subj_id in SUBJECTS_with_this_task:
        print(f"subject {subj_id} not in the list of subjects with task {task}")
        continue

    ################################# LOAD FIT MEASURES #################################

    ALL_RECORDS = os.listdir(f"{fitted_measures_root}/{task}/")
    ALL_RECORDS = [i for i in ALL_RECORDS if "MEASURE" in i]
    ALL_RECORDS.sort()
    MEASURES_fit_lst = list()
    for s in ALL_RECORDS:
        fit_measure = np.load(
            f"{fitted_measures_root}/{task}/{s}", allow_pickle="TRUE"
        ).item()
        MEASURES_fit_lst.append(fit_measure)
    MA.set_MEASURES_fit_lst(MEASURES_fit_lst)
    print("fitted MEASURES loaded ...")

    ################################# LOAD DATA #################################

    print(
        f"subject-level dFC assessment CODE started running ... for task {task} of subject {subj_id} ..."
    )

    BOLD = data_loader.load_TS(
        data_root=roi_root,
        file_name="time_series.npy",
        SESSIONs=[task],
        subj_id2load=subj_id,
    )

    ################################# dFC ASSESSMENT #################################

    tic = time.time()
    print("Measurement Started ...")

    print("dFC estimation started...")
    dFC_dict = MA.subj_lvl_dFC_assess(time_series_dict=BOLD)
    print("dFC estimation done.")

    print(f"Measurement required {time.time() - tic:0.3f} seconds.")

    ################################# SAVE DATA #################################

    folder = f"{output_root}/{task}/{subj_id}"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for dFC_id, dFC in enumerate(dFC_dict["dFC_lst"]):
        np.save(f"{folder}/dFC_{str(dFC_id)}.npy", dFC)

#######################################################################################
