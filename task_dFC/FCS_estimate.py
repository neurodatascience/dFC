from pydfc import (
    data_loader,
    MultiAnalysis,
)
import time
import numpy as np
import os
import warnings

warnings.simplefilter('ignore')

os.environ["MKL_NUM_THREADS"] = '16'
os.environ["NUMEXPR_NUM_THREADS"] = '16'
os.environ["OMP_NUM_THREADS"] = '16'

################################# Parameters #################################
# data paths
# main_root = '../../DATA/ds002785/' # for local
main_root = '../../../DATA/task-based/openneuro/ds002785' # for server
roi_root = f"{main_root}/derivatives/ROI_timeseries"
output_root = f"{main_root}/derivatives/fitted_MEASURES"

# for consistency we use 0 for resting state
TASKS = [
    'task-restingstate', 
    'task-anticipation', 
    'task-emomatching', 
    'task-faces', 
    'task-gstroop', 
    'task-workingmemory'
]

job_id = int(os.getenv("SGE_TASK_ID"))
TASK_id = job_id-1 # SGE_TASK_ID starts from 1 not 0
if TASK_id >= len(TASKS):
    print("TASK_id out of TASKS")
    exit()
task = TASKS[TASK_id]

###### MEASUREMENT PARAMETERS ######

# W is in sec

params_methods = { 
    # Sliding Parameters
    'W': 44, 'n_overlap': 1.0, 'sw_method':'pear_corr', 'tapered_window':True, 
    # TIME_FREQ
    'TF_method':'WTC', 
    # CLUSTERING AND DHMM
    'clstr_base_measure':'SlidingWindow', 
    # HMM
    'hmm_iter': 20, 'dhmm_obs_state_ratio': 16/24, 
    # State Parameters
    'n_states': 12, 'n_subj_clstrs': 20, 
    # Parallelization Parameters
    'n_jobs': 2, 'verbose': 0, 'backend': 'loky', 
    # SESSION
    'session': task, 
    # Hyper Parameters
    'normalization': True, 
    'num_subj': None, # None or 216?
    'num_time_point': None,  # None or set?
}

###### HYPER PARAMETERS ALTERNATIVE ######

MEASURES_name_lst = [ 
    'SlidingWindow', 
    'Time-Freq', 
    'CAP', 
    'ContinuousHMM', 
    'Windowless', 
    'Clustering', 
    'DiscreteHMM' 
]

alter_hparams = { 
    # 'session': ['Rest1_RL', 'Rest2_LR', 'Rest2_RL'], 
    # 'n_overlap': [0, 0.25, 0.75, 1], 
    # 'n_states': [6, 16], 
    # # 'normalization': [], 
    # 'num_subj': [50, 100, 200], 
    # 'num_select_nodes': [30, 50, 333], 
    # 'num_time_point': [800, 1000], 
    # 'Fs_ratio': [0.50, 0.75, 1.5], 
    # 'noise_ratio': [1.00, 2.00, 3.00], 
    # 'num_realization': [] 
}

###### MultiAnalysis PARAMETERS ######

params_multi_analysis = { 
    # Parallelization Parameters
    'n_jobs': None, 'verbose': 0, 'backend': 'loky' 
}

################################# LOAD DATA #################################

BOLD = data_loader.load_TS(
        data_root=roi_root, 
        file_name='time_series.npy', 
        SESSIONs=task, 
        subj_id2load=None
)

################################# Visualize BOLD #################################

# for session in BOLD:
#     BOLD.visualize(start_time=0, end_time=2000, nodes_lst=list(range(10)),
#         save_image=False, output_root=None)

################################ Measures of dFC #################################

MA = MultiAnalysis( 
    analysis_name=f"task-based-dFC-ds002785-{task}",
    **params_multi_analysis 
)

MEASURES_lst = MA.measures_initializer(
    MEASURES_name_lst, 
    params_methods, 
    alter_hparams 
)

tic = time.time()
print('Measurement Started ...')

################################# estimate FCS #################################

for MEASURE_id, measure in enumerate(MEASURES_lst):

    print('MEASURE: ' + measure.measure_name)
    print("FCS estimation started...")

    if measure.is_state_based:
        measure.estimate_FCS(time_series=BOLD)
            
    # dFC_analyzer.estimate_group_FCS(time_series_dict=BOLD)
    print("FCS estimation done.")

    # Save
    if not os.path.exists(f"{output_root}/{task}"):
        os.makedirs(f"{output_root}/{task}")
    np.save(f"{output_root}/{task}/MEASURE_{str(MEASURE_id)}.npy", measure)

print('Measurement required %0.3f seconds.' % (time.time() - tic, ))
np.save(f"{output_root}/{task}/multi_analysis.npy", MA)

#################################################################################
