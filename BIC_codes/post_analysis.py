from functions.dFC_funcs import *
import numpy as np
import time
import hdf5storage
import scipy.io as sio
import os
os.environ["MKL_NUM_THREADS"] = '64'
os.environ["NUMEXPR_NUM_THREADS"] = '64'
os.environ["OMP_NUM_THREADS"] = '64'

################################# Parameters #################################

subj_id = 1

###### DATA PARAMETERS ######

output_root = './../../../../../RESULTs/methods_implementation/'
# output_root = '/data/origami/dFC/RESULTs/methods_implementation/'
# output_root = '/Users/mte/Documents/McGill/Project/dFC/RESULTs/methods_implementation/'

# DATA_type is either 'sample' or 'Gordon' or 'simulated' or 'ICA'
params_data_load = { \
    'DATA_type': 'Gordon', \
    'num_subj': 2, \
    'select_nodes': True, \
    'rand_node_slct': False, \
    'num_select_nodes': 50, \

    'data_root_simul': './../../../../DATA/TVB data/', \
    'data_root_sample': './sampleDATA/', \
    'data_root_gordon': './../../../../DATA/HCP/HCP_Gordon/', \
    'data_root_ica': './../../../../DATA/HCP/HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d50_ts2/'
}

###### MEASUREMENT PARAMETERS ######

# W is in sec

params_methods = { \
    # Sliding Parameters
    'W': 44, 'n_overlap': 0.5, \
    # State Parameters
    'n_states': 6, 'n_subj_clstrs': 20, 'n_hid_states': 4, \
    # Parallelization Parameters
    'n_jobs': 2, 'verbose': 0, 'backend': 'loky' \
}

###### SIMILARITY PARAMETERS ######

sim_assess_params= { \
    'run_analysis': True, \
    'num_samples': 100, \
    'matching_method': 'score', \
    'n_jobs': 8, 'backend': 'loky' \
}

###### SIMILARITY PARAMETERS ######

dyn_conn_det_params = { \
    'run_analysis': False, \
    'N': 30, 'L': 1200, 'p': 100, \
    'n_jobs': 8, 'backend': 'loky' \
}

###### dFC ANALYZER PARAMETERS ######

params_dFC_analyzer = { \
    # VISUALIZATION
    'vis_TR_idx': list(range(10, 20, 1)),'save_image': True, 'output_root': output_root, \
    # Parallelization Parameters
    'n_jobs': 8, 'verbose': 0, 'backend': 'loky', \
    # Similarity Assessment Parameters
    'sim_assess_params': sim_assess_params, \
    # Dynamic Connection Detector Parameters
    'dyn_conn_det_params': dyn_conn_det_params \
}

################################# LOAD DATA #################################

data_loader = DATA_LOADER(**params_data_load)
BOLD = data_loader.load(subj_id2load=subj_id)

dFC_analyzer = np.load('./dFC_analyzer.npy',allow_pickle='TRUE')

################################# POST ANALYSIS #################################

dFC_analyzer.post_analyze()