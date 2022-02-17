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

dFC_analyzer = np.load('./dFC_analyzer.npy',allow_pickle='TRUE').item()
data_loader = np.load('./data_loader.npy',allow_pickle='TRUE').item()

BOLD = data_loader.load(subj_id2load=subj_id)

################################# SIMILARITY ASSESSMENT #################################

tic = time.time()
print('Measurement Started ...')

print("dFCM estimation started...")
SUBJs_dFC_session_sim_dict = dFC_analyzer.estimate_all_dFCM(time_series_dict=BOLD)
print("dFCM estimation done.")

print('Measurement required %0.3f seconds.' % (time.time() - tic, ))

#### Methods dFC Corr MAT ###

fig_name = None
if dFC_analyzer.params['save_image']:
    output_root = dFC_analyzer.params['output_root']+'dFC/'
    fig_name = output_root + 'avg_dFC_corr' 

visualize_conn_mat(dFC_analyzer.methods_corr, \
    title='intra session dFC correlation', \
    name_lst_key='measure_lst', mat_key='corr_mat', \
    cmap='viridis',\
    save_image=dFC_analyzer.params['save_image'], output_root=fig_name, \
        fix_lim=True \
)

# Save
np.save('./dFC_session_sim.npy', SUBJs_dFC_session_sim_dict) 



################################# SIMILARITY ASSESSMENT #################################

# print_dict(dFC_analyzer.methods_corr)
dFC_analyzer.similarity_analyze(SUBJs_dFC_session_sim_dict)

################################# STATE MATCH #################################

state_match = dFC_analyzer.state_match()
# Save
np.save('./state_match.npy', state_match) 

################################# POST ANALYSIS #################################

dFC_analyzer.post_analyze()