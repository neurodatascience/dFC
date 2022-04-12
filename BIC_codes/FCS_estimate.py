from functions.dFC_funcs import *
import numpy as np
import time
import hdf5storage
import scipy.io as sio
import os
os.environ["MKL_NUM_THREADS"] = '64'
os.environ["NUMEXPR_NUM_THREADS"] = '64'
os.environ["OMP_NUM_THREADS"] = '64'

print('################################# CODE started running ... #################################')

################################# Parameters #################################

###### DATA PARAMETERS ######

output_root = './../../../../../RESULTs/methods_implementation/'
# output_root = '/data/origami/dFC/RESULTs/methods_implementation/'
# output_root = '/Users/mte/Documents/McGill/Project/dFC/RESULTs/methods_implementation/'

# DATA_type is either 'sample' or 'Gordon' or 'simulated' or 'ICA'
params_data_load = { \
    'DATA_type': 'Gordon', \
    # 'normalization': True, \
    # 'num_subj': 10, \
    # 'select_nodes': True, \
    # 'rand_node_slct': False, \
    # 'num_select_nodes': 50, \
    # 'num_time_point': 1200, \
    # 'Fs_ratio': 1.00, \
    # 'noise_ratio': 0.00, \

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
    'n_jobs': 2, 'verbose': 0, 'backend': 'loky', \
    # Hyper Parameters
    'normalization': True, \
    'num_subj': 10, \
    'num_select_nodes': 100, \
    'num_time_point': 1200, \
    'Fs_ratio': 1.00, \
    'noise_ratio': 1.00, \
    'num_realization': 1 \
}

###### HYPER PARAMETERS ALTERNATIVE ######

MEASURES_name_lst = [ \
                'SlidingWindow', \
                'Time-Freq', \
                'ContinuousHMM', \
                'Windowless', \
                'Clustering', \
                'DiscreteHMM' \
                ]

alter_hparams = { \
            'n_states': [6, 12, 16], \
            'normalization': [True], \
            'num_subj': [50, 100, 395], \
            'num_select_nodes': [50, 100, 333], \
            'num_time_point': [500, 800, 1200], \
            'Fs_ratio': [0.50, 1.00, 1.50], \
            'noise_ratio': [0.00, 0.50, 1.00], \
            'num_realization': [1, 2, 3] \
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
    # Methods Parameters
    'params_methods': params_methods, \
    'methods': MEASURES_name_lst, \
    # hyper parameters alternative values
    'alter_hparams': alter_hparams, \
    # Similarity Assessment Parameters
    'sim_assess_params': sim_assess_params, \
    # Dynamic Connection Detector Parameters
    'dyn_conn_det_params': dyn_conn_det_params \
}


################################# LOAD DATA #################################

data_loader = DATA_LOADER(**params_data_load)
BOLD = data_loader.load()

################################# Visualize BOLD #################################

for session in BOLD:
    BOLD[session].visualize(start_time=0, end_time=50, nodes_lst=list(range(10)), \
        save_image=params_dFC_analyzer['save_image'], output_root=output_root+'BOLD_signal_'+session)

################################# Measures of dFC #################################

dFC_analyzer = DFC_ANALYZER( \
    analysis_name='reproducibility assessment', \
    **params_dFC_analyzer \
)

tic = time.time()
print('Measurement Started ...')

################################# estimate FCS #################################

print("FCS estimation started...")
dFC_analyzer.estimate_group_FCS(time_series_dict=BOLD)
print("FCS estimation done.")

print('Measurement required %0.3f seconds.' % (time.time() - tic, ))

### Visualize FCS ###

# dFC_analyzer.visualize_FCS(normalize=True, \
#                         threshold=0.0, \
#                         )

# Save
np.save('./dFC_analyzer.npy', dFC_analyzer) 
np.save('./data_loader.npy', data_loader) 

# ################################# SIMILARITY ASSESSMENT #################################

# # print_dict(dFC_analyzer.methods_corr)
# dFC_analyzer.similarity_analyze(SUBJs_dFC_session_sim_dict)

# ################################# STATE MATCH #################################

# state_match = dFC_analyzer.state_match()
# # Save
# np.save('./state_match.npy', state_match) 
