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
    'SESSIONs':['Rest1_LR' , 'Rest1_RL', 'Rest2_LR', 'Rest2_RL'], \

    'data_root_simul': './../../../../DATA/TVB data/', \
    'data_root_sample': './sampleDATA/', \
    'data_root_gordon': './../../../../DATA/HCP/HCP_Gordon/', \
    'data_root_ica': './../../../../DATA/HCP/HCP_PTN1200/node_timeseries/3T_HCP1200_MSMAll_d50_ts2/'
}

###### MEASUREMENT PARAMETERS ######

# W is in sec

params_methods = { \
    # Sliding Parameters
    'W': 44, 'n_overlap': 0.5, 'sw_method':'pear_corr', 'tapered_window':True, \
    # TIME_FREQ
    'TF_method':'WTC', \
    # CLUSTERING AND DHMM
    'clstr_base_measure':'SlidingWindow', \
    # HMM
    'hmm_iter': 50, 'n_hid_states': 24, \
    # State Parameters
    'n_states': 12, 'n_subj_clstrs': 20, \
    # Parallelization Parameters
    'n_jobs': 2, 'verbose': 0, 'backend': 'loky', \
    # SESSION
    'session': 'Rest1_LR', \
    # Hyper Parameters
    'normalization': True, \
    'num_subj': 100, \
    'num_select_nodes': 100, \
    'num_time_point': 1200, \
    'Fs_ratio': 1.00, \
    'noise_ratio': 0.00, \
    'num_realization': 1 \
}

###### HYPER PARAMETERS ALTERNATIVE ######

MEASURES_name_lst = [ \
                'SlidingWindow', \
                'Time-Freq', \
                'CAP', \
                'ContinuousHMM', \
                'Windowless', \
                'Clustering', \
                'DiscreteHMM' \
                ]

alter_hparams = { \
            # 'session': [], \
            'n_states': [6], \
            # 'normalization': [], \
            # 'num_subj': [5], \
            'num_select_nodes': [50], \
            # 'num_time_point': [500], \
            'Fs_ratio': [0.50], \
            'noise_ratio': [2.00], \
            # 'num_realization': [] \
            }

###### dFC ANALYZER PARAMETERS ######

params_dFC_analyzer = { \
    # VISUALIZATION
    'vis_TR_idx': list(range(10, 20, 1)),'save_image': True, 'output_root': output_root, \
    # Parallelization Parameters
    'n_jobs': None, 'verbose': 0, 'backend': 'loky' \
}


################################# LOAD DATA #################################

data_loader = DATA_LOADER(**params_data_load)
BOLD = data_loader.load()

################################# Visualize BOLD #################################

# for session in BOLD:
#     BOLD[session].visualize(start_time=0, end_time=50, nodes_lst=list(range(10)), \
#         save_image=params_dFC_analyzer['save_image'], output_root=output_root+'BOLD_signal_'+session)

################################# Measures of dFC #################################

dFC_analyzer = DFC_ANALYZER( \
    analysis_name='reproducibility assessment', \
    **params_dFC_analyzer \
)

MEASURES_lst = dFC_analyzer.measures_initializer( \
    MEASURES_name_lst, \
    params_methods, \
    alter_hparams \
    )

tic = time.time()
print('Measurement Started ...')

################################# estimate FCS #################################

task_id = int(os.getenv("SGE_TASK_ID"))
MEASURE_id = task_id-1 # SGE_TASK_ID starts from 1 not 0


if MEASURE_id >= len(MEASURES_lst):
    print("MEASURE_id out of MEASURES_lst ")
else:
    measure = MEASURES_lst[MEASURE_id]

    print("FCS estimation started...")

    time_series = BOLD[measure.params['session']]
    if measure.is_state_based:
        measure.estimate_FCS(time_series=time_series)
            
    # dFC_analyzer.estimate_group_FCS(time_series_dict=BOLD)
    print("FCS estimation done.")

    print('Measurement required %0.3f seconds.' % (time.time() - tic, ))

    # Save
    np.save('./fitted_MEASURES/MEASURE_'+str(MEASURE_id)+'.npy', measure) 
    np.save('./dFC_analyzer.npy', dFC_analyzer) 
    np.save('./data_loader.npy', data_loader) 

#################################################################################