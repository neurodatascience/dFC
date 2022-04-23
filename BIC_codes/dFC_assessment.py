from functions.dFC_funcs import *
import numpy as np
import time
import hdf5storage
import scipy.io as sio
import os
os.environ["MKL_NUM_THREADS"] = '64'
os.environ["NUMEXPR_NUM_THREADS"] = '64'
os.environ["OMP_NUM_THREADS"] = '64'

print('################################# subject-level dFC assessment CODE started running ... #################################')

################################# Parameters #################################

# subj_id = '100206'

###### DATA PARAMETERS ######

output_root = './../../../../../RESULTs/methods_implementation/'
# output_root = '/data/origami/dFC/RESULTs/methods_implementation/'
# output_root = '/Users/mte/Documents/McGill/Project/dFC/RESULTs/methods_implementation/'

################################# LOAD DATA #################################

dFC_analyzer = np.load('./dFC_analyzer.npy',allow_pickle='TRUE').item()
data_loader = np.load('./data_loader.npy',allow_pickle='TRUE').item()

task_id = int(os.getenv("SGE_TASK_ID"))
subj_id = data_loader.SUBJECTS[task_id-1] # SGE_TASK_ID starts from 1 not 0

BOLD = data_loader.load(subj_id2load=subj_id)

################################# SIMILARITY ASSESSMENT #################################

tic = time.time()
print('Measurement Started ...')

print("dFCM estimation started...")
dFCM_dict = dFC_analyzer.subj_lvl_dFC_assess(time_series_dict=BOLD)
# SUBJ_output = dFC_analyzer.group_dFCM_assess(time_series_dict=BOLD)
print("dFCM estimation done.")

print('Measurement required %0.3f seconds.' % (time.time() - tic, ))


################################# POST ANALYSIS #################################

SUBJ_output = {}

dFCM_lst = dFCM_dict['dFCM_lst']

########################## DEFAULT VALUES #######################

param_dict = dFC_analyzer.params_methods
analysis_name_lst = [ \
    'corr_mat', \
    'dFC_distance', \
    'dFC_distance_var', \
    'FO', \
    'CO', \
    'TP', \
    'trans_freq' \
    ]
dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
SUBJ_output['default_values'] = dFC_analyzer.post_analysis( \
    dFCM_lst=dFCM_lst2check, \
    analysis_name_lst=analysis_name_lst \
    )

########################## 6_states #######################

param_dict = {'n_states': 6, 'is_state_based': True}
analysis_name_lst = [ \
    'corr_mat', \
    'dFC_distance', \
    'dFC_distance_var', \
    'FO', \
    'CO', \
    'TP', \
    'trans_freq' \
    ]
dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
SUBJ_output['6_states'] = dFC_analyzer.post_analysis( \
    dFCM_lst=dFCM_lst2check, \
    analysis_name_lst=analysis_name_lst \
    )

########################## SlidingWindow_100_nodes #######################

param_dict = {'measure_name': 'SlidingWindow', 'num_select_nodes': 100}
analysis_name_lst = [ \
    'corr_mat', \
    'dFC_distance', \
    'dFC_distance_var', \
    'FO', \
    'CO', \
    'TP', \
    'trans_freq' \
    ]
dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
SUBJ_output['SlidingWindow_100_nodes'] = dFC_analyzer.post_analysis( \
    dFCM_lst=dFCM_lst2check, \
    analysis_name_lst=analysis_name_lst \
    )

# Save
np.save('./dFC_assessed/SUBJ_'+str(subj_id)+'_output.npy', SUBJ_output) 


#################################################################################