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

################################# LOAD #################################

dFC_analyzer = np.load('./dFC_analyzer.npy',allow_pickle='TRUE').item()
data_loader = np.load('./data_loader.npy',allow_pickle='TRUE').item()

################################# LOAD FIT MEASURES #################################

if dFC_analyzer.MEASURES_fit_lst==[]:
    ALL_RECORDS = os.listdir('./fitted_MEASURES/')
    ALL_RECORDS = [i for i in ALL_RECORDS if 'MEASURE' in i]
    ALL_RECORDS.sort()
    MEASURES_fit_lst = list()
    for s in ALL_RECORDS:
        fit_measure = np.load('./fitted_MEASURES/'+s,allow_pickle='TRUE').item()
        MEASURES_fit_lst.append(fit_measure)
    dFC_analyzer.set_MEASURES_fit_lst(MEASURES_fit_lst)
    print('fitted MEASURES loaded ...')
    # np.save('./dFC_analyzer.npy', dFC_analyzer) 

################################# LOAD DATA #################################

task_id = int(os.getenv("SGE_TASK_ID"))
subj_id = data_loader.SUBJECTS[task_id-1] # SGE_TASK_ID starts from 1 not 0

BOLD = data_loader.load(subj_id2load=subj_id)

################################# dFC ASSESSMENT #################################

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

analysis_name_lst = [ \
    'corr_mat', \
    'across_node_corr_mat', \
    'dFC_avg', \
    'dFC_var', \
    'dFC_distance', \
    # 'dFC_distance_var', \
    'FO', \
    'CO', \
    'TP', \
    'trans_freq' \
    ]

for filter in dFC_analyzer.hyper_param_info:
    param_dict = dFC_analyzer.hyper_param_info[filter]
    dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
    SUBJ_output[filter] = dFC_analyzer.post_analysis( \
        dFCM_lst=dFCM_lst2check, \
        analysis_name_lst=analysis_name_lst \
        )


# ########################## DEFAULT VALUES #######################

# param_dict = dFC_analyzer.params_methods
# dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
# SUBJ_output['default_values'] = dFC_analyzer.post_analysis( \
#     dFCM_lst=dFCM_lst2check, \
#     analysis_name_lst=analysis_name_lst \
#     )

# ########################## 6_states #######################

# param_dict = {'n_states': [6]}
# dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
# SUBJ_output['6_states'] = dFC_analyzer.post_analysis( \
#     dFCM_lst=dFCM_lst2check, \
#     analysis_name_lst=analysis_name_lst \
#     )

# ########################## 16_states #######################

# param_dict = {'n_states': [16]}
# dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
# SUBJ_output['16_states'] = dFC_analyzer.post_analysis( \
#     dFCM_lst=dFCM_lst2check, \
#     analysis_name_lst=analysis_name_lst \
#     )

# ########################## Fs_ratio_0.5 #######################

# param_dict = {'Fs_ratio': [0.5]}
# dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
# SUBJ_output['Fs_ratio_0.5'] = dFC_analyzer.post_analysis( \
#     dFCM_lst=dFCM_lst2check, \
#     analysis_name_lst=analysis_name_lst \
#     )

# ########################## noise_ratio_2 #######################

# param_dict = {'noise_ratio': [2.0]}
# dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
# SUBJ_output['noise_ratio_2'] = dFC_analyzer.post_analysis( \
#     dFCM_lst=dFCM_lst2check, \
#     analysis_name_lst=analysis_name_lst \
#     )

# ########################## noise_ratio_3 #######################

# param_dict = {'noise_ratio': [3.0]}
# dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
# SUBJ_output['noise_ratio_3'] = dFC_analyzer.post_analysis( \
#     dFCM_lst=dFCM_lst2check, \
#     analysis_name_lst=analysis_name_lst \
#     )

# ########################## num_select_nodes_50 #######################

# param_dict = {'num_select_nodes': [50]}
# dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
# SUBJ_output['num_select_nodes_50'] = dFC_analyzer.post_analysis( \
#     dFCM_lst=dFCM_lst2check, \
#     analysis_name_lst=analysis_name_lst \
#     )

# ########################## num_select_nodes_100 #######################

# param_dict = {'num_select_nodes': [100]}
# dFCM_lst2check = filter_dFCM_lst(dFCM_lst, **param_dict)
# SUBJ_output['num_select_nodes_100'] = dFC_analyzer.post_analysis( \
#     dFCM_lst=dFCM_lst2check, \
#     analysis_name_lst=analysis_name_lst \
#     )

# Save
np.save('./dFC_assessed/SUBJ_'+str(subj_id)+'_output.npy', SUBJ_output) 
#################################################################################