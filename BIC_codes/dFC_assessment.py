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
SUBJ_output = dFC_analyzer.subj_lvl_dFC_assess(time_series_dict=BOLD)
# SUBJ_output = dFC_analyzer.group_dFCM_assess(time_series_dict=BOLD)
print("dFCM estimation done.")

print('Measurement required %0.3f seconds.' % (time.time() - tic, ))

# #### Methods dFC Corr MAT ###

# fig_name = None
# if dFC_analyzer.params['save_image']:
#     output_root = dFC_analyzer.params['output_root']+'dFC/'
#     fig_name = output_root + 'avg_dFC_corr' 

# visualize_conn_mat(dFC_analyzer.methods_corr, \
#     title='intra session dFC correlation', \
#     name_lst_key='measure_lst', mat_key='corr_mat', \
#     cmap='viridis',\
#     save_image=dFC_analyzer.params['save_image'], output_root=fig_name, \
#         fix_lim=True \
# )

# Save
np.save('./dFC_assessed/SUBJ_'+str(subj_id)+'_output.npy', SUBJ_output) 

# ################################# SIMILARITY ASSESSMENT #################################

# # print_dict(dFC_analyzer.methods_corr)
# dFC_analyzer.similarity_analyze(SUBJs_dFC_session_sim_dict)

# ################################# STATE MATCH #################################

# state_match = dFC_analyzer.state_match()
# # Save
# np.save('./state_match.npy', state_match) 
