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

# Data parameters
# main_root = '../../DATA/ds002785/' # for local
main_root = '../../../DATA/task-based/openneuro/ds002785/' # for server

# subjects used for dFC assessment do not need to be the same as those used for FCS_estimate
# you can set the new roi root and data load parameters here:
roi_root = main_root + 'ROI_timeseries/' 

TR = 2.00

params_data_load = { 
    # data root
    'data_root': roi_root,
    # file name
    'file_name': 'time_series.npy',
    # SESSION
    'SESSIONs': ['task-all'], 
    # networks to include in analysis
    'networks2include':None, 
    # locs
    'roi_locs_file': 'center_locs.npy',
    # labels
    'roi_labels_file': 'region_labels.npy',
    # sampling frequency
    'Fs': 1/TR,
}

fitted_measures_root = main_root + 'fitted_MEASURES/'
output_root = main_root + 'dFC_assessed/'

################################# LOAD FIT MEASURES #################################

MA = np.load(fitted_measures_root+'multi_analysis.npy', allow_pickle='TRUE').item()
SUBJECTS = data_loader.find_subj_list(data_root=roi_root, sessions=params_data_load['SESSIONs'])

ALL_RECORDS = os.listdir(fitted_measures_root)
ALL_RECORDS = [i for i in ALL_RECORDS if 'MEASURE' in i]
ALL_RECORDS.sort()
MEASURES_fit_lst = list()
for s in ALL_RECORDS:
    fit_measure = np.load(fitted_measures_root+s, allow_pickle='TRUE').item()
    MEASURES_fit_lst.append(fit_measure)
MA.set_MEASURES_fit_lst(MEASURES_fit_lst)
print('fitted MEASURES loaded ...')

################################# LOAD DATA #################################

job_id = int(os.getenv("SGE_TASK_ID"))
if job_id > len(SUBJECTS):
    print('job_id > len(SUBJECTS)')
    exit()
subj_id = SUBJECTS[job_id-1] # SGE_TASK_ID starts from 1 not 0
print('dFC assessment CODE started running ... for subject: ' + subj_id + ' ...')

BOLD = data_loader.load_from_array(subj_id2load=subj_id, **params_data_load)

################################# dFC ASSESSMENT #################################

tic = time.time()
print('Measurement Started ...')

print("dFC estimation started...")
dFC_dict = MA.subj_lvl_dFC_assess(time_series_dict=BOLD)
print("dFC estimation done.")

print('Measurement required %0.3f seconds.' % (time.time() - tic, ))

################################# SAVE DATA #################################

folder = output_root + subj_id
if not os.path.exists(folder):
    os.makedirs(folder)

for dFC_id, dFC in enumerate(dFC_dict['dFC_lst']):
    np.save(folder+'/dFC_'+str(dFC_id)+'.npy', dFC) 
    
#######################################################################################