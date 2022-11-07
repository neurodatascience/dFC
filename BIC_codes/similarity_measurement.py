from functions.dFC_funcs import *
import numpy as np
import time
import hdf5storage
import scipy.io as sio
import os
os.environ["MKL_NUM_THREADS"] = '64'
os.environ["NUMEXPR_NUM_THREADS"] = '64'
os.environ["OMP_NUM_THREADS"] = '64'

print('################################# subject-level similarity measurement CODE started running ... #################################')

################################# Parameters #################################
###### DATA PARAMETERS ######

input_root = './'
output_root = './'

################################# LOAD #################################

dFC_analyzer = np.load(input_root+'dFC_analyzer.npy',allow_pickle='TRUE').item()
data_loader = np.load(input_root+'data_loader.npy',allow_pickle='TRUE').item()

################################# LOAD ASSESSED dFC #################################

task_id = int(os.getenv("SGE_TASK_ID"))
subj_id = data_loader.SUBJECTS[task_id-1] # SGE_TASK_ID starts from 1 not 0
folder = input_root+'dFC_assessed/SUBJ_'+str(subj_id)

ALL_RECORDS = os.listdir(folder+'/')
ALL_RECORDS = [i for i in ALL_RECORDS if 'dFCM' in i]
ALL_RECORDS.sort()
dFCM_lst = list()
for s in ALL_RECORDS:
    dFCM = np.load(folder+'/'+s, allow_pickle='TRUE').item()
    dFCM_lst.append(dFCM)
print('assessed dFCMs loaded ...')

################################# SIMILARITY MEASUREMENT #################################

similarity_assessment = SIMILARITY_ASSESSMENT(dFCM_lst=dFCM_lst)

tic = time.time()
print('Measurement Started ...')

print("dFCM estimation started...")
SUBJ_output = similarity_assessment.run(FILTERS=dFC_analyzer.hyper_param_info, downsampling_method='default')
print("dFCM estimation done.")

print('Measurement required %0.3f seconds.' % (time.time() - tic, ))

# Save
folder = output_root+'similarity_measured'
if not os.path.exists(folder):
    os.makedirs(folder)

np.save(folder+'/SUBJ_'+str(subj_id)+'_output.npy', SUBJ_output) 
#################################################################################