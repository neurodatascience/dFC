import os
import time
import warnings

import numpy as np

from pydfc import MultiAnalysis, data_loader

warnings.simplefilter("ignore")

os.environ["MKL_NUM_THREADS"] = "16"
os.environ["NUMEXPR_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"

################################# Parameters #################################

# Data parameters
# main_root = '../../DATA/ds002785/' # for local
main_root = "../../../DATA/task-based/openneuro/ds002785/"  # for server
dFC_assessed_root = main_root + "dFC_assessed/"
output_root = main_root + "validation_results/"

################################# LOAD FIT MEASURES #################################

SUBJECTS = data_loader.find_subj_list(
    data_root=roi_root, sessions=params_data_load["SESSIONs"]
)

ALL_RECORDS = os.listdir(dFC_assessed_root)
ALL_RECORDS = [i for i in ALL_RECORDS if "dFC" in i]
ALL_RECORDS.sort()
dFC_lst = list()
for s in ALL_RECORDS:
    dFC = np.load(dFC_assessed_root + s, allow_pickle="TRUE").item()
    dFC_lst.append(dFC)
print("dFCs loaded ...")

################################# SIMILARITY MEASUREMENT #################################

# similarity_assessment = SIMILARITY_ASSESSMENT(dFCM_lst=dFCM_dict['dFCM_lst'])

# tic = time.time()
# print('Measurement Started ...')

# print("Similarity measurement started...")
# SUBJ_output = similarity_assessment.run(FILTERS=dFC_analyzer.hyper_param_info, downsampling_method='default')
# print("Similarity measurement done.")

# print('Measurement required %0.3f seconds.' % (time.time() - tic, ))

# # Save
# folder = output_root+'similarity_measured'
# if not os.path.exists(folder):
#     os.makedirs(folder)

# np.save(folder+'/SUBJ_'+str(subj_id)+'_output.npy', SUBJ_output)

#######################################################################################
