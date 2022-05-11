
import sys
sys.path.append('./BIC_codes/')
from functions.dFC_funcs import *
import numpy as np
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
import os

################################# LOAD RESULTS #################################

assessment_results_root = './../../../RESULTs/methods_implementation/server/methods_implementation/'

ALL_RECORDS = os.listdir(assessment_results_root+'dFC_assessed/')
ALL_RECORDS = [i for i in ALL_RECORDS if 'SUBJ_' in i]
ALL_RECORDS.sort()
SUBJs_output_lst = list()
for s in ALL_RECORDS:
    output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
    SUBJs_output_lst.append(output)

print('assessed dFCs loaded ...')


################################# dFC SAMPLES #################################


for filter in ['num_select_nodes_50']:

    for SUBJs_output in SUBJs_output_lst[1:2]:

        for measure_id in SUBJs_output[filter]['dFCM_samples']:
            TRs = SUBJs_output[filter]['common_TRs'][:10]
            samples = {}
            for tr in TRs:
                samples['TR'+str(tr)] = SUBJs_output[filter]['dFCM_samples'][measure_id]['TR'+str(tr)]
            visualize_conn_mat(samples, 
                title=SUBJs_output[filter]['measure_lst'][int(measure_id)].measure_name, 
                fix_lim=False, 
                disp_diag=False
                )

################################# dFC SIMILARITY #################################


RESULTS = {}
for filter in ['default_values', '6_states', 'num_select_nodes_50', 'Fs_ratio_0.5', 'noise_ratio_2']:

    all_subj_avg = list()
    for SUBJs_output in SUBJs_output_lst:
        avg_distance_matrix = np.mean(SUBJs_output[filter]['dFC_distance']['euclidean'], axis=0)
        all_subj_avg.append(avg_distance_matrix)

    all_subj_avg = np.array(all_subj_avg)
    all_subj_avg = np.mean(all_subj_avg, axis=0)

    RESULTS[filter] = {}
    RESULTS[filter]['corr_mat'] = all_subj_avg
    RESULTS[filter]['name_lst'] = list()
    for measure in SUBJs_output[filter]['measure_lst']:
        RESULTS[filter]['name_lst'].append(measure.measure_name)

############ Distance Matrices ############
visualize_conn_mat(RESULTS, title='Euclidean Distance', fix_lim=False, disp_diag=True, cmap='viridis', name_lst_key='name_lst', mat_key='corr_mat')

############ Hierarchical Clustering ############
for filter in RESULTS:
    # convert the redundant n*n square matrix form into a condensed nC2 array
    distArray = ssd.squareform(RESULTS[filter]['corr_mat']) 

    plt.figure(figsize=(25, 5))
    dend = shc.dendrogram(shc.linkage(distArray, method='single', metric='euclidean'), distance_sort='ascending', no_plot=False, labels=RESULTS[filter]['name_lst'])
    plt.title('Hierarchical Clustering of Methods ' + filter)

################################# TIME RECORD #################################

for filter in ['default_values', 'num_select_nodes_50']:

    print('********** time record of ' + filter + '**********')

    avg_FCS_fit = {}
    avg_dFC_assess = {}
    for SUBJs_output in SUBJs_output_lst:
        measure_name_lst = list()
        for measure_id in SUBJs_output[filter]['time_record_dict']:
            if not measure_id in avg_FCS_fit:
                avg_FCS_fit[measure_id] = list()
            if not measure_id in avg_dFC_assess:
                avg_dFC_assess[measure_id] = list()
            avg_FCS_fit[measure_id].append(SUBJs_output[filter]['time_record_dict'][measure_id]['FCS_fit'])
            avg_dFC_assess[measure_id].append(SUBJs_output[filter]['time_record_dict'][measure_id]['dFC_assess'])
            measure_name_lst.append(SUBJs_output[filter]['measure_lst'][int(measure_id)].measure_name)
            
    for measure_id in avg_FCS_fit:
        if None in avg_FCS_fit[measure_id]:
            FCS_result = measure_name_lst[int(measure_id)] + ': FCS_fit '+' = None'
        else:
            avg_FCS_fit[measure_id] = np.mean(avg_FCS_fit[measure_id])
            FCS_result = measure_name_lst[int(measure_id)] + ': FCS_fit '+' = %0.3f' % (avg_FCS_fit[measure_id])
        avg_dFC_assess[measure_id] = np.mean(avg_dFC_assess[measure_id])
        dFC_result = 'dFC_assess '+' = %0.3f' % (avg_dFC_assess[measure_id])
        print( FCS_result + ' , ' + dFC_result )

#################################################################################
