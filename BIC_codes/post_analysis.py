
import sys
sys.path.append('./BIC_codes/')
from functions.dFC_funcs import *
import numpy as np
import matplotlib.pyplot as plt
import os

################################# LOAD RESULTS #################################

assessment_results_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/'
output_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/output/'
save_image = True

ALL_RECORDS = os.listdir(assessment_results_root+'dFC_assessed/')
ALL_RECORDS = [i for i in ALL_RECORDS if 'SUBJ_' in i]
ALL_RECORDS.sort()
for s in ALL_RECORDS[:1]:
    output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

FILTERS = [key for key in output]
print(FILTERS)

# FILTERS_new = list()
# for filter in FILTERS:
#     if 'Fs' in filter:
#         FILTERS_new.append(filter)
# FILTERS = FILTERS_new

################################# dFC SAMPLES #################################


for filter in ['default_values']:

    # for SUBJs_output in SUBJs_output_lst:
    for s in ALL_RECORDS[:1]:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

        for measure_id in SUBJs_output[filter]['dFCM_samples']:
            TRs = SUBJs_output[filter]['common_TRs'][:10]
            samples = {}
            for tr in TRs:
                samples['TR'+str(tr)] = SUBJs_output[filter]['dFCM_samples'][measure_id]['TR'+str(tr)]
            visualize_conn_mat(samples, 
                title=SUBJs_output[filter]['measure_lst'][int(measure_id)].measure_name+'_'+filter, 
                fix_lim=False, 
                disp_diag=False,
                save_image=save_image, output_root=output_root
                )

################################# FCS visualization #################################

for filter in ['default_values']:

    # for SUBJs_output in SUBJs_output_lst:
    for s in ALL_RECORDS[:1]:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

        for measure in SUBJs_output[filter]['measure_lst']:

            measure.visualize_FCS(normalize=False, fix_lim=False)

################################# dFC var #################################

for filter in ['default_values']:
    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

        for i, measure in enumerate(SUBJs_output[filter]['measure_lst']):
            if not measure.measure_name in RESULTS:
                RESULTS[measure.measure_name] = list()
            RESULTS[measure.measure_name].append(SUBJs_output[filter]['dFC_var'][i])

    for key in RESULTS:
        RESULTS[key] = np.array(RESULTS[key])
        RESULTS[key] = np.mean(RESULTS[key], axis=0)

    visualize_conn_mat(RESULTS, title='dFC var ' + filter, fix_lim=False, disp_diag=True, cmap='viridis', normalize=True,
                        save_image=save_image, output_root=output_root)

################################# dFC avg #################################

for filter in ['default_values']:
    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

        for i, measure in enumerate(SUBJs_output[filter]['measure_lst']):
            if not measure.measure_name in RESULTS:
                RESULTS[measure.measure_name] = list()
            RESULTS[measure.measure_name].append(SUBJs_output[filter]['dFC_avg'][i])

    for key in RESULTS:
        RESULTS[key] = np.array(RESULTS[key])
        RESULTS[key] = np.mean(RESULTS[key], axis=0)

    visualize_conn_mat(RESULTS, title='dFC avg ' + filter, fix_lim=False, disp_diag=True, cmap='viridis', normalize=True,
                        save_image=save_image, output_root=output_root)

################################# dFC SIMILARITY #################################

distance_metric = 'correlation'

RESULTS = {}
for filter in FILTERS:

    all_subj_dist_mat = list()
    all_subj_var_dist_mat = list()
    # for SUBJs_output in SUBJs_output_lst:
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

        avg_distance_matrix = np.mean(SUBJs_output[filter]['dFC_distance'][distance_metric], axis=0)
        var_distance_matrix = np.var(SUBJs_output[filter]['dFC_distance'][distance_metric], axis=0)
        all_subj_dist_mat.append(avg_distance_matrix)
        all_subj_var_dist_mat.append(var_distance_matrix)

    all_subj_dist_mat = np.array(all_subj_dist_mat)
    all_subj_var_dist_mat = np.array(all_subj_var_dist_mat)
    all_subj_avg = np.mean(all_subj_dist_mat, axis=0)
    across_subj_var = np.var(all_subj_dist_mat, axis=0)
    across_time_var = np.mean(all_subj_var_dist_mat, axis=0)

    RESULTS[filter] = {}
    RESULTS[filter]['avg_corr_mat'] = all_subj_avg
    RESULTS[filter]['var_mat'] = across_subj_var
    RESULTS[filter]['temporal_var'] = across_time_var
    RESULTS[filter]['name_lst'] = list()
    for measure in SUBJs_output[filter]['measure_lst']:
        RESULTS[filter]['name_lst'].append(measure.measure_name)

############ Distance Matrices ############
visualize_conn_mat(RESULTS, title=distance_metric+' distance average', fix_lim=False, disp_diag=True, cmap='viridis', name_lst_key='name_lst', mat_key='avg_corr_mat',
                    save_image=save_image, output_root=output_root)
visualize_conn_mat(RESULTS, title=distance_metric+' distance across subj var', fix_lim=False, disp_diag=True, cmap='viridis', name_lst_key='name_lst', mat_key='var_mat',
                    save_image=save_image, output_root=output_root)
visualize_conn_mat(RESULTS, title=distance_metric+' distance temporal var', fix_lim=False, disp_diag=True, cmap='viridis', name_lst_key='name_lst', mat_key='temporal_var',
                    save_image=save_image, output_root=output_root)

############ Hierarchical Clustering ############
for filter in RESULTS:
    dist_mat_dendo(dist_mat=RESULTS[filter]['avg_corr_mat'], labels=RESULTS[filter]['name_lst'], title='Hierarchical Clustering of Methods ' + filter, save_image=save_image, output_root=output_root)

################################# TIME RECORD #################################

for filter in ['default_values', 'num_select_nodes_50']:

    print('********** time record of ' + filter + '**********')

    avg_FCS_fit = {}
    avg_dFC_assess = {}
    # for SUBJs_output in SUBJs_output_lst:
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

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

if not save_image:
    plt.show()
#################################################################################
