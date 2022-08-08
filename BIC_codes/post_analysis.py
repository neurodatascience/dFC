
import sys
sys.path.append('./BIC_codes/')
from functions.dFC_funcs import *
import numpy as np
import matplotlib.pyplot as plt
import os

print('################################# POST ANALYSIS STARTED RUNNING ... #################################')

################################# LOAD RESULTS #################################

assessment_results_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/'
assessment_results_root = './'
output_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/output/'
output_root = './output/'
save_image = True

ALL_RECORDS = os.listdir(assessment_results_root+'dFC_assessed/')
ALL_RECORDS = [i for i in ALL_RECORDS if 'SUBJ_' in i]
ALL_RECORDS.sort()
for s in ALL_RECORDS[:1]:
    output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

FILTERS = [key for key in output]
print(FILTERS)

for filter in FILTERS:
    measure_name_lst = None
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
        # check measures order
        if measure_name_lst is None:
            measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        else:
            assert measure_name_lst==[measure.measure_name for measure in SUBJs_output[filter]['measure_lst']], \
                'measures order mismatch'

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
            visualize_conn_mat_dict(samples, 
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

            measure.visualize_FCS(normalize=False, fix_lim=False, save_image=save_image, output_root=output_root)

################################# dFC Similarity #################################

'''
    - MI((all dFC timepoints of one Subj using method_i), (all dFC timepoints of one Subj using method_j)) -> avg(MI) and var(MI)
    - corr((all dFC timepoints of one Subj using method_i), (all dFC timepoints of one Subj using method_j)) -> avg(corr) and var(corr)
    - dendogram based on avg(corr)
'''
for similarity_metric in ['corr', 'MI']:

    RESULTS = {}
    for filter in FILTERS:

        all_subj_dist_mat = list()
        for s in ALL_RECORDS:
            SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
                
            all_subj_dist_mat.append(SUBJs_output[filter]['subj_dFC_sim'][similarity_metric])

        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        all_subj_dist_mat = np.array(all_subj_dist_mat)
        all_subj_avg = np.mean(all_subj_dist_mat, axis=0)
        across_subj_var = np.var(all_subj_dist_mat, axis=0)

        RESULTS[filter] = {}
        RESULTS[filter]['avg_distance_mat'] = all_subj_avg
        RESULTS[filter]['var_mat'] = across_subj_var
        RESULTS[filter]['name_lst'] = measure_name_lst

    ############ Distance Matrices ############
    visualize_conn_mat_dict(RESULTS, title=similarity_metric+' average', fix_lim=False, 
    disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='avg_distance_mat',
                        save_image=save_image, output_root=output_root)

    visualize_conn_mat_dict(RESULTS, title=similarity_metric+' across subj var', fix_lim=False, 
    disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='var_mat',
                        save_image=save_image, output_root=output_root)

    ############ Hierarchical Clustering ############
    if similarity_metric == 'corr':
        for filter in RESULTS:
            dist_mat = 1 - np.abs(RESULTS[filter]['avg_distance_mat'])
            dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS[filter]['name_lst'], 
            title='Hierarchical Clustering of Methods ' + filter+' using '+similarity_metric, 
            save_image=save_image, output_root=output_root
            )

################################# dFC var #################################

'''
    - avg(variance/fluctuations of dFC in one Subj)
    - similarity(avg(variance/fluctuations of dFC in one Subj using method_i), avg(variance/fluctuations of dFC in one Subj using method_j))
'''
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

    # dFC var similarity
    corr = {}
    corr['dFC_var_similarity'] = {}
    corr['dFC_var_similarity']['corr_mat'] = np.zeros((len(SUBJs_output[filter]['measure_lst']), len(SUBJs_output[filter]['measure_lst'])))
    corr['dFC_var_similarity']['name_lst'] = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
    for i, measure_i in enumerate(RESULTS):
        for j, measure_j in enumerate(RESULTS):
            corr['dFC_var_similarity']['corr_mat'][i, j] = np.corrcoef(dFC_mat2vec(RESULTS[measure_i]), dFC_mat2vec(RESULTS[measure_j]))[0,1]


    visualize_conn_mat_dict(RESULTS, title='dFC var ' + filter, fix_lim=False, disp_diag=True, cmap='viridis', normalize=False, 
                        save_image=save_image, output_root=output_root)
    
    visualize_conn_mat_dict(corr, title='dFC var similarity ' + filter, fix_lim=False, 
        disp_diag=False, cmap='viridis', normalize=False, name_lst_key='name_lst', mat_key='corr_mat',
                        save_image=save_image, output_root=output_root)

################################# dFC avg #################################

'''
    - avg(avg of dFC -static FC- in one Subj)
    - similarity(avg(avg of dFC -static FC- in one Subj using method_i), avg(avg of dFC -static FC- in one Subj using method_j))
'''

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

    # dFC avg similarity
    corr = {}
    corr['dFC_avg_similarity'] = {}
    corr['dFC_avg_similarity']['corr_mat'] = np.zeros((len(SUBJs_output[filter]['measure_lst']), len(SUBJs_output[filter]['measure_lst'])))
    corr['dFC_avg_similarity']['name_lst'] = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
    for i, measure_i in enumerate(RESULTS):
        for j, measure_j in enumerate(RESULTS):
            corr['dFC_avg_similarity']['corr_mat'][i, j] = np.corrcoef(dFC_mat2vec(RESULTS[measure_i]), dFC_mat2vec(RESULTS[measure_j]))[0,1]

    visualize_conn_mat_dict(RESULTS, title='dFC avg ' + filter, fix_lim=False, disp_diag=True, cmap='viridis', normalize=False,
                        save_image=save_image, output_root=output_root)

    visualize_conn_mat_dict(corr, title='dFC avg similarity ' + filter, fix_lim=False, 
        disp_diag=False, cmap='viridis', normalize=False, name_lst_key='name_lst', mat_key='corr_mat',
                        save_image=save_image, output_root=output_root)

################################# Across Node Temporal Correlation #################################

'''
    - corr((dFConnection(node_i, node_j) timecourse using method m), (dFConnection(node_i, node_j) timecourse using method n))
'''
for filter in ['default_values']:
    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

        for i in range(SUBJs_output[filter]['across_node_corr_mat'].shape[0]):
            for j in range(i):
                measure_name_i = zip_name(SUBJs_output[filter]['measure_lst'][i].measure_name)
                measure_name_j = zip_name(SUBJs_output[filter]['measure_lst'][j].measure_name)
                if not measure_name_i in RESULTS:
                    RESULTS[measure_name_i] = {}
                if not measure_name_j in RESULTS[measure_name_i]:
                    RESULTS[measure_name_i][measure_name_j] = list()
                mat = SUBJs_output[filter]['across_node_corr_mat'][i,j,:,:]
                mat = np.nan_to_num(mat)
                RESULTS[measure_name_i][measure_name_j].append(mat)

    for key_i in RESULTS:
        for key_j in RESULTS[key_i]:
            RESULTS[key_i][key_j] = np.array(RESULTS[key_i][key_j])
            RESULTS[key_i][key_j] = np.mean(RESULTS[key_i][key_j], axis=0)
            
    visualize_conn_mat_2D_dict(RESULTS, title='across node temporal corr ' + filter, fix_lim=False, 
        disp_diag=False, cmap='viridis', normalize=False,
                            save_image=save_image, output_root=output_root)


################################# dFC Distance #################################

'''
    - distance((dFC_t using method_i), (dFC_t using method_j)) over time -> avg(distance) in one Subj, var(distance) in one Subj
    - avg(avg(distance) in one Subj) -> avg_distance_mat
    - var(avg(distance) in one Subj) -> across_subj_var_distance_mat
    - avg(var(distance) in one Subj) -> across_time_var_distance_mat
    - dendogram based on avg_distance_mat
'''
for distance_metric in ['correlation', 'euclidean', 'ECM', 'degree', 'shortest_path', 'clustering_coef']:

    RESULTS = {}
    for filter in FILTERS:

        all_subj_avg_dist_mat = list()
        all_subj_var_dist_mat = list()
        for s in ALL_RECORDS:
            SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

            avg_distance_matrix = np.mean(SUBJs_output[filter]['dFC_distance'][distance_metric], axis=0)
            var_distance_matrix = np.var(SUBJs_output[filter]['dFC_distance'][distance_metric], axis=0)
            all_subj_avg_dist_mat.append(avg_distance_matrix)
            all_subj_var_dist_mat.append(var_distance_matrix)

        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        all_subj_avg_dist_mat = np.array(all_subj_avg_dist_mat)
        all_subj_var_dist_mat = np.array(all_subj_var_dist_mat)
        all_subj_avg = np.mean(all_subj_avg_dist_mat, axis=0)
        across_subj_var = np.var(all_subj_avg_dist_mat, axis=0)
        avg_across_time_var = np.mean(all_subj_var_dist_mat, axis=0)

        RESULTS[filter] = {}
        RESULTS[filter]['avg_distance_mat'] = all_subj_avg
        RESULTS[filter]['across_subj_var_distance_mat'] = across_subj_var
        RESULTS[filter]['across_time_var_distance_mat'] = avg_across_time_var 
        RESULTS[filter]['name_lst'] = measure_name_lst

    ############ Distance Matrices ############
    visualize_conn_mat_dict(RESULTS, title=distance_metric+' distance average', fix_lim=False, 
    disp_diag=True, cmap='viridis', name_lst_key='name_lst', mat_key='avg_distance_mat',
                        save_image=save_image, output_root=output_root)

    visualize_conn_mat_dict(RESULTS, title=distance_metric+' distance across subj var', fix_lim=False, 
    disp_diag=True, cmap='viridis', name_lst_key='name_lst', mat_key='across_subj_var_distance_mat',
                        save_image=save_image, output_root=output_root)

    visualize_conn_mat_dict(RESULTS, title=distance_metric+' distance temporal var', fix_lim=False, 
    disp_diag=True, cmap='viridis', name_lst_key='name_lst', mat_key='across_time_var_distance_mat',
                        save_image=save_image, output_root=output_root)

    ############ Hierarchical Clustering ############
    for filter in RESULTS:
        dist_mat_dendo(dist_mat=RESULTS[filter]['avg_distance_mat'], labels=RESULTS[filter]['name_lst'], 
        title='Hierarchical Clustering of Methods ' + filter+' using '+distance_metric, 
        save_image=save_image, output_root=output_root
        )

################################# TIME RECORD #################################

for filter in ['default_values']:

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
