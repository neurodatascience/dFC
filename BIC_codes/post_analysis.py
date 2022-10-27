
import sys
sys.path.append('./BIC_codes/')
from functions.dFC_funcs import *
import numpy as np
import matplotlib.pyplot as plt
import os

print('################################# POST ANALYSIS STARTED RUNNING ... #################################')

################################# LOAD RESULTS #################################

# assessment_results_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/'
assessment_results_root = './'
# output_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/out/'
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
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])

        for measure_id in SUBJs_output[filter]['dFCM_samples']:
            TRs = SUBJs_output[filter]['common_TRs'][:10]
            samples = {}
            for tr in TRs:
                samples['TR'+str(tr)] = SUBJs_output[filter]['dFCM_samples'][measure_id][SUBJs_output[filter]['common_TRs'].index(tr), :, :]
            visualize_conn_mat_dict(samples, node_networks=node_networks, 
                title=SUBJs_output[filter]['measure_lst'][int(measure_id)].measure_name+'_'+filter, 
                normalize=True, fix_lim=False, 
                disp_diag=False,
                save_image=save_image, output_root=output_root
                )

################################# FCS visualization #################################

for filter in ['default_values']:

    # for SUBJs_output in SUBJs_output_lst:
    for s in ALL_RECORDS[:1]:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])

        for measure in SUBJs_output[filter]['measure_lst']:

            measure.visualize_FCS(node_networks=node_networks,
                    normalize=True, fix_lim=False, 
                    save_image=save_image, output_root=output_root
                    )

################################# dFC Similarity #################################

################# whole subject #################
'''
    - corr((all dFC timepoints of one Subj using method_i), (all dFC timepoints of one Subj using method_j)) -> avg(corr) and var(corr)
    - spearman correlation, pearson correlation, Mutual Information (MI), Euclidean Distance
    - dendogram based on avg(corr)
'''
metric_list = [
            'corr',
            'spearman',
            'MI',
            'euclidean_distance'
]
for metric in metric_list:

    RESULTS = {}
    for filter in FILTERS:

        all_subjs_sim_mat = list()
        for s in ALL_RECORDS:
            SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
            # SUBJs_output[filter]['all']['metric'] = (1, method, method)
            all_subjs_sim_mat.append(np.squeeze(SUBJs_output[filter]['all'][metric]))

        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        all_subjs_sim_mat = np.array(all_subjs_sim_mat)
        all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
        across_subj_var = np.var(all_subjs_sim_mat, axis=0)

        RESULTS[filter] = {}
        RESULTS[filter]['avg_mat'] = all_subjs_avg
        RESULTS[filter]['var_mat'] = across_subj_var
        RESULTS[filter]['name_lst'] = measure_name_lst

    ############ Distance Matrices ############
    visualize_conn_mat_dict(RESULTS, title=metric+' average', fix_lim=False, 
    disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='avg_mat',
                        save_image=save_image, output_root=output_root)

    visualize_conn_mat_dict(RESULTS, title=metric+' across subj var', fix_lim=False, 
    disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='var_mat',
                        save_image=save_image, output_root=output_root)

    ############ Hierarchical Clustering ############
    for filter in ['default_values']:
        dist_mat = 1 - RESULTS[filter]['avg_mat']
        dist_mat = 0.5*(dist_mat + dist_mat.T)
        # diagonal values of dist_mat must equal exactly zero
        np.fill_diagonal(dist_mat, 0)
        dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS[filter]['name_lst'], 
            title='Hierarchical Clustering of Methods ' + filter+' using '+metric, 
            save_image=save_image, output_root=output_root
        )
################# feature-based #################
'''
    - spatial
    - temporal
    - inter-time-correlation
    - inter-connection-correlation
    - dFC-avg
    - dFC-var
'''
feature2extract_list = [
            'spatial', 'temporal', 
            'inter_time_corr', 'inter_conn_corr', 
            'dFC_avg', 'dFC_var', 
]
for feature2extract in feature2extract_list:

    RESULTS = {}
    for filter in FILTERS:

        all_subjs_sim_mat = list()
        for s in ALL_RECORDS:
            SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
            # SUBJs_output[filter]['feature_based'][feature2extract] = (sample, method, method)
            all_subjs_sim_mat.append(np.mean(SUBJs_output[filter]['feature_based'][feature2extract], axis=0))

        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        all_subjs_sim_mat = np.array(all_subjs_sim_mat)
        all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
        across_subj_var = np.var(all_subjs_sim_mat, axis=0)

        RESULTS[filter] = {}
        RESULTS[filter]['avg_mat'] = all_subjs_avg
        RESULTS[filter]['var_mat'] = across_subj_var
        RESULTS[filter]['name_lst'] = measure_name_lst

    ############ Distance Matrices ############
    visualize_conn_mat_dict(RESULTS, title=feature2extract+' average', fix_lim=False, 
    disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='avg_mat',
                        save_image=save_image, output_root=output_root)

    visualize_conn_mat_dict(RESULTS, title=feature2extract+' across subj var', fix_lim=False, 
    disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='var_mat',
                        save_image=save_image, output_root=output_root)

    ############ Hierarchical Clustering ############
    for filter in ['default_values']:
        dist_mat = 1 - RESULTS[filter]['avg_mat']
        dist_mat = 0.5*(dist_mat + dist_mat.T)
        # diagonal values of dist_mat must equal exactly zero
        np.fill_diagonal(dist_mat, 0)
        dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS[filter]['name_lst'], 
            title='Hierarchical Clustering of Methods ' + filter+' using '+feature2extract, 
            save_image=save_image, output_root=output_root
        )
################# graph-based #################
'''
    - spatial
    - temporal-avg
    - ECM, shortest_path, degree, clustering_coef
'''
graph_property_list = [
            'ECM',
            'shortest_path',
            'degree',
            'clustering_coef'
]
###### spatial #####
for graph_property in graph_property_list:

    RESULTS = {}
    for filter in FILTERS:

        all_subjs_sim_mat = list()
        for s in ALL_RECORDS:
            SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
            # SUBJs_output[filter]['graph_based']['graph_spatial'][graph_property] = (sample, method, method)
            all_subjs_sim_mat.append(np.mean(SUBJs_output[filter]['graph_based']['graph_spatial'][graph_property], axis=0))

        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        all_subjs_sim_mat = np.array(all_subjs_sim_mat)
        all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
        across_subj_var = np.var(all_subjs_sim_mat, axis=0)

        RESULTS[filter] = {}
        RESULTS[filter]['avg_mat'] = all_subjs_avg
        RESULTS[filter]['var_mat'] = across_subj_var
        RESULTS[filter]['name_lst'] = measure_name_lst

    ############ Distance Matrices ############
    visualize_conn_mat_dict(RESULTS, title='spatial '+graph_property+' average', fix_lim=False, 
    disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='avg_mat',
                        save_image=save_image, output_root=output_root)

    visualize_conn_mat_dict(RESULTS, title='spatial '+graph_property+' across subj var', fix_lim=False, 
    disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='var_mat',
                        save_image=save_image, output_root=output_root)

    ############ Hierarchical Clustering ############
    for filter in ['default_values']:
        dist_mat = 1 - RESULTS[filter]['avg_mat']
        dist_mat = 0.5*(dist_mat + dist_mat.T)
        # diagonal values of dist_mat must equal exactly zero
        np.fill_diagonal(dist_mat, 0)
        dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS[filter]['name_lst'], 
            title='Hierarchical Clustering of Methods ' + filter+' using '+ 'spatial '+ graph_property, 
            save_image=save_image, output_root=output_root
        )
###### temporal #####
for graph_property in graph_property_list:

    RESULTS = {}
    for filter in FILTERS:

        all_subjs_sim_mat = list()
        for s in ALL_RECORDS:
            SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
            # SUBJs_output[filter]['graph_based']['graph_temporal'][graph_property] = (1, method, method)
            all_subjs_sim_mat.append(np.squeeze(SUBJs_output[filter]['graph_based']['graph_temporal'][graph_property]))

        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        all_subjs_sim_mat = np.array(all_subjs_sim_mat)
        all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
        across_subj_var = np.var(all_subjs_sim_mat, axis=0)

        RESULTS[filter] = {}
        RESULTS[filter]['avg_mat'] = all_subjs_avg
        RESULTS[filter]['var_mat'] = across_subj_var
        RESULTS[filter]['name_lst'] = measure_name_lst

    ############ Distance Matrices ############
    visualize_conn_mat_dict(RESULTS, title='temporal '+graph_property+' average', fix_lim=False, 
    disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='avg_mat',
                        save_image=save_image, output_root=output_root)

    visualize_conn_mat_dict(RESULTS, title='temporal '+graph_property+' across subj var', fix_lim=False, 
    disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='var_mat',
                        save_image=save_image, output_root=output_root)

    ############ Hierarchical Clustering ############
    for filter in ['default_values']:
        dist_mat = 1 - RESULTS[filter]['avg_mat']
        dist_mat = 0.5*(dist_mat + dist_mat.T)
        # diagonal values of dist_mat must equal exactly zero
        np.fill_diagonal(dist_mat, 0)
        dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS[filter]['name_lst'], 
            title='Hierarchical Clustering of Methods ' + filter+' using '+ 'temporal '+ graph_property, 
            save_image=save_image, output_root=output_root
        )

################################# inter_subject similarity #################################

'''
    - returns correspondence of inter-subject relation between results of dFC 
        measures in each ssession
    - dendogram based on inter-subject similarity
'''

RESULTS = {}
for filter in FILTERS:
    features_subj = list()
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

        features_i = list()
        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):

            dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]

            # rank normalization
            dFC_mat_i = rank_norm(dFC_mat_i) 

            features_i.append(dFC_mat2vec(dFC_mat_i))
            
        features_i = np.array(features_i)
        features_subj.append(features_i)

    features_subj = np.array(features_subj) # features_subj = (subj, method, time, connections)

    sim_mat = np.zeros((len(SUBJs_output[filter]['measure_lst']), len(SUBJs_output[filter]['measure_lst'])))
    for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):
        for j, measure_j in enumerate(SUBJs_output[filter]['measure_lst']):

            features_i = np.squeeze(features_subj[:,i,:,:]) # features_i = (subj, time, connections)
            features_i = features_i.reshape(features_i.shape[0], -1) # features_i = (subj, dFC_values)
            features_j = np.squeeze(features_subj[:,j,:,:])
            features_j = features_j.reshape(features_j.shape[0], -1)

            inter_subj_sim_i = np.corrcoef(features_i) # C_i = (subj, subj)
            inter_subj_sim_j = np.corrcoef(features_j) # C_j = (subj, subj)

            inter_subj_sim_i = dFC_mat2vec(inter_subj_sim_i)
            inter_subj_sim_j = dFC_mat2vec(inter_subj_sim_j)

            spear_coef, p_value = stats.spearmanr(inter_subj_sim_i, inter_subj_sim_j)
            sim_mat[i, j] = spear_coef

    RESULTS[filter] = {}
    RESULTS[filter]['sim_mat'] = sim_mat
    measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
    RESULTS[filter]['name_lst'] = measure_name_lst

############ Distance Matrices ############
visualize_conn_mat_dict(RESULTS, title='inter-subject similarity', fix_lim=False, 
disp_diag=False, cmap='viridis', name_lst_key='name_lst', mat_key='sim_mat',
                    save_image=save_image, output_root=output_root)

############ Hierarchical Clustering ############
for filter in RESULTS:
    dist_mat = 1 - RESULTS[filter]['sim_mat']
    dist_mat = 0.5*(dist_mat + dist_mat.T)
    # diagonal values of dist_mat must equal exactly zero
    np.fill_diagonal(dist_mat, 0)
    dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS[filter]['name_lst'], 
    title='Hierarchical Clustering of Methods ' + filter+' using inter-subject similarity', 
    save_image=save_image, output_root=output_root
    )

################################# dFC var #################################

'''
    - avg(variance/fluctuations of dFC in one Subj)
'''
for filter in ['default_values']:
    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])

        for i, measure in enumerate(SUBJs_output[filter]['measure_lst']):
            if not measure.measure_name in RESULTS:
                RESULTS[measure.measure_name] = list()
            # SUBJs_output[filter]['dFC_var'][i] = (1, connection)
            var_mat = np.squeeze(dFC_vec2mat(SUBJs_output[filter]['dFC_var'][i])) # (ROI, ROI)
            np.fill_diagonal(var_mat, 0)
            RESULTS[measure.measure_name].append(var_mat)

    for key in RESULTS:
        RESULTS[key] = np.array(RESULTS[key])
        RESULTS[key] = np.mean(RESULTS[key], axis=0)

    visualize_conn_mat_dict(RESULTS, node_networks=node_networks, 
                title='dFC var ' + filter, 
                fix_lim=False, disp_diag=True, cmap='jet', normalize=False, 
                save_image=save_image, output_root=output_root)

################################# dFC avg #################################

'''
    - avg(avg of dFC -static FC- in one Subj)
'''

for filter in ['default_values']:
    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])

        for i, measure in enumerate(SUBJs_output[filter]['measure_lst']):
            if not measure.measure_name in RESULTS:
                RESULTS[measure.measure_name] = list()
            # SUBJs_output[filter]['dFC_avg'][i] = (1, connection)
            avg_mat = np.squeeze(dFC_vec2mat(SUBJs_output[filter]['dFC_avg'][i])) # (ROI, ROI)
            RESULTS[measure.measure_name].append(avg_mat)

    for key in RESULTS:
        RESULTS[key] = np.array(RESULTS[key])
        RESULTS[key] = np.mean(RESULTS[key], axis=0)

    visualize_conn_mat_dict(RESULTS, node_networks=node_networks, 
            title='dFC avg ' + filter, 
            fix_lim=False, disp_diag=True, cmap='jet', normalize=False,
            save_image=save_image, output_root=output_root)

################################# Across Node Temporal Correlation #################################

'''
    - spearman_corr((dFConnection(node_i, node_j) timecourse using method m), (dFConnection(node_i, node_j) timecourse using method n))
'''
for filter in ['default_values']:
    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])

        # SUBJs_output[filter]['feature_based']['temporal'] = (connection, method, method)

        for i in range(SUBJs_output[filter]['feature_based']['temporal'].shape[1]):
            for j in range(i):
                measure_name_i = zip_name(SUBJs_output[filter]['measure_lst'][i].measure_name)
                measure_name_j = zip_name(SUBJs_output[filter]['measure_lst'][j].measure_name)
                if not measure_name_i in RESULTS:
                    RESULTS[measure_name_i] = {}
                if not measure_name_j in RESULTS[measure_name_i]:
                    RESULTS[measure_name_i][measure_name_j] = list()
                mat = dFC_vec2mat(SUBJs_output[filter]['feature_based']['temporal'][:, i, j]) # (ROI, ROI)
                RESULTS[measure_name_i][measure_name_j].append(mat)

    for key_i in RESULTS:
        for key_j in RESULTS[key_i]:
            RESULTS[key_i][key_j] = np.array(RESULTS[key_i][key_j])
            RESULTS[key_i][key_j] = np.mean(RESULTS[key_i][key_j], axis=0)
            
    visualize_conn_mat_2D_dict(RESULTS, node_networks=node_networks, 
        title='across node temporal spearman corr ' + filter, fix_lim=False, 
        disp_diag=False, cmap='jet', normalize=True,
        save_image=save_image, output_root=output_root
        )

################################# Subject Clustering #################################
'''

'''
# RESULTS = {}
#     for filter in FILTERS:

#         all_subj_avg_dist_mat = list()
#         all_subj_var_dist_mat = list()
#         for s in ALL_RECORDS:
#             SUBJs_output = np.load(assessment_results_root+'dFC_assessed/'+s, allow_pickle='True').item()

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
