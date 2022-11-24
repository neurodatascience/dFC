
import sys
sys.path.append('./BIC_codes/')
from functions.dFC_funcs import *
import numpy as np
import matplotlib.pyplot as plt
import os

print('################################# POST ANALYSIS STARTED RUNNING ... #################################')

################################# LOAD RESULTS #################################

assessment_results_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/'
# assessment_results_root = './'
output_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/out/'
# output_root = './output/'
FOLDER_name = 'similarity_measured/'
save_image = True

ALL_RECORDS = os.listdir(assessment_results_root+FOLDER_name)
ALL_RECORDS = [i for i in ALL_RECORDS if 'SUBJ_' in i]
ALL_RECORDS.sort()
for s in ALL_RECORDS[:1]:
    output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

print('*** %d subjects were found.' % (len(ALL_RECORDS)))

FILTERS = [key for key in output]
print(FILTERS)

for filter in FILTERS:
    measure_name_lst = None
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
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
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])

        for measure_id in SUBJs_output[filter]['dFCM_samples']:
            TRs = SUBJs_output[filter]['common_TRs'][:10]
            samples = {}
            samples_ranked = {}
            for tr in TRs:
                samples['TR'+str(tr)] = SUBJs_output[filter]['dFCM_samples'][measure_id][SUBJs_output[filter]['common_TRs'].index(tr), :, :]
                samples_ranked['TR'+str(tr)] = rank_norm(SUBJs_output[filter]['dFCM_samples'][measure_id][SUBJs_output[filter]['common_TRs'].index(tr), :, :])
            visualize_conn_mat_dict(samples, node_networks=node_networks, 
                title=SUBJs_output[filter]['measure_lst'][int(measure_id)].measure_name+'_'+filter, 
                normalize=False, fix_lim=False, 
                disp_diag=False,
                save_image=save_image, output_root=output_root+'dFC_sample/'
                )
            visualize_conn_mat_dict(samples_ranked, node_networks=node_networks, 
                title=SUBJs_output[filter]['measure_lst'][int(measure_id)].measure_name+'_ranked_'+filter, 
                normalize=False, fix_lim=False, 
                disp_diag=False, cmap='plasma', center_0=False,
                save_image=save_image, output_root=output_root+'dFC_sample/'
                )

################################# FCS visualization #################################

for filter in ['default_values']:

    # for SUBJs_output in SUBJs_output_lst:
    for s in ALL_RECORDS[:1]:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        for measure in SUBJs_output[filter]['measure_lst']:

            measure.visualize_FCS(
                    normalize=True, fix_lim=False, 
                    save_image=save_image, output_root=output_root+'FCS/'
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
            SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
            # SUBJs_output[filter]['all']['metric'] = (1, method, method)
            all_subjs_sim_mat.append(np.squeeze(SUBJs_output[filter]['all'][metric]))

        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        all_subjs_sim_mat = np.array(all_subjs_sim_mat)
        all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
        across_subj_var = np.var(all_subjs_sim_mat, axis=0)

        RESULTS[filter] = {}
        RESULTS[filter]['avg_mat'] = all_subjs_avg
        RESULTS[filter]['var_mat'] = across_subj_var
        RESULTS[filter]['avg_div_var_mat'] = np.divide(all_subjs_avg, across_subj_var, out=np.zeros_like(all_subjs_avg), where=across_subj_var!=0)
        RESULTS[filter]['var_div_avg_mat'] = np.divide(across_subj_var, all_subjs_avg, out=np.zeros_like(across_subj_var), where=all_subjs_avg!=0)
        RESULTS[filter]['name_lst'] = measure_name_lst

    ############ VISUALIZE ############
    for key in RESULTS[filter]:
        if key=='name_lst':
            continue
        visualize_sim_mat(RESULTS, mat_key=key, title=metric+' '+key, 
                                        name_lst_key='name_lst', 
                                        cmap='viridis',
                                        save_image=save_image, output_root=output_root+'dFC_similarity/'+metric+'/'
        )
    ############ Hierarchical Clustering ############
    for filter in ['default_values']:
        if metric=='MI':
            normalized_mat = np.divide(RESULTS[filter]['avg_mat'], np.max(RESULTS[filter]['avg_mat']))
            dist_mat = 1 - normalized_mat
        elif metric=='euclidean_distance':
            dist_mat = RESULTS[filter]['avg_mat']
        else:
            dist_mat = 1 - RESULTS[filter]['avg_mat']
        dist_mat = 0.5*(dist_mat + dist_mat.T)
        # diagonal values of dist_mat must equal exactly zero
        np.fill_diagonal(dist_mat, 0)
        dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS[filter]['name_lst'], 
            title='Hierarchical Clustering of Methods ' + filter+' using '+metric, 
            save_image=save_image, output_root=output_root+'dFC_similarity/'+metric+'/'
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
            SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
            # SUBJs_output[filter]['feature_based'][feature2extract] = (sample, method, method)
            all_subjs_sim_mat.append(np.mean(SUBJs_output[filter]['feature_based'][feature2extract], axis=0))

        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        all_subjs_sim_mat = np.array(all_subjs_sim_mat)
        all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
        across_subj_var = np.var(all_subjs_sim_mat, axis=0)

        RESULTS[filter] = {}
        RESULTS[filter]['avg_mat'] = all_subjs_avg
        RESULTS[filter]['var_mat'] = across_subj_var
        RESULTS[filter]['avg_div_var_mat'] = np.divide(all_subjs_avg, across_subj_var, out=np.zeros_like(all_subjs_avg), where=across_subj_var!=0)
        RESULTS[filter]['var_div_avg_mat'] = np.divide(across_subj_var, all_subjs_avg, out=np.zeros_like(across_subj_var), where=all_subjs_avg!=0)
        RESULTS[filter]['name_lst'] = measure_name_lst

    ############ VISUALIZE ############
    for key in RESULTS[filter]:
        if key=='name_lst':
            continue
        visualize_sim_mat(RESULTS, mat_key=key, title=feature2extract+' '+key, 
                                        name_lst_key='name_lst', 
                                        cmap='viridis',
                                        save_image=save_image, output_root=output_root+'feature_based/'+feature2extract+'/'
        )
    ############ Hierarchical Clustering ############
    for filter in ['default_values']:
        dist_mat = 1 - RESULTS[filter]['avg_mat']
        dist_mat = 0.5*(dist_mat + dist_mat.T)
        # diagonal values of dist_mat must equal exactly zero
        np.fill_diagonal(dist_mat, 0)
        dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS[filter]['name_lst'], 
            title='Hierarchical Clustering of Methods ' + filter+' using '+feature2extract, 
            save_image=save_image, output_root=output_root+'feature_based/'+feature2extract+'/'
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
            SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
            # SUBJs_output[filter]['graph_based']['graph_spatial'][graph_property] = (sample, method, method)
            all_subjs_sim_mat.append(np.mean(SUBJs_output[filter]['graph_based']['graph_spatial'][graph_property], axis=0))

        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        all_subjs_sim_mat = np.array(all_subjs_sim_mat)
        all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
        across_subj_var = np.var(all_subjs_sim_mat, axis=0)

        RESULTS[filter] = {}
        RESULTS[filter]['avg_mat'] = all_subjs_avg
        RESULTS[filter]['var_mat'] = across_subj_var
        RESULTS[filter]['avg_div_var_mat'] = np.divide(all_subjs_avg, across_subj_var, out=np.zeros_like(all_subjs_avg), where=across_subj_var!=0)
        RESULTS[filter]['var_div_avg_mat'] = np.divide(across_subj_var, all_subjs_avg, out=np.zeros_like(across_subj_var), where=all_subjs_avg!=0)
        RESULTS[filter]['name_lst'] = measure_name_lst

    ############ VISUALIZE ############
    for key in RESULTS[filter]:
        if key=='name_lst':
            continue
        visualize_sim_mat(RESULTS, mat_key=key, title='spatial '+graph_property+' '+key, 
                                        name_lst_key='name_lst', 
                                        cmap='viridis',
                                        save_image=save_image, output_root=output_root+'graph_based/'+graph_property+'/'
        )
    ############ Hierarchical Clustering ############
    for filter in ['default_values']:
        dist_mat = 1 - RESULTS[filter]['avg_mat']
        dist_mat = 0.5*(dist_mat + dist_mat.T)
        # diagonal values of dist_mat must equal exactly zero
        np.fill_diagonal(dist_mat, 0)
        dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS[filter]['name_lst'], 
            title='Hierarchical Clustering of Methods ' + filter+' using '+ 'spatial '+ graph_property, 
            save_image=save_image, output_root=output_root+'graph_based/'+graph_property+'/'
        )
###### temporal #####
for graph_property in graph_property_list:

    RESULTS = {}
    for filter in FILTERS:

        all_subjs_sim_mat = list()
        for s in ALL_RECORDS:
            SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
            # SUBJs_output[filter]['graph_based']['graph_temporal'][graph_property] = (1, method, method)
            all_subjs_sim_mat.append(np.squeeze(SUBJs_output[filter]['graph_based']['graph_temporal'][graph_property]))

        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
        all_subjs_sim_mat = np.array(all_subjs_sim_mat)
        all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
        across_subj_var = np.var(all_subjs_sim_mat, axis=0)

        RESULTS[filter] = {}
        RESULTS[filter]['avg_mat'] = all_subjs_avg
        RESULTS[filter]['var_mat'] = across_subj_var
        RESULTS[filter]['avg_div_var_mat'] = np.divide(all_subjs_avg, across_subj_var, out=np.zeros_like(all_subjs_avg), where=across_subj_var!=0)
        RESULTS[filter]['var_div_avg_mat'] = np.divide(across_subj_var, all_subjs_avg, out=np.zeros_like(across_subj_var), where=all_subjs_avg!=0)
        RESULTS[filter]['name_lst'] = measure_name_lst

    ############ VISUALIZE ############
    for key in RESULTS[filter]:
        if key=='name_lst':
            continue
        visualize_sim_mat(RESULTS, mat_key=key, title='temporal '+graph_property+' '+key, 
                                        name_lst_key='name_lst', 
                                        cmap='viridis',
                                        save_image=save_image, output_root=output_root+'graph_based/'+graph_property+'/'
        )
    ############ Hierarchical Clustering ############
    for filter in ['default_values']:
        dist_mat = 1 - RESULTS[filter]['avg_mat']
        dist_mat = 0.5*(dist_mat + dist_mat.T)
        # diagonal values of dist_mat must equal exactly zero
        np.fill_diagonal(dist_mat, 0)
        dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS[filter]['name_lst'], 
            title='Hierarchical Clustering of Methods ' + filter+' using '+ 'temporal '+ graph_property, 
            save_image=save_image, output_root=output_root+'graph_based/'+graph_property+'/'
        )

################################# inter_subject similarity #################################

'''
    - returns correspondence of inter-subject relation between results of dFC 
        measures in each session
    - dendogram based on inter-subject similarity
'''
subj_lvl_feature_lst = [
    'dFC_values',
    'FO',
    'dFC_var'
]
for subj_lvl_feature in subj_lvl_feature_lst:
    RESULTS = {}
    inter_subj_sim_sessions = list()
    session_name_lst = list()
    for filter in FILTERS:
        if filter=='default_values':
            session_name_lst.append('session_Rest1_LR')
        elif 'session' in filter:
            session_name_lst.append(filter)
        features_subj = list()
        inter_subj_sim_session = list()
        subj_name_lst = list()
        for s in ALL_RECORDS:
            SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
            subj_name_lst.append(SUBJs_output[filter]['TS_info_lst'][0]['subj_id_lst'][0])

            features_i = list()
            for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):

                if subj_lvl_feature=='dFC_values':
                    dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]
                    # rank normalization
                    dFC_mat_i = rank_norm(dFC_mat_i) 
                    features_i.append(dFC_mat2vec(dFC_mat_i).flatten())
                elif subj_lvl_feature=='dFC_var':
                    dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]
                    # rank normalization
                    # dFC_mat_i = rank_norm(dFC_mat_i) 
                    dFC_var = np.var(dFC_mat_i, axis=0)
                    dFC_var = rank_norm(dFC_var)
                    dFC_var = cat_data(dFC_var, N=10)
                    dFC_var = np.where(dFC_var == np.max(dFC_var), 1, 0)
                    features_i.append(dFC_mat2vec(dFC_var))
                elif subj_lvl_feature=='FO':
                    FO = SUBJs_output[filter]['FO'][i]
                    if not measure_i.is_state_based:
                        continue
                    else:
                        FO = [FO[FCS] for FCS in FO]
                        FO = np.array(FO)
                    features_i.append(FO)
            features_i = np.array(features_i)
            features_subj.append(features_i)

        features_subj = np.array(features_subj) # features_subj = (subj, method, subj_lvl_feature)
        num_subj = features_subj.shape[0]

        for i in range(features_subj.shape[1]):

            features_i = np.squeeze(features_subj[:,i,:]) # features_i = (subj, subj_lvl_feature)

            inter_subj_sim_i = np.corrcoef(features_i) # inter_subj_sim_i = (subj, subj)
            # inter_subj_sim_i, p_value = stats.spearmanr(features_i, axis=1) # inter_subj_sim_i = (subj, subj)

            inter_subj_sim_i = dFC_mat2vec(inter_subj_sim_i)

            inter_subj_sim_session.append(inter_subj_sim_i)

        inter_subj_sim_session = np.array(inter_subj_sim_session) # (method, inter_subj_sim_values)

        inter_subj_sim_sessions.append(inter_subj_sim_session)

    inter_subj_sim_sessions = np.array(inter_subj_sim_sessions) # (session, method, inter_subj_sim_values)

    if subj_lvl_feature=='dFC_values':
        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
    elif subj_lvl_feature=='FO': 
        measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst'] if measure.is_state_based]

    RESULTS['across_session'] = {}
    for measure_id, measure_name in enumerate(measure_name_lst):
        sim_mat = np.zeros((inter_subj_sim_sessions.shape[0], inter_subj_sim_sessions.shape[0]))
        for session_i in range(inter_subj_sim_sessions.shape[0]):
            for session_j in range(inter_subj_sim_sessions.shape[0]):
                spear_coef, p_value = stats.spearmanr(inter_subj_sim_sessions[session_i, measure_id,:], inter_subj_sim_sessions[session_j, measure_id,:])
                sim_mat[session_i, session_j] = spear_coef

        RESULTS['across_session'][measure_name] = {}
        RESULTS['across_session'][measure_name]['sim_mat'] = sim_mat
        RESULTS['across_session'][measure_name]['name_lst'] = session_name_lst

    RESULTS['across_method'] = {}
    for session_id, session in enumerate(session_name_lst):
        sim_mat = np.zeros((len(measure_name_lst), len(measure_name_lst)))
        for i in range(len(measure_name_lst)):
            for j in range(len(measure_name_lst)):

                spear_coef, p_value = stats.spearmanr(inter_subj_sim_sessions[session_id, i,:], inter_subj_sim_sessions[session_id, j,:])
                sim_mat[i, j] = spear_coef

        RESULTS['across_method'][session] = {}
        RESULTS['across_method'][session]['sim_mat'] = sim_mat
        RESULTS['across_method'][session]['name_lst'] = measure_name_lst

    RESULTS['method_divide_session'] = {}
    avg_across_method = np.mean(np.array([RESULTS['across_method'][session]['sim_mat'] for session in RESULTS['across_method']]), axis=0)
    avg_across_session = np.zeros((len(measure_name_lst), len(measure_name_lst)))
    for measure_i, measure_name_i in enumerate(measure_name_lst):
        for measure_j, measure_name_j in enumerate(measure_name_lst):
            across_session_measure_i = dFC_mat2vec(RESULTS['across_session'][measure_name_i]['sim_mat'])
            across_session_measure_j = dFC_mat2vec(RESULTS['across_session'][measure_name_j]['sim_mat'])
            avg_across_session[measure_i, measure_j] = np.mean(np.divide(across_session_measure_i + across_session_measure_j, 2))

    RESULTS['method_divide_session']['avg_across_method'] = {}
    RESULTS['method_divide_session']['avg_across_method']['sim_mat'] = avg_across_method
    RESULTS['method_divide_session']['avg_across_method']['name_lst'] = measure_name_lst

    RESULTS['method_divide_session']['avg_across_session'] = {}
    RESULTS['method_divide_session']['avg_across_session']['sim_mat'] = avg_across_session
    RESULTS['method_divide_session']['avg_across_session']['name_lst'] = measure_name_lst

    RESULTS['method_divide_session']['divide'] = {}
    RESULTS['method_divide_session']['divide']['sim_mat'] = np.divide(avg_across_method, avg_across_session, out=np.zeros_like(avg_across_method), where=avg_across_session!=0)
    np.fill_diagonal(RESULTS['method_divide_session']['divide']['sim_mat'], np.nan)
    RESULTS['method_divide_session']['divide']['name_lst'] = measure_name_lst

    common_key = 'subj_corr_diff_methods_'
    for session_id, session in enumerate(session_name_lst):
        RESULTS[common_key+session] = {}
        for measure_id, measure_name in enumerate(measure_name_lst):
            RESULTS[common_key+session][measure_name] = {}
            RESULTS[common_key+session][measure_name]['sim_mat'] = rank_norm(np.squeeze(dFC_vec2mat(np.expand_dims(inter_subj_sim_sessions[session_id, measure_id, :], axis=0), N=num_subj)))
            np.fill_diagonal(RESULTS[common_key+session][measure_name]['sim_mat'], np.nan)
            RESULTS[common_key+session][measure_name]['name_lst'] = subj_name_lst

    ############ VISUALIZE ############
    for key in RESULTS:
        annot = True
        if common_key in key:
            annot = False
        visualize_sim_mat(RESULTS[key], mat_key='sim_mat', title='inter-subject-corr similarity '+key+ ' based on '+subj_lvl_feature, 
                                        name_lst_key='name_lst', 
                                        cmap='viridis',
                                        annot=annot,
                                        save_image=save_image, output_root=output_root+'inter_subject/'+subj_lvl_feature+'/'
        )
    ############ Hierarchical Clustering ############
    for session in RESULTS['across_method']:
        dist_mat = 1 - RESULTS['across_method'][session]['sim_mat']
        dist_mat = 0.5*(dist_mat + dist_mat.T)
        # diagonal values of dist_mat must equal exactly zero
        np.fill_diagonal(dist_mat, 0)
        dist_mat_dendo(dist_mat=dist_mat, labels=RESULTS['across_method'][session]['name_lst'], 
        title='Hierarchical Clustering of Methods ' + filter+' using inter-subject similarity based on '+subj_lvl_feature, 
        save_image=save_image, output_root=output_root+'inter_subject/'+subj_lvl_feature+'/'
        )

################################# dFC var #################################

'''
    - avg(variance/fluctuations of dFC in one Subj)
'''
for filter in ['default_values']:
    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])
        n_regions = SUBJs_output[filter]['TS_info_lst'][0]['n_regions']

        for i, measure in enumerate(SUBJs_output[filter]['measure_lst']):
            if not measure.measure_name in RESULTS:
                RESULTS[measure.measure_name] = list()
            # SUBJs_output[filter]['dFC_var'][i] = (1, connection)
            var_mat = np.squeeze(dFC_vec2mat(SUBJs_output[filter]['dFC_var'][i], N=n_regions)) # (ROI, ROI)
            np.fill_diagonal(var_mat, 0)
            RESULTS[measure.measure_name].append(rank_norm(var_mat))

    for key in RESULTS:
        RESULTS[key] = np.array(RESULTS[key])
        RESULTS[key] = np.mean(RESULTS[key], axis=0)

    visualize_conn_mat_dict(RESULTS, node_networks=node_networks, 
                title='dFC var ' + filter, 
                fix_lim=False, disp_diag=True, cmap='jet', normalize=False, 
                save_image=save_image, output_root=output_root+'dFC_var/')

################################# dFC avg #################################

'''
    - avg(avg of dFC -static FC- in one Subj)
'''

for filter in ['default_values']:
    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])
        n_regions = SUBJs_output[filter]['TS_info_lst'][0]['n_regions']

        for i, measure in enumerate(SUBJs_output[filter]['measure_lst']):
            if not measure.measure_name in RESULTS:
                RESULTS[measure.measure_name] = list()
            # SUBJs_output[filter]['dFC_avg'][i] = (1, connection)
            avg_mat = np.squeeze(dFC_vec2mat(SUBJs_output[filter]['dFC_avg'][i], N=n_regions)) # (ROI, ROI)
            RESULTS[measure.measure_name].append(rank_norm(avg_mat))

    for key in RESULTS:
        RESULTS[key] = np.array(RESULTS[key])
        RESULTS[key] = np.mean(RESULTS[key], axis=0)

    visualize_conn_mat_dict(RESULTS, node_networks=node_networks, 
            title='dFC avg ' + filter, 
            fix_lim=False, disp_diag=False, cmap='jet', normalize=False,
            save_image=save_image, output_root=output_root+'dFC_avg/')

################################# Across Node Temporal Correlation #################################

'''
    - spearman_corr((dFConnection(node_i, node_j) timecourse using method m), (dFConnection(node_i, node_j) timecourse using method n))
'''
for filter in ['default_values']:
    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])
        n_regions = SUBJs_output[filter]['TS_info_lst'][0]['n_regions']

        # SUBJs_output[filter]['feature_based']['temporal'] = (connection, method, method)

        for i in range(SUBJs_output[filter]['feature_based']['temporal'].shape[1]):
            for j in range(i):
                measure_name_i = zip_name(SUBJs_output[filter]['measure_lst'][i].measure_name)
                measure_name_j = zip_name(SUBJs_output[filter]['measure_lst'][j].measure_name)
                if not measure_name_i in RESULTS:
                    RESULTS[measure_name_i] = {}
                if not measure_name_j in RESULTS[measure_name_i]:
                    RESULTS[measure_name_i][measure_name_j] = list()
                mat = np.squeeze(
                    dFC_vec2mat(
                        np.expand_dims(
                            SUBJs_output[filter]['feature_based']['temporal'][:, i, j], 
                            axis=0
                        ), 
                        N=n_regions
                    )
                ) # (ROI, ROI)
                RESULTS[measure_name_i][measure_name_j].append(mat)

    for key_i in RESULTS:
        for key_j in RESULTS[key_i]:
            RESULTS[key_i][key_j] = np.array(RESULTS[key_i][key_j])
            RESULTS[key_i][key_j] = np.mean(RESULTS[key_i][key_j], axis=0)
            
    visualize_conn_mat_2D_dict(RESULTS, node_networks=node_networks, 
        title='across node temporal spearman corr ' + filter, fix_lim=False, 
        disp_diag=False, cmap='jet', normalize=False,
        save_image=save_image, output_root=output_root+'across_node/'
        )

################################# High Variation Regions #################################
'''
    - high variation regions over methods and over time.
'''
for filter in ['default_values']:
    var_over_time = list()
    var_over_method = list()
    for s in ALL_RECORDS:

        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])

        dFC_mat_lst = list()
        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):

            dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]

            # rank normalization
            dFC_mat_i = rank_norm(dFC_mat_i)

            # dFC mat
            var_over_time.append(np.var(dFC_mat_i, axis=0))
            dFC_mat_lst.append(dFC_mat_i)

        dFC_mat_lst = np.array(dFC_mat_lst)
        var_over_method.append(np.mean(np.var(dFC_mat_lst, axis=0), axis=0))

    var_over_time = np.array(var_over_time) # (subj*method, ROI, ROI)
    var_over_time = np.mean(var_over_time, axis=0) # (ROI, ROI)
    var_over_method = np.array(var_over_method) # (subj, ROI, ROI)
    var_over_method = np.mean(var_over_method, axis=0) # (ROI, ROI)

    RESULTS = {}
    # RESULTS['var_over_time'] = rank_norm(var_over_time)
    # RESULTS['var_over_method'] = rank_norm(var_over_method)
    RESULTS['var_over_time'] = np.divide(var_over_time, np.max(var_over_time))
    RESULTS['var_over_method'] = np.divide(var_over_method, np.max(var_over_method))
    RESULTS['var_over_method/var_over_time'] = np.divide(var_over_method, var_over_time) - 1
    RESULTS['var_over_method*var_over_time'] = np.multiply(var_over_method, var_over_time)
    for key in RESULTS:
        RESULTS[key] = rank_norm(RESULTS[key])
        RESULTS[key] = cat_data(RESULTS[key], N=10)
        RESULTS[key] = np.where(RESULTS[key] == np.max(RESULTS[key]), 1, 0)

############ VISUALIZE ############

    visualize_conn_mat_dict(RESULTS, node_networks=node_networks, 
        title='high variation regions '+filter, fix_lim=False, 
        disp_diag=True, cmap='plasma', center_0=False,
        save_image=save_image, output_root=output_root+'variation/'
    )

################################# Variation Value Comparison #################################
'''
    - compare variation over methods with variation over time.
'''
RESULTS = {}
for filter in ['default_values']:
    diff_mat_dict = {}
    temp_var_dict = {}
    lag_1_diff_mat_dict = {}
    for s in ALL_RECORDS:

        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):
            measure_name_i = measure_i.measure_name

            dFC_mat_i = rank_norm(SUBJs_output[filter]['dFCM_samples'][str(i)])

            for j, measure_j in enumerate(SUBJs_output[filter]['measure_lst']):
                measure_name_j = measure_j.measure_name

                dFC_mat_j = rank_norm(SUBJs_output[filter]['dFCM_samples'][str(j)])

                if not measure_name_i in diff_mat_dict:
                    diff_mat_dict[measure_name_i] = {}
                    temp_var_dict[measure_name_i] = {}
                    lag_1_diff_mat_dict[measure_name_i] = {}
                if not measure_name_j in diff_mat_dict[measure_name_i]:
                    diff_mat_dict[measure_name_i][measure_name_j] = list()
                    temp_var_dict[measure_name_i][measure_name_j] = list()
                    lag_1_diff_mat_dict[measure_name_i][measure_name_j] = list()

                dFC = np.concatenate((dFC_mat_i[None,:,:,:], dFC_mat_j[None,:,:,:]), axis=0)
                diff_mat_dict[measure_name_i][measure_name_j].append(np.mean(np.var(dFC, axis=0), axis=0))
                temp_var_dict[measure_name_i][measure_name_j].append(np.mean(np.var(dFC, axis=1), axis=0))
                lag_1_diff_mat_dict[measure_name_i][measure_name_j].append(np.mean(np.mean(0.25*(np.diff(dFC, axis=1)**2), axis=1), axis=0))

    measure_name_lst = [measure_key_i for measure_key_i in diff_mat_dict]

    divide_temp = np.zeros((len(measure_name_lst),len(measure_name_lst)))
    divide_1_lag = np.zeros((len(measure_name_lst),len(measure_name_lst)))
    for i, measure_key_i in enumerate(measure_name_lst):
        for j, measure_key_j in enumerate(measure_name_lst):

            A = np.mean(np.array(diff_mat_dict[measure_key_i][measure_key_j]), axis=0)
            B = np.mean(np.array(temp_var_dict[measure_key_i][measure_key_j]), axis=0)
            C = np.mean(np.array(lag_1_diff_mat_dict[measure_key_i][measure_key_j]), axis=0)

            divide_temp[i, j] = np.mean(np.divide(A, B, out=np.zeros_like(A), where=B!=0))
            divide_1_lag[i, j] = np.mean(np.divide(A, C, out=np.zeros_like(A), where=C!=0))

    RESULTS['divide_temp'] = {'sim_mat': divide_temp, 'name_lst': measure_name_lst}
    RESULTS['divide_1_lag'] = {'sim_mat': divide_1_lag, 'name_lst': measure_name_lst}

############ VISUALIZE ############

    visualize_sim_mat(RESULTS, mat_key='sim_mat', title='variation in different dimensions '+filter, 
                                    name_lst_key='name_lst', 
                                    cmap='viridis',
                                    save_image=save_image, output_root=output_root+'variation/'
    )

################################# Similarity in different Variation Levels #################################
'''
    - measure spearman correlation similarity in different variation levels.
'''
num_var_band = 10

RESULTS = {}
for filter in ['default_values']:
    RESULTS['sim'] = {}
    RESULTS['sim']['sim_mat'] = list()
    for n in range(1, num_var_band+1):
        RESULTS['sim_high_var'+str(n)] = {}
        RESULTS['sim_high_var'+str(n)]['sim_mat'] = list()
    for s in ALL_RECORDS:

        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        sim_mat = np.zeros((len(SUBJs_output[filter]['measure_lst']), len(SUBJs_output[filter]['measure_lst'])))
        sim_mat_high_var = np.zeros((num_var_band, len(SUBJs_output[filter]['measure_lst']), len(SUBJs_output[filter]['measure_lst'])))
        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):

            dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]

            temp_var_mat_i = np.var(dFC_mat_i, axis=0)

            for j, measure_j in enumerate(SUBJs_output[filter]['measure_lst']):

                dFC_mat_j = SUBJs_output[filter]['dFCM_samples'][str(j)]

                # all similarity
                sim, p = stats.spearmanr(dFC_mat_i.flatten(), dFC_mat_j.flatten())
                sim_mat[i, j] = sim

                # var band similarity
                temp_var_mat_j = np.var(dFC_mat_j, axis=0)
                temp_var_mat = np.divide(temp_var_mat_i + temp_var_mat_j, 2)
                for n in range(1, num_var_band+1):
                    var_mask = rank_norm(temp_var_mat)
                    var_mask = cat_data(var_mask, N=num_var_band)
                    var_mask = np.where(var_mask == n, 1, 0) # (roi, roi)
                    # high_var_func_conns is not syymetric!
                    var_mask = np.divide(var_mask + var_mask.T, 2)
                    var_mask = np.where(var_mask == 0, 0, 1)

                    masked_i = dFC_mask(dFC_mat_i, var_mask==1)
                    masked_j = dFC_mask(dFC_mat_j, var_mask==1)

                    sim, p = stats.spearmanr(masked_i.flatten(), masked_j.flatten())

                    sim_mat_high_var[n-1, i, j] = sim

        RESULTS['sim']['sim_mat'].append(sim_mat)
        for n in range(1,num_var_band+1):
            RESULTS['sim_high_var'+str(n)]['sim_mat'].append(sim_mat_high_var[n-1,:,:])

    for key in RESULTS:
        RESULTS[key]['sim_mat'] = np.mean(np.array(RESULTS[key]['sim_mat']), axis=0)
        RESULTS[key]['name_lst'] = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]

############ VISUALIZE ############

    visualize_sim_mat(RESULTS, mat_key='sim_mat', title='Similarity in different Variation Levels '+filter, 
                                    name_lst_key='name_lst', 
                                    cmap='viridis',
                                    save_image=save_image, output_root=output_root+'variation/'
    )

################################# Similarity inter Time vs. Method #################################
'''
    - compare spearman correlation similarity between consecutive time points and between methods.
'''
RESULTS = {}
for filter in ['default_values']:
    RESULTS['sim'] = {}
    RESULTS['sim']['sim_mat'] = list()
    RESULTS['sim_mat_across_method'] = {}
    RESULTS['sim_mat_across_method']['sim_mat'] = list()
    RESULTS['sim_mat_across_time'] = {}
    RESULTS['sim_mat_across_time']['sim_mat'] = list()
    for s in ALL_RECORDS:

        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        n_time = SUBJs_output[filter]['dFCM_samples'][str(0)].shape[0]

        sim_mat = np.zeros((len(SUBJs_output[filter]['measure_lst']), len(SUBJs_output[filter]['measure_lst'])))
        sim_mat_across_method = np.zeros((n_time-1, len(SUBJs_output[filter]['measure_lst']), len(SUBJs_output[filter]['measure_lst'])))
        sim_mat_across_time = np.zeros((n_time-1, len(SUBJs_output[filter]['measure_lst']), len(SUBJs_output[filter]['measure_lst'])))
        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):

            dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]

            for j, measure_j in enumerate(SUBJs_output[filter]['measure_lst']):

                dFC_mat_j = SUBJs_output[filter]['dFCM_samples'][str(j)]

                # all similarity
                sim, p = stats.spearmanr(dFC_mat_i.flatten(), dFC_mat_j.flatten())
                sim_mat[i, j] = sim

                for t in range(n_time-1):
                    sim, p = stats.spearmanr(dFC_mat_i[t,:,:].flatten(), dFC_mat_j[t,:,:].flatten())
                    sim_mat_across_method[t, i, j] = sim
                    sim_i, p = stats.spearmanr(dFC_mat_i[t,:,:].flatten(), dFC_mat_i[t+1,:,:].flatten())
                    sim_j, p = stats.spearmanr(dFC_mat_j[t,:,:].flatten(), dFC_mat_j[t+1,:,:].flatten())
                    sim_mat_across_time[t, i, j] = (sim_i + sim_j) / 2

        RESULTS['sim']['sim_mat'].append(sim_mat)
        RESULTS['sim_mat_across_method']['sim_mat'].append(np.mean(sim_mat_across_method, axis=0))
        RESULTS['sim_mat_across_time']['sim_mat'].append(np.mean(sim_mat_across_time, axis=0))

    measure_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
    for key in RESULTS:
        RESULTS[key]['sim_mat'] = np.mean(np.array(RESULTS[key]['sim_mat']), axis=0)
        RESULTS[key]['name_lst'] = measure_lst

    RESULTS['divide_method_time'] = {'sim_mat': np.divide(RESULTS['sim_mat_across_method']['sim_mat'], RESULTS['sim_mat_across_time']['sim_mat']) - 1, 'name_lst': measure_lst}

############ VISUALIZE ############

    visualize_sim_mat(RESULTS, mat_key='sim_mat', title='Similarity inter Time vs. Method '+filter, 
                                    name_lst_key='name_lst', 
                                    cmap='viridis',
                                    save_image=save_image, output_root=output_root+'variation/'
    )

################################# TIME RECORD #################################

for filter in ['default_values']:

    print('********** time record of ' + filter + '**********')

    avg_FCS_fit = {}
    avg_dFC_assess = {}
    # for SUBJs_output in SUBJs_output_lst:
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

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
