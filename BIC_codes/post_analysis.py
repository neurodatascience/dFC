
import sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os

sys.path.append('./BIC_codes/')
from functions.dFC_funcs import *
from functions.post_analysis_funcs import *

print('################################# POST ANALYSIS STARTED RUNNING ... #################################')

################################# LOAD RESULTS #################################

# assessment_results_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/'
assessment_results_root = './'
# output_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/out/'
output_root = './output/'
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

# the dictionary that collects all RESULTS
ALL_RESULTS = {} 

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

################################# dFC values distributions #################################

RESULTS = {}
for filter in ['default_values']:
    for s in ALL_RECORDS[:100]:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):

            dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]

            if not measure_i.measure_name in RESULTS:
                RESULTS[measure_i.measure_name] = list()
            RESULTS[measure_i.measure_name].append(dFC_mat2vec(dFC_mat_i).flatten())

    for measure in RESULTS:
        RESULTS[measure] = np.array(RESULTS[measure]).flatten()

    ALL_RESULTS['dFC_values_dist'] = deepcopy(RESULTS)

############ VISUALIZE ############

    joint_dist_plot(data=RESULTS,
        title='dFC values distributions',
        save_image=save_image, output_root=output_root+'indiv_prop/'
        )

################################# dFC Similarity #################################

################# whole subject #################
'''
    - corr((all dFC timepoints of one Subj using method_i), (all dFC timepoints of one Subj using method_j)) -> avg(corr) and var(corr)
    - spearman correlation, pearson correlation, Mutual Information (MI), Euclidean Distance
    - dendogram based on avg(corr)
'''
ALL_RESULTS['dFC_similarity_overall'] = {}

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
        sim_distribution = make_sim_distribution(
            sim_mats_lst=all_subjs_sim_mat, 
            name_lst=measure_name_lst,
            zip_names=True
        )
        all_subjs_sim_mat = np.array(all_subjs_sim_mat)
        all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
        across_subj_var = np.var(all_subjs_sim_mat, axis=0)

        # change default_values to session_Rest1_LR
        if filter=='default_values':
            new_filter = 'session_Rest1_LR'
            RESULTS[new_filter] = {}
            RESULTS[new_filter]['avg_mat'] = all_subjs_avg
            RESULTS[new_filter]['var_mat'] = across_subj_var
            RESULTS[new_filter]['avg_div_var_mat'] = np.divide(all_subjs_avg, np.sqrt(across_subj_var), out=np.zeros_like(all_subjs_avg), where=np.sqrt(across_subj_var)!=0)
            RESULTS[new_filter]['sim_distribution'] = sim_distribution
            RESULTS[new_filter]['name_lst'] = measure_name_lst
        else:
            RESULTS[filter] = {}
            RESULTS[filter]['avg_mat'] = all_subjs_avg
            RESULTS[filter]['var_mat'] = across_subj_var
            RESULTS[filter]['avg_div_var_mat'] = np.divide(all_subjs_avg, np.sqrt(across_subj_var), out=np.zeros_like(all_subjs_avg), where=np.sqrt(across_subj_var)!=0)
            RESULTS[filter]['sim_distribution'] = sim_distribution
            RESULTS[filter]['name_lst'] = measure_name_lst

    ALL_RESULTS['dFC_similarity_overall'][metric] = deepcopy(RESULTS)
    ############ VISUALIZE ############
    for key in RESULTS[filter]:
        if key=='name_lst' or key=='sim_distribution':
            continue
        visualize_sim_mat(RESULTS, mat_key=key, title=metric+' '+key, 
                                        name_lst_key='name_lst', 
                                        cmap='viridis',
                                        save_image=save_image, output_root=output_root+'dFC_similarity/'+metric+'/'
        )
    for filter in ['session_Rest1_LR']:
        pairwise_cat_plots(RESULTS[filter]['sim_distribution'], x='', y='sim',
            title=metric+' total similarity distributions '+filter,
            save_image=save_image, output_root=output_root+'dFC_similarity/'+metric+'/'
            )
    ############ Hierarchical Clustering ############
    for filter in ['session_Rest1_LR']:
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
ALL_RESULTS['dFC_similarity_feature_based'] = {}

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
        RESULTS[filter]['avg_div_var_mat'] = np.divide(all_subjs_avg, np.sqrt(across_subj_var), out=np.zeros_like(all_subjs_avg), where=np.sqrt(across_subj_var)!=0)
        RESULTS[filter]['name_lst'] = measure_name_lst

    ALL_RESULTS['dFC_similarity_feature_based'][feature2extract] = deepcopy(RESULTS)

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

############ Spatial vs. Temporal Scatter plot ############
    
for filter in ['default_values']:

    all_subjs_spatial_sim_mat = list()
    all_subjs_temporal_sim_mat = list()
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        # SUBJs_output[filter]['feature_based'][feature2extract] = (sample, method, method)
        all_subjs_spatial_sim_mat.append(np.mean(SUBJs_output[filter]['feature_based']['spatial'], axis=0))
        all_subjs_temporal_sim_mat.append(np.mean(SUBJs_output[filter]['feature_based']['temporal'], axis=0))

    measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
    all_subjs_spatial_sim_mat = np.mean(np.array(all_subjs_spatial_sim_mat), axis=0)
    all_subjs_temporal_sim_mat = np.mean(np.array(all_subjs_temporal_sim_mat), axis=0)

    scatter_data = {'spatial':list(), 'temporal':list(), 'labels':list()}
    for i in range(len(measure_name_lst)):
        for j in range(i):
            scatter_data['spatial'].append(all_subjs_spatial_sim_mat[i,j])
            scatter_data['temporal'].append(all_subjs_temporal_sim_mat[i,j])
            scatter_data['labels'].append(zip_name(measure_name_lst[i])+'-'+zip_name(measure_name_lst[j]))

    ############ visualization ############
    scatter_plot(
        data=scatter_data, x='temporal', y='spatial', 
        labels='labels', title='spatial similarity vs temporal similarity',
        save_image=save_image, output_root=output_root+'variation/'
    )

ALL_RESULTS['spatial_vs_temporal_similarity'] = deepcopy(scatter_data)
    
################# graph-based #################
'''
    - spatial
    - temporal-avg
    - ECM, shortest_path, degree, clustering_coef
'''

ALL_RESULTS['dFC_similarity_graph'] = {}

graph_property_list = [
            'ECM',
            'shortest_path',
            'degree',
            'clustering_coef'
]
###### spatial #####
ALL_RESULTS['dFC_similarity_graph']['spatial'] = {}

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
        RESULTS[filter]['avg_div_var_mat'] = np.divide(all_subjs_avg, np.sqrt(across_subj_var), out=np.zeros_like(all_subjs_avg), where=np.sqrt(across_subj_var)!=0)
        RESULTS[filter]['name_lst'] = measure_name_lst

    ALL_RESULTS['dFC_similarity_graph']['spatial'][graph_property] = deepcopy(RESULTS)

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

################################# inter_subject similarity #################################

'''
    - returns correspondence of inter-subject relation between results of dFC 
        measures in each session
    - dendogram based on inter-subject similarity
'''
ALL_RESULTS['subj_clustring'] = {}

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

    ALL_RESULTS['subj_clustring'][subj_lvl_feature] = deepcopy(RESULTS)

    ############ VISUALIZE ############
    for key in RESULTS:
        annot = True
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
            title='Hierarchical Clustering of Methods ' + session +' using inter-subject similarity based on '+subj_lvl_feature, 
            save_image=save_image, output_root=output_root+'inter_subject/'+subj_lvl_feature+'/'
        )

################################# dFC var #################################

'''
    - avg(variance/fluctuations of dFC in one Subj)
    - rank normed
'''
for filter in ['default_values']:
    RESULTS = {}
    RESULTS['avg_dFC_var'] = {}
    RESULTS['var_dFC_var'] = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])
        n_regions = SUBJs_output[filter]['TS_info_lst'][0]['n_regions']

        for i, measure in enumerate(SUBJs_output[filter]['measure_lst']):
            if not measure.measure_name in RESULTS['avg_dFC_var']:
                RESULTS['avg_dFC_var'][measure.measure_name] = list()
            # SUBJs_output[filter]['dFC_var'][i] = (1, connection)
            var_mat = np.squeeze(dFC_vec2mat(SUBJs_output[filter]['dFC_var'][i], N=n_regions)) # (ROI, ROI)
            np.fill_diagonal(var_mat, 0)
            RESULTS['avg_dFC_var'][measure.measure_name].append(rank_norm(var_mat))

    for key in RESULTS['avg_dFC_var']:
        RESULTS['avg_dFC_var'][key] = np.array(RESULTS['avg_dFC_var'][key])
        RESULTS['var_dFC_var'][key] = np.var(RESULTS['avg_dFC_var'][key], axis=0)
        RESULTS['avg_dFC_var'][key] = np.mean(RESULTS['avg_dFC_var'][key], axis=0)

    ALL_RESULTS['dFC_var'] = deepcopy(RESULTS)

    visualize_conn_mat_dict(RESULTS['avg_dFC_var'], node_networks=node_networks, 
                title='avg dFC var ' + filter, center_0=False,
                fix_lim=False, disp_diag=True, cmap='plasma', normalize=False, 
                save_image=save_image, output_root=output_root+'dFC_var/')

    visualize_conn_mat_dict(RESULTS['avg_dFC_var'], node_networks=node_networks, segmented=True,
                title='segmented avg dFC var ' + filter, center_0=False,
                fix_lim=False, disp_diag=True, cmap='plasma', normalize=False, 
                save_image=save_image, output_root=output_root+'dFC_var/')

    visualize_conn_mat_dict(RESULTS['var_dFC_var'], node_networks=node_networks, 
                title='var of dFC var ' + filter, center_0=False,
                fix_lim=False, disp_diag=True, cmap='plasma', normalize=False, 
                save_image=save_image, output_root=output_root+'dFC_var/')

################################# dFC avg #################################

'''
    - avg(avg of dFC -static FC- in one Subj)
    - rank normed
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

    ALL_RESULTS['dFC_avg'] = deepcopy(RESULTS)

    visualize_conn_mat_dict(RESULTS, node_networks=node_networks, 
            title='dFC avg ' + filter, center_0=False,
            fix_lim=False, disp_diag=False, cmap='plasma', normalize=False,
            save_image=save_image, output_root=output_root+'dFC_avg/')

    visualize_conn_mat_dict(RESULTS, node_networks=node_networks, segmented=True,
            title='segmented dFC avg ' + filter, center_0=False,
            fix_lim=False, disp_diag=False, cmap='plasma', normalize=False,
            save_image=save_image, output_root=output_root+'dFC_avg/')

################################# Across Func Conn total Correlation #################################

'''
    - spearman_corr((dFConnection(node_i, node_j) timecourse using method m), (dFConnection(node_i, node_j) timecourse using method n))
'''

for filter in ['default_values']:
    RESULTS = {}
    for s in ALL_RECORDS:

        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])
        n_regions = SUBJs_output[filter]['TS_info_lst'][0]['n_regions']
        n_time = SUBJs_output[filter]['dFCM_samples'][str(0)].shape[0]

        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):

            dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]
            # rank normalization
            dFC_mat_i_norm = stats.rankdata(dFC_mat_i.flatten()).reshape(n_time, n_regions, n_regions)
            dFC_mat_i_vec = dFC_mat2vec(dFC_mat_i_norm)

            for j, measure_j in enumerate(SUBJs_output[filter]['measure_lst']):

                if j >= i :
                    continue

                dFC_mat_j = SUBJs_output[filter]['dFCM_samples'][str(j)]
                # rank normalization
                dFC_mat_j_norm = stats.rankdata(dFC_mat_j.flatten()).reshape(n_time, n_regions, n_regions)
                dFC_mat_j_vec = dFC_mat2vec(dFC_mat_j_norm)

                sim = list()
                for func_conn in range(dFC_mat_i_vec.shape[1]):
                    if np.var(dFC_mat_i_vec[:,func_conn])==0 or np.var(dFC_mat_j_vec[:,func_conn])==0:
                        sim.append(0)
                    else:
                        sim.append(np.corrcoef(dFC_mat_i_vec[:,func_conn], dFC_mat_j_vec[:,func_conn])[0,1])

                sim = np.array(sim)
                if not measure_i.measure_name in RESULTS:
                    RESULTS[measure_i.measure_name] = {}
                if not measure_j.measure_name in RESULTS[measure_i.measure_name]:
                    RESULTS[measure_i.measure_name][measure_j.measure_name] = list()

                RESULTS[measure_i.measure_name][measure_j.measure_name].append(dFC_vec2mat(sim[None,:], N=n_regions)[0])

    measure_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
    for key_i in RESULTS:
        for key_j in RESULTS[key_i]:
            RESULTS[key_i][key_j] = np.mean(np.array(RESULTS[key_i][key_j]), axis=0)

    ALL_RESULTS['across_func_conns'] = deepcopy(RESULTS)

############ VISUALIZE ############

    visualize_conn_mat_2D_dict(RESULTS, node_networks=node_networks, 
        title='across node total spearman corr ' + filter, fix_lim=False, 
        disp_diag=False, cmap='seismic', normalize=False, center_0=True,
        save_image=save_image, output_root=output_root+'across_node/total/'
    )

    visualize_conn_mat_2D_dict(RESULTS, node_networks=node_networks, segmented=True,
        title='segmented across node total spearman corr ' + filter, fix_lim=False, 
        disp_diag=False, cmap='seismic', normalize=False, center_0=True,
        save_image=save_image, output_root=output_root+'across_node/total/'
    )

    visualize_conn_mat_2D_dict(RESULTS, node_networks=node_networks, 
        title='across node total spearman corr normalized ' + filter, fix_lim=False, 
        disp_diag=False, cmap='seismic', normalize=True, center_0=True,
        save_image=save_image, output_root=output_root+'across_node/total/'
    )

    visualize_conn_mat_2D_dict(RESULTS, node_networks=node_networks, segmented=True,
        title='segmented across node total spearman corr normalized ' + filter, fix_lim=False, 
        disp_diag=False, cmap='seismic', normalize=True, center_0=True,
        save_image=save_image, output_root=output_root+'across_node/total/'
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

    scatter_data = {'var_method':list(), 'var_time':list()}
    scatter_data['var_method'] = var_over_method.flatten()
    scatter_data['var_time'] = var_over_time.flatten()

    RESULTS = {}
    RESULTS['var_over_time'] = np.divide(var_over_time, np.max(var_over_time))
    RESULTS['var_over_method'] = np.divide(var_over_method, np.max(var_over_method))
    RESULTS['var_over_method/var_over_time'] = np.divide(var_over_method, var_over_time) - 1
    RESULTS['var_over_method*var_over_time'] = np.multiply(var_over_method, var_over_time)
    for key in RESULTS:
        RESULTS[key] = rank_norm(RESULTS[key])

    ALL_RESULTS['high_var_func_conns'] = deepcopy(RESULTS)

############ VISUALIZE ############

    scatter_plot(
        data=scatter_data, x='var_time', y='var_method', 
        title='var method vs time across func conns',
        hist=True,
        save_image=save_image, output_root=output_root+'variation/'
    )

    visualize_conn_mat_dict(RESULTS, node_networks=node_networks, 
        title='variation across regions '+filter, fix_lim=False, 
        disp_diag=True, cmap='plasma', center_0=False,
        save_image=save_image, output_root=output_root+'variation/'
    )

    # func conn segmented
    visualize_conn_mat_dict(RESULTS, node_networks=node_networks, segmented=True,
        title='segmented high variation regions '+filter, fix_lim=False, 
        disp_diag=True, cmap='plasma', center_0=False,
        save_image=save_image, output_root=output_root+'variation/'
    )

    for key in RESULTS:
        RESULTS[key] = cat_data(RESULTS[key], N=10)
        RESULTS[key] = np.where(RESULTS[key] == np.max(RESULTS[key]), 1, 0)

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

    scatter_data = {'var_method':list(), 'var_time':list(), 'labels':list()}
    scatter_data_across_func_conn = {}
    divide_temp = np.zeros((len(measure_name_lst),len(measure_name_lst)))
    divide_1_lag = np.zeros((len(measure_name_lst),len(measure_name_lst)))
    for i, measure_key_i in enumerate(measure_name_lst):
        for j, measure_key_j in enumerate(measure_name_lst):

            A = np.mean(np.array(diff_mat_dict[measure_key_i][measure_key_j]), axis=0)
            B = np.mean(np.array(temp_var_dict[measure_key_i][measure_key_j]), axis=0)
            C = np.mean(np.array(lag_1_diff_mat_dict[measure_key_i][measure_key_j]), axis=0)

            # collect data for scatter plot
            if j<i:
                if not zip_name(measure_key_i) in scatter_data_across_func_conn:
                    scatter_data_across_func_conn[zip_name(measure_key_i)] = {}
                if not zip_name(measure_key_j) in scatter_data_across_func_conn[zip_name(measure_key_i)]:
                    scatter_data_across_func_conn[zip_name(measure_key_i)][zip_name(measure_key_j)] = {'var_method':list(), 'var_time':list()}

                scatter_data_across_func_conn[zip_name(measure_key_i)][zip_name(measure_key_j)]['var_method'] = A.flatten()
                scatter_data_across_func_conn[zip_name(measure_key_i)][zip_name(measure_key_j)]['var_time'] = B.flatten()

                scatter_data['var_method'].append(np.mean(A))
                scatter_data['var_time'].append(np.mean(B))
                scatter_data['labels'].append(zip_name(measure_key_i)+'-'+zip_name(measure_key_j))

            divide_temp[i, j] = np.mean(np.divide(A, B, out=np.zeros_like(A), where=B!=0))
            divide_1_lag[i, j] = np.mean(np.divide(A, C, out=np.zeros_like(A), where=C!=0))

    RESULTS['divide_temp'] = {'sim_mat': divide_temp, 'name_lst': measure_name_lst}
    RESULTS['divide_1_lag'] = {'sim_mat': divide_1_lag, 'name_lst': measure_name_lst}

    ALL_RESULTS['var_comparison'] = deepcopy(RESULTS)

############ VISUALIZE ############

    visualize_sim_mat(RESULTS, mat_key='sim_mat', title='variation in different dimensions '+filter, 
                                    name_lst_key='name_lst', 
                                    cmap='viridis',
                                    save_image=save_image, output_root=output_root+'variation/'
    )

    pairwise_scatter_plots(
        data=scatter_data_across_func_conn, x='var_time', y='var_method', 
        title='var method vs time across func conns across methods pairs', hist=True,
        save_image=save_image, output_root=output_root+'variation/'
    )

    scatter_plot(
        data=scatter_data, x='var_time', y='var_method', 
        labels='labels', title='var method vs time',
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

    ALL_RESULTS['sim_across_diff_var_lvls'] = deepcopy(RESULTS)

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
    RESULTS['divide_method_time'] = {}
    RESULTS['divide_method_time']['sim_mat'] = list()
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
        RESULTS['divide_method_time']['sim_mat'].append(np.mean(np.divide(sim_mat_across_method, sim_mat_across_time, out=np.zeros_like(sim_mat_across_method), where=sim_mat_across_time!=0), axis=0))

    measure_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]
    for key in RESULTS:
        RESULTS[key]['sim_mat'] = np.mean(np.array(RESULTS[key]['sim_mat']), axis=0)
        RESULTS[key]['name_lst'] = measure_lst

    ALL_RESULTS['sim_inter_time_vs_method'] = deepcopy(RESULTS)

############ VISUALIZE ############

    visualize_sim_mat(RESULTS, mat_key='sim_mat', title='Similarity inter Time vs. Method '+filter, 
                                    name_lst_key='name_lst', 
                                    cmap='viridis',
                                    save_image=save_image, output_root=output_root+'variation/'
    )

    data = {'sim_mat_across_method':list(), 'sim_mat_across_time':list(), 'labels':list()}
    for i in range(RESULTS['sim_mat_across_method']['sim_mat'].shape[0]):
        for j in range(i):
            data['sim_mat_across_method'].append(RESULTS['sim_mat_across_method']['sim_mat'][i,j])
            data['sim_mat_across_time'].append(RESULTS['sim_mat_across_time']['sim_mat'][i,j])
            data['labels'].append(zip_name(measure_lst[i])+'-'+zip_name(measure_lst[j]))

    scatter_plot(data, x='sim_mat_across_method', y='sim_mat_across_time',
        labels='labels', title='scatter inter time vs method',
        save_image=save_image, output_root=output_root+'variation/'
    )

    ALL_RESULTS['sim_inter_time_vs_method_scatter'] = deepcopy(data)

################################# Randomization Tests #################################

metric = 'spearman'

ALL_RESULTS['randomization'] = {}

'''
find the similarity between the dFC obtained by each method 
and a dFC which is a constant sequence of mean of all methods dFC
'''
for filter in ['default_values']:

    RESULTS = {}
    RESULTS['sim'] = list()
    RESULTS['dFC_method'] = list()
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        dFC_dict = {}
        for i, measure in enumerate(SUBJs_output[filter]['measure_lst']):
            dFC_mat = SUBJs_output[filter]['dFCM_samples'][str(i)]
            dFC_dict[measure.measure_name] = dFC_mat

        dFC_mean = list()
        for i, measure_name in enumerate(dFC_dict):
            dFC_mat = dFC_dict[measure_name]
            n_time = dFC_mat.shape[0]
            dFC_mean.append(np.mean(dFC_mat, axis=0))
        dFC_mean = np.mean(np.array(dFC_mean), axis=0)
        dFC_mean = np.repeat(dFC_mean[None,:,:], n_time, axis=0)
        dFC_mean_vec = dFC_mat2vec(dFC_mean)

        for i, measure_i_name in enumerate(dFC_dict):

            dFC_mat_i = dFC_dict[measure_i_name]
            dFC_mat_i_vec = dFC_mat2vec(dFC_mat_i)

            sim, p = stats.spearmanr(dFC_mat_i_vec.flatten(), dFC_mean_vec.flatten())
            
            RESULTS['sim'].append(sim)
            RESULTS['dFC_method'].append(measure_i_name)

    ALL_RESULTS['randomization']['sim_with_static_FC'] = deepcopy(RESULTS)

############ VISUALIZE ############
    cat_plot(data=RESULTS, x='dFC_method', y='sim', 
            kind='violin',
            title='similarity with constant static FC',
            save_image=save_image, output_root=output_root+'randomization/'
            )


'''
find the similarity between the dFC obtained by each method 
but with randomized temporal order
'''
for filter in ['default_values']:

    all_subjs_sim_mat = list()
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        # SUBJs_output[filter]['all']['metric'] = (1, method, method)
        all_subjs_sim_mat.append(np.squeeze(SUBJs_output[filter]['all'][metric]))

    all_subjs_sim_mat = np.array(all_subjs_sim_mat)
    all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
    measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]

    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        dFC_dict = {}
        for i, measure in enumerate(SUBJs_output[filter]['measure_lst']):

            dFC_mat = SUBJs_output[filter]['dFCM_samples'][str(i)]
            dFC_dict[zip_name(measure.measure_name)] = dFC_mat

        output = randomize_time(dFC_dict, N=100)

        for measure_i_name in output:
            for measure_j_name in output[measure_i_name]:

                if not measure_i_name in RESULTS:
                    RESULTS[measure_i_name] = {}
                if not measure_j_name in RESULTS[measure_i_name]:
                    RESULTS[measure_i_name][measure_j_name] = {'sim':list(), '':list()}
            
                RESULTS[measure_i_name][measure_j_name]['sim'].extend(output[measure_i_name][measure_j_name]['sim'])
                RESULTS[measure_i_name][measure_j_name][''].extend(output[measure_i_name][measure_j_name][''])

    for measure_i_name in output:
        for measure_j_name in output[measure_i_name]:
            sim = all_subjs_avg[measure_name_lst.index(unzip_name(measure_i_name)), measure_name_lst.index(unzip_name(measure_j_name))]
            # change diagonal from 0 to 1
            if measure_i_name==measure_j_name:
                sim = 1
            RESULTS[measure_i_name][measure_j_name]['actual_sim'] = [sim for item in RESULTS[measure_i_name][measure_j_name]['sim']]

    ALL_RESULTS['randomization']['shuffled_time'] = deepcopy(RESULTS)

############ VISUALIZE ############
    pairwise_cat_plots(RESULTS, x='', y='sim', z='actual_sim',
        title='randomized time similarity',
        save_image=save_image, output_root=output_root+'randomization/'
        )

'''
find the similarity between dFC matrices created using
a random state time course but real FC patterns obtained 
by each method; for SW and TF all FC patterns of each 
subject are used
'''
for filter in ['default_values']:

    all_subjs_sim_mat = list()
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        # SUBJs_output[filter]['all']['metric'] = (1, method, method)
        all_subjs_sim_mat.append(np.squeeze(SUBJs_output[filter]['all'][metric]))

    all_subjs_sim_mat = np.array(all_subjs_sim_mat)
    all_subjs_avg = np.mean(all_subjs_sim_mat, axis=0)
    measure_name_lst = [measure.measure_name for measure in SUBJs_output[filter]['measure_lst']]

    RESULTS = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        n_time = SUBJs_output[filter]['dFCM_samples'][str(0)].shape[0]
        
        FCS_dict = {}
        for i, measure in enumerate(SUBJs_output[filter]['measure_lst']):
            FCS = measure.FCS
            if len(FCS)==0:
                FCS = SUBJs_output[filter]['dFCM_samples'][str(i)]
            FCS_dict[zip_name(measure.measure_name)] = FCS

        output = dFC_rand_sim(FCS_dict, n_time, N=100)

        for measure_i_name in output:
            for measure_j_name in output[measure_i_name]:

                if not measure_i_name in RESULTS:
                    RESULTS[measure_i_name] = {}
                if not measure_j_name in RESULTS[measure_i_name]:
                    RESULTS[measure_i_name][measure_j_name] = {'sim':list(), '':list()}
            
                RESULTS[measure_i_name][measure_j_name]['sim'].extend(output[measure_i_name][measure_j_name]['sim'])
                RESULTS[measure_i_name][measure_j_name][''].extend(output[measure_i_name][measure_j_name][''])

    for measure_i_name in output:
        for measure_j_name in output[measure_i_name]:
            sim = all_subjs_avg[measure_name_lst.index(unzip_name(measure_i_name)), measure_name_lst.index(unzip_name(measure_j_name))]
            # change diagonal from 0 to 1
            if measure_i_name==measure_j_name:
                sim = 1
            RESULTS[measure_i_name][measure_j_name]['actual_sim'] = [sim for item in RESULTS[measure_i_name][measure_j_name]['sim']]

    ALL_RESULTS['randomization']['random_state_TC'] = deepcopy(RESULTS)

############ VISUALIZE ############
    pairwise_cat_plots(RESULTS, x='', y='sim', z='actual_sim',
        title='random state time course dFC',
        save_image=save_image, output_root=output_root+'randomization/'
        )

################################# SIMILARITY OF ADJACENT TIME POINTS #################################

RESULTS = {}
key_name = 'similarity of adjacent time points'
for filter in ['default_values']:
    RESULTS[key_name] = list()
    RESULTS['dFC_method'] = list()
    for s in ALL_RECORDS:

        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        n_time = SUBJs_output[filter]['dFCM_samples'][str(0)].shape[0]

        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):

            dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]

            for t in range(n_time-1):
                sim_t, p = stats.spearmanr(dFC_mat_i[t,:,:].flatten(), dFC_mat_i[t+1,:,:].flatten())

                RESULTS[key_name].append(sim_t)
                RESULTS['dFC_method'].append(measure_i.measure_name)

    ALL_RESULTS['adjacent_time_points'] = deepcopy(RESULTS)

############ VISUALIZE ############

    cat_plot(data=RESULTS, x='dFC_method', y=key_name, 
        kind='violin',
        title=key_name + ' ' + filter,
        save_image=save_image, output_root=output_root+'indiv_prop/'
        )

################################# TRANSITION FREQUENCY #################################
'''
 - plot normalized transition frequency
'''
RESULTS = {}
key_name = 'trans_freq'
for filter in ['default_values']:
    RESULTS[key_name] = list()
    RESULTS['dFC_method'] = list()
    for s in ALL_RECORDS:

        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):

            if SUBJs_output[filter]['transition_stats'][i] == {}:
                continue
            RESULTS[key_name].append(SUBJs_output[filter]['transition_stats'][i]['trans_norm'])
            RESULTS['dFC_method'].append(measure_i.measure_name)

    ALL_RESULTS['transition_freq'] = deepcopy(RESULTS)

############ VISUALIZE ############

    cat_plot(data=RESULTS, x='dFC_method', y=key_name, 
        kind='violin',
        title=key_name + ' ' + filter,
        save_image=save_image, output_root=output_root+'indiv_prop/'
        )

################################# DWELL TIME #################################
'''
 - plot normalized (not downsampled) dwell times
 - the dwell times are not averaged wihtin each subj, they include all individual 
 dwells
'''
RESULTS = {}
key_name = 'dwell_time'
for filter in ['default_values']:
    RESULTS[key_name] = list()
    RESULTS['dFC_method'] = list()
    for s in ALL_RECORDS:

        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):
            if SUBJs_output[filter]['transition_stats'][i] == {}:
                continue
            dwell_time_lst = SUBJs_output[filter]['transition_stats'][i]['dwell_time_norm']
            # add each of the ind dwell times to the list along with the method's name
            for dwell_time in dwell_time_lst:
                RESULTS[key_name].append(dwell_time)
                RESULTS['dFC_method'].append(measure_i.measure_name)

    ALL_RESULTS['dwell_time'] = deepcopy(RESULTS)

############ VISUALIZE ############

    cat_plot(data=RESULTS, x='dFC_method', y=key_name, 
        kind='violin',
        title=key_name + ' ' + filter,
        save_image=save_image, output_root=output_root+'indiv_prop/'
        )

################################# TIME RECORD #################################

RESULTS = {}
for filter in ['default_values']:

    RESULTS['FCS_fit_time (s)'] = list()
    RESULTS['dFC_assess_time (s)'] = list()
    RESULTS['dFC_method'] = list()
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        for measure_id in SUBJs_output[filter]['time_record_dict']:

            if SUBJs_output[filter]['time_record_dict'][measure_id]['FCS_fit'] is None:
                RESULTS['FCS_fit_time (s)'].append(0)
            else:
                RESULTS['FCS_fit_time (s)'].append(SUBJs_output[filter]['time_record_dict'][measure_id]['FCS_fit'])
            RESULTS['dFC_assess_time (s)'].append(SUBJs_output[filter]['time_record_dict'][measure_id]['dFC_assess'])
            RESULTS['dFC_method'].append(SUBJs_output[filter]['measure_lst'][int(measure_id)].measure_name)

    ALL_RESULTS['time_record'] = deepcopy(RESULTS)

############ VISUALIZE ############

    cat_plot(data=RESULTS, x='dFC_method', y='dFC_assess_time (s)', 
        kind='bar',
        title='dFC assess time record of ' + filter,
        save_image=save_image, output_root=output_root+'time/'
        )

    cat_plot(data=RESULTS, x='dFC_method', y='FCS_fit_time (s)', 
        kind='bar',
        title='FCS fit time record of ' + filter,
        save_image=save_image, output_root=output_root+'time/'
        )

np.save(output_root+'ALL_RESULTS.npy', ALL_RESULTS)

if not save_image:
    plt.show()
#################################################################################
