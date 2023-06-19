
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import deepcopy
import os

# sys.path.append('./BIC_codes/')
from functions.dFC_funcs import *
from functions.post_analysis_funcs import *

print('################################# POST ANALYSIS STARTED RUNNING ... #################################')

################################# LOAD RESULTS #################################

# local code
# assessment_results_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/'
# output_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/out/'

# server code
assessment_results_root = './'
output_root = './output/'

FOLDER_name = 'similarity_measured/'
num_randomization = 10
num_subj2include = None # None -> all

ALL_RECORDS = os.listdir(assessment_results_root+FOLDER_name)
ALL_RECORDS = [i for i in ALL_RECORDS if 'SUBJ_' in i]
ALL_RECORDS.sort()
for s in ALL_RECORDS[:1]:
    output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

print('*** %d subjects were found.' % (len(ALL_RECORDS)))

if not num_subj2include is None:
    ALL_RECORDS = ALL_RECORDS[:num_subj2include]
    print('*** %d subjects were included.' % (num_subj2include))

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

# the dictionary that collects all RESULTS
ALL_RESULTS = {} 

################################ common variables #################################

for filter in ['default_values']:

    # for SUBJs_output in SUBJs_output_lst:
    for s in ALL_RECORDS[:1]:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        node_networks = node_info2network(SUBJs_output[filter]['TS_info_lst'][0]['nodes_info'])

ALL_RESULTS['node_networks'] = node_networks
ALL_RESULTS['num_subj'] = len(ALL_RECORDS)


################################# dFC SAMPLES #################################
'''
only the first 10 time points are visualized
the rank_norm is performed globally (all time points of each subject together)
'''
for filter in ['default_values']:

    RESULTS = {}

    # for SUBJs_output in SUBJs_output_lst:
    for s in ALL_RECORDS[:1]:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        for measure_id in SUBJs_output[filter]['dFCM_samples']:
            TRs = SUBJs_output[filter]['common_TRs'][4:10]

            # rank the whole dFC
            dFC_mat = SUBJs_output[filter]['dFCM_samples'][measure_id]
            dFC_mat_ranked = rank_norm(dFC_mat, global_norm=True)

            samples = {}
            samples_ranked = {}
            for tr in TRs:
                samples['TR'+str(tr)] = dFC_mat[SUBJs_output[filter]['common_TRs'].index(tr), :, :]
                samples_ranked['TR'+str(tr)] = dFC_mat_ranked[SUBJs_output[filter]['common_TRs'].index(tr), :, :]
            
            measure_name = SUBJs_output[filter]['measure_lst'][int(measure_id)].measure_name

            if not measure_name in RESULTS:
                RESULTS[measure_name] = {}

            RESULTS[measure_name]['samples'] = samples
            RESULTS[measure_name]['samples_ranked'] = samples_ranked

ALL_RESULTS['dFC_sample'] = deepcopy(RESULTS)

################################# FCS visualization #################################

for filter in ['default_values']:

    # for SUBJs_output in SUBJs_output_lst:
    for s in ALL_RECORDS[:1]:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

ALL_RESULTS['measure_lst'] = SUBJs_output[filter]['measure_lst']

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

        # E(VAR(sim)) across subj
        var_lst = []
        for i, measure_i in enumerate(sim_distribution):
            for j, measure_j in enumerate(sim_distribution[measure_i]):
                var_lst.append(
                    np.var(np.array(sim_distribution[measure_i][measure_j]['sim']))
                )
        E_VAR = np.mean(np.array(var_lst))

        # E(E(sim)) and VAR(E(sim)) across subj
        avg_lst = []
        for i, measure_i in enumerate(sim_distribution):
            for j, measure_j in enumerate(sim_distribution[measure_i]):
                avg_lst.append(
                    np.mean(np.array(sim_distribution[measure_i][measure_j]['sim']))
                )
        E_E = np.mean(np.array(avg_lst))
        VAR_E = np.var(np.array(avg_lst))

        # change default_values to session_Rest1_LR
        if filter=='default_values':
            new_filter = 'session_Rest1_LR'
            RESULTS[new_filter] = {}
            RESULTS[new_filter]['avg_mat'] = all_subjs_avg
            RESULTS[new_filter]['var_mat'] = across_subj_var
            RESULTS[new_filter]['avg_div_var_mat'] = np.divide(all_subjs_avg, np.sqrt(across_subj_var), out=np.zeros_like(all_subjs_avg), where=np.sqrt(across_subj_var)!=0)
            RESULTS[new_filter]['sim_distribution'] = sim_distribution
            RESULTS[new_filter]['overall_stat'] = {'E_VAR': E_VAR, 'E_E': E_E, 'VAR_E': VAR_E}
            RESULTS[new_filter]['name_lst'] = measure_name_lst
        else:
            RESULTS[filter] = {}
            RESULTS[filter]['avg_mat'] = all_subjs_avg
            RESULTS[filter]['var_mat'] = across_subj_var
            RESULTS[filter]['avg_div_var_mat'] = np.divide(all_subjs_avg, np.sqrt(across_subj_var), out=np.zeros_like(all_subjs_avg), where=np.sqrt(across_subj_var)!=0)
            RESULTS[filter]['sim_distribution'] = sim_distribution
            RESULTS[filter]['overall_stat'] = {'E_VAR': E_VAR, 'E_E': E_E, 'VAR_E': VAR_E}
            RESULTS[filter]['name_lst'] = measure_name_lst

    ALL_RESULTS['dFC_similarity_overall'][metric] = deepcopy(RESULTS)

################# Hierarchical Clstr with Confidence Interval #################
'''
    - plot the most frequent hierclstr structures across subjects
'''
metric = 'spearman'

for filter in ['default_values']:

    Z_lst = list()
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        # SUBJs_output[filter]['all']['metric'] = (1, method, method)
        sim_mat = np.squeeze(SUBJs_output[filter]['all'][metric])

        dist_mat = corr2distance(sim_mat, metric=metric)
        Z = distance2Z(dist_mat, method='ward')
        Z_lst.append(Z)

ALL_RESULTS['Hierclstr_CI'] = deepcopy(Z_lst)

################# session ANOVA #################
'''
    - performing a two-way ANOVA test to investigate the effect of direction (LR/RL)
    and session on the observed similarity values
'''
sim_RESULTS = ALL_RESULTS['dFC_similarity_overall']['spearman']
# sim_RESULTS[filter]['sim_distribution'][measure_i][measure_j]['sim']

data = {}
for filter in sim_RESULTS:
    #session_Rest1_LR
    session = filter[filter.index('Rest'):filter.index('Rest')+5] #session_Rest1_LR
    direction = filter[filter.index('Rest')+6:filter.index('Rest')+8]

    sim_distribution_all = sim_RESULTS[filter]['sim_distribution']
    for i, measure_i in enumerate(sim_distribution_all):
        for j, measure_j in enumerate(sim_distribution_all[measure_i]):
            sim_distribution = sim_distribution_all[measure_i][measure_j]

            if not measure_i in data:
                data[measure_i] = {}
            if not measure_j in data[measure_i]:
                data[measure_i][measure_j] = {'sim': list(), 'session': list(), 'direction': list()}

            data[measure_i][measure_j]['sim'].extend(sim_distribution['sim'])
            data[measure_i][measure_j]['session'].extend([session for sim in sim_distribution['sim']])
            data[measure_i][measure_j]['direction'].extend([direction for sim in sim_distribution['sim']])
            
RESULTS = {}
for i, measure_i in enumerate(data):
    for j, measure_j in enumerate(data[measure_i]):
        
        if not measure_i in RESULTS:
            RESULTS[measure_i] = {}
        RESULTS[measure_i][measure_j] = two_way_anova(data[measure_i][measure_j])

ALL_RESULTS['session_ANOVA'] = deepcopy(RESULTS)

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
            if np.any(np.isnan(np.mean(SUBJs_output[filter]['feature_based'][feature2extract], axis=0))):
                continue
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

############ Spatial vs. Temporal Scatter plot ############
    
for filter in ['default_values']:

    all_subjs_spatial_sim_mat = list()
    all_subjs_temporal_sim_mat = list()
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        # SUBJs_output[filter]['feature_based'][feature2extract] = (sample, method, method)
        if (
            np.any(np.isnan(np.mean(SUBJs_output[filter]['feature_based']['spatial'], axis=0))) 
            or np.any(np.isnan(np.mean(SUBJs_output[filter]['feature_based']['temporal'], axis=0)))
        ):
            continue
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

################################# TSNE #################################
'''
    - visualize subjects dFC obtained by different methods in a 2D space
        considering their distances
'''
RESULTS = {}
for filter in ['default_values']:
    features_dict = {}
    for s in ALL_RECORDS:
        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()
        subj_name = SUBJs_output[filter]['TS_info_lst'][0]['subj_id_lst'][0]

        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):
            measure_name = measure_i.measure_name

            dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]
            # rank normalization
            dFC_mat_i = rank_norm(dFC_mat_i) 

            if not measure_name in features_dict:
                features_dict[measure_name] = {'dFC':list(), 'sample_name':list()}
            features_dict[measure_name]['dFC'].append(dFC_mat2vec(dFC_mat_i).flatten())
            features_dict[measure_name]['sample_name'].append(subj_name+'_'+measure_name)

sample_lst = list()
features_all = list()
for measure_name in features_dict:
    features_all.append(features_dict[measure_name]['dFC'])
    sample_lst.extend(features_dict[measure_name]['sample_name'])
features_all = np.array(features_all)

# observation matrix
X = np.reshape(features_all, (-1, features_all.shape[-1]))

RESULTS['sample_lst'] = sample_lst
# overall corr
RESULTS['corr'] = np.corrcoef(X)

RESULTS['X_red'] = {}
RESULTS['X_embedded'] = {}
RESULTS['X_red_corr'] = {}
for n_components in [100, 500, 1000, 1500, 2000]:
    pca = PCA(n_components=n_components)
    RESULTS['X_red'][n_components] = pca.fit_transform(X)

    # corr of subjects after PCA
    RESULTS['X_red_corr'][n_components] = np.corrcoef(RESULTS['X_red'][n_components])

    # embessing X into 2D space using TSNE
    RESULTS['X_embedded'][n_components] = TSNE(
                                            n_components=2, learning_rate='auto',
                                            init='random', perplexity=30
                                        ).fit_transform(RESULTS['X_red'][n_components])
    
ALL_RESULTS['TSNE']= deepcopy(RESULTS)

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
            dFC_mat_lst.append(dFC_mat_i)

        dFC_mat_lst = np.array(dFC_mat_lst) # (method, time, ROI, ROI)
        var_over_method.append(np.mean(np.var(dFC_mat_lst, axis=0), axis=0))
        var_over_time.append(np.mean(np.var(dFC_mat_lst, axis=1), axis=0))

    var_over_time = np.array(var_over_time) # (subj, ROI, ROI)
    var_over_method = np.array(var_over_method) # (subj, ROI, ROI)

    # collect var over method and time across all func conns of all subjects
    scatter_data = {'var_method':list(), 'var_time':list()}
    scatter_data['var_method'] = dFC_mat2vec(var_over_method).flatten() # (subj*(ROI)*(ROI-1)/2,)
    scatter_data['var_time'] = dFC_mat2vec(var_over_time).flatten() # (subj*(ROI)*(ROI-1)/2,)

    var_over_time = np.mean(var_over_time, axis=0) # (ROI, ROI)
    var_over_method = np.mean(var_over_method, axis=0) # (ROI, ROI)

    RESULTS = {}
    RESULTS['var_over_time'] = var_over_time
    RESULTS['var_over_method'] = var_over_method

    RATIO = {
        'var across method / time - 1': np.divide(
                                            var_over_method, var_over_time, 
                                            out=np.zeros_like(var_over_method), 
                                            where=var_over_time!=0
                                        ) - 1 
    }

    ALL_RESULTS['var_across_func_conns'] = deepcopy(RESULTS)
    ALL_RESULTS['var_across_func_conns_ratio'] = deepcopy(RATIO)
    ALL_RESULTS['var_method_vs_time_across_func_conns_scatter'] = deepcopy(scatter_data)

    for key in RESULTS:
        RESULTS[key] = rank_norm(RESULTS[key])

    for key in RESULTS:
        RESULTS[key] = cat_data(RESULTS[key], N=10)
        RESULTS[key] = np.where(RESULTS[key] == np.max(RESULTS[key]), 1, 0)

    ALL_RESULTS['high_var_func_conns'] = deepcopy(RESULTS)

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

    RESULTS['var_method_divide_temp'] = {'sim_mat': divide_temp, 'name_lst': measure_name_lst}
    RESULTS['var_method_divide_1_lag'] = {'sim_mat': divide_1_lag, 'name_lst': measure_name_lst}

ALL_RESULTS['var_comparison'] = deepcopy(RESULTS)
ALL_RESULTS['var_method_vs_time_method_pairs_across_func_conns'] = deepcopy(scatter_data_across_func_conn)
ALL_RESULTS['var_method_vs_time_method_pairs'] = deepcopy(scatter_data)

################################# Var method vs. Time Clstrwise #################################
'''
    - compute var over method and time within each group of methods
'''
ALL_RESULTS['var_method_vs_time_clstrwise'] = {}

clstr1 = ['CAP', 'Windowless']
clstr2 = ['Clustering', 'ContinuousHMM', 'DiscreteHMM']
clstr3 = ['SlidingWindow', 'Time-Freq']
clstrs_dict = {'clstr1':clstr1, 'clstr2':clstr2, 'clstr3':clstr3}

for filter in ['default_values']:
    var_over_time = {'clstr1': list(), 'clstr2': list(), 'clstr3': list()}
    var_over_method = {'clstr1': list(), 'clstr2': list(), 'clstr3': list()}
    for s in ALL_RECORDS:

        SUBJs_output = np.load(assessment_results_root+FOLDER_name+s, allow_pickle='True').item()

        dFC_mat_lst = {'clstr1': list(), 'clstr2': list(), 'clstr3': list()}
        for i, measure_i in enumerate(SUBJs_output[filter]['measure_lst']):

            dFC_mat_i = SUBJs_output[filter]['dFCM_samples'][str(i)]

            # rank normalization
            dFC_mat_i = rank_norm(dFC_mat_i)

            # dFC mat
            if measure_i.measure_name in clstr1:
                dFC_mat_lst['clstr1'].append(dFC_mat_i)
            elif measure_i.measure_name in clstr2:
                dFC_mat_lst['clstr2'].append(dFC_mat_i)
            elif measure_i.measure_name in clstr3:
                dFC_mat_lst['clstr3'].append(dFC_mat_i)

        # Calc var_over_method and var_over_time in each subj
        for clstr in dFC_mat_lst:
            dFC_mat_lst[clstr] = np.array(dFC_mat_lst[clstr]) # (method, time, ROI, ROI)
            var_over_method[clstr].append(np.mean(np.var(dFC_mat_lst[clstr], axis=0), axis=0))
            var_over_time[clstr].append(np.mean(np.var(dFC_mat_lst[clstr], axis=1), axis=0))

    for clstr in dFC_mat_lst:
        var_over_time[clstr] = np.array(var_over_time[clstr]) # (subj, ROI, ROI)
        var_over_method[clstr] = np.array(var_over_method[clstr]) # (subj, ROI, ROI)

        var_over_time[clstr] = dFC_mat2vec(var_over_time[clstr]).flatten() # (subj*(ROI)*(ROI-1)/2,)
        var_over_method[clstr] = dFC_mat2vec(var_over_method[clstr]).flatten() # (subj*(ROI)*(ROI-1)/2,)

    # collect var over method and time across all func conns of all subjects
    scatter_data = {}
    for clstr in var_over_method:
        scatter_data[clstr] = {'var_method':list(), 'var_time':list()}
        scatter_data[clstr]['var_method'] = var_over_method[clstr]
        scatter_data[clstr]['var_time'] = var_over_time[clstr]

    ALL_RESULTS['var_method_vs_time_clstrwise']['scatter_data'] = deepcopy(scatter_data)
    ALL_RESULTS['var_method_vs_time_clstrwise']['clstrs_dict'] = deepcopy(clstrs_dict)

################################# Randomization Tests #################################

metric = 'spearman'

ALL_RESULTS['randomization'] = {}

########### Similarity with static FC ###########
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

########### Shuffled ###########

'''
find the similarity between the dFC obtained by each method 
but with randomized temporal(time)/spatial(region)/all(both) order
 - temporal: find the similarity between the dFC obtained by each method 
            but with randomized temporal order. The null keeps the spatial 
            order but not the temporal orders
 - spatial: find the similarity between the dFC obtained by each method 
            but with randomized spatial order. The null keeps the temporal 
            order but not the spatial orders
 - temporal: find the similarity between the dFC obtained by each method 
            but with randomized both temporal AND spatial order. The null keeps 
            neither the spatial order nor the temporal order
'''
metric = 'spearman'
modes_lst = [
    'temporal',
    'spatial',
    'all'
]
ALL_RESULTS['randomization']['shuffled'] = {'similarity_dict': {}, 'modes_lst': modes_lst}

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

        for mode in modes_lst:
            key = mode+'_shuffle_sim'
            output = randomized_dFC_sim(dFC_dict, N=num_randomization, mode=mode)

            for measure_i_name in output:
                for measure_j_name in output[measure_i_name]:

                    if not measure_i_name in RESULTS:
                        RESULTS[measure_i_name] = {}
                    if not measure_j_name in RESULTS[measure_i_name]:
                        RESULTS[measure_i_name][measure_j_name] = {}
                    if not key in RESULTS[measure_i_name][measure_j_name]:
                        RESULTS[measure_i_name][measure_j_name][key] = list()
                
                    RESULTS[measure_i_name][measure_j_name][key].extend(output[measure_i_name][measure_j_name]['sim'])

    for measure_i_name in output:
        for measure_j_name in output[measure_i_name]:
            sim = all_subjs_avg[measure_name_lst.index(unzip_name(measure_i_name)), measure_name_lst.index(unzip_name(measure_j_name))]
            # change diagonal from 0 to 1
            if measure_i_name==measure_j_name:
                sim = 1
            RESULTS[measure_i_name][measure_j_name]['actual_sim'] = [sim for item in RESULTS[measure_i_name][measure_j_name][key]]

ALL_RESULTS['randomization']['shuffled']['similarity_dict'] = RESULTS

########### Random state time course ###########

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

        output = dFC_rand_sim(FCS_dict, n_time, N=num_randomization)

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

################################# inter-state spatial similarity #################################

key_name = 'inter-state_spatial_sim'
RESULTS = {'sim':list(), 'dFC_method':list()}
for measure in ALL_RESULTS['measure_lst']:

    if len(measure.FCS)==0:
        continue
    FCSs = np.array(measure.FCS)
    
    for i, FCS_i in enumerate(FCSs):
        for j, FCS_j in enumerate(FCSs):

            if j>=i:
                continue

            vec_i = dFC_mat2vec(FCS_i)
            vec_j = dFC_mat2vec(FCS_j)

            RESULTS['sim'].append(np.corrcoef(vec_i, vec_j)[0, 1])
            RESULTS['dFC_method'].append(measure.measure_name)

ALL_RESULTS[key_name] = deepcopy(RESULTS)

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


################################# SAVE ALL RESULTS #################################

np.save(output_root+'ALL_RESULTS.npy', ALL_RESULTS)

#################################################################################
