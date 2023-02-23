
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

# output_root = './../../../../RESULTs/methods_implementation/server/methods_implementation/out/'
output_root = './output/'
save_image = True

# the dictionary that collects all RESULTS
ALL_RESULTS = np.load(output_root+'ALL_RESULTS.npy', allow_pickle='True').item()

################################ common variables #################################

node_networks = ALL_RESULTS['node_networks']

################################# dFC SAMPLES #################################

RESULTS = ALL_RESULTS['dFC_sample']

for measure_name in RESULTS:
    
    visualize_conn_mat_dict(RESULTS[measure_name]['samples'], node_networks=node_networks, 
        title=measure_name+'_'+filter, 
        normalize=False, fix_lim=False, 
        disp_diag=False,
        save_image=save_image, output_root=output_root+'dFC_sample/'
        )
    visualize_conn_mat_dict(RESULTS[measure_name]['samples_ranked'], node_networks=node_networks, 
        title=measure_name+'_ranked_'+filter, 
        normalize=False, fix_lim=False, 
        disp_diag=False, cmap='plasma', center_0=False,
        save_image=save_image, output_root=output_root+'dFC_sample/'
        )

################################# FCS visualization #################################

for measure in ALL_RESULTS['measure_lst']:

    measure.visualize_FCS(
            normalize=True, fix_lim=False, 
            save_image=save_image, output_root=output_root+'FCS/'
            )

################################# dFC values distributions #################################

RESULTS = ALL_RESULTS['dFC_values_dist']

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

for metric in ALL_RESULTS['dFC_similarity_overall'] :

    RESULTS = ALL_RESULTS['dFC_similarity_overall'][metric]

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

for feature2extract in ALL_RESULTS['dFC_similarity_feature_based']:

    RESULTS = ALL_RESULTS['dFC_similarity_feature_based'][feature2extract]

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
    
scatter_data = ALL_RESULTS['spatial_vs_temporal_similarity']

############ visualization ############
scatter_plot(
    data=scatter_data, x='temporal', y='spatial', 
    labels='labels', title='spatial similarity vs temporal similarity',
    save_image=save_image, output_root=output_root+'variation/'
)
    
################# graph-based #################
'''
    - spatial
    - temporal-avg
    - ECM, shortest_path, degree, clustering_coef
'''

###### spatial #####

for graph_property in ALL_RESULTS['dFC_similarity_graph']['spatial']:

    RESULTS = ALL_RESULTS['dFC_similarity_graph']['spatial'][graph_property]

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

for subj_lvl_feature in ALL_RESULTS['subj_clustring']:

    RESULTS = ALL_RESULTS['subj_clustring'][subj_lvl_feature] 

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

RESULTS = ALL_RESULTS['dFC_var']

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

RESULTS = ALL_RESULTS['dFC_avg'] 

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

RESULTS = ALL_RESULTS['across_func_conns'] = deepcopy(RESULTS)

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

RESULTS = ALL_RESULTS['var_across_func_conns'] 
scatter_data = ALL_RESULTS['var_method_vs_time_across_func_conns_scatter'] 

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

RESULTS = ALL_RESULTS['high_var_func_conns'] 

visualize_conn_mat_dict(RESULTS, node_networks=node_networks, 
    title='high variation regions '+filter, fix_lim=False, 
    disp_diag=True, cmap='plasma', center_0=False,
    save_image=save_image, output_root=output_root+'variation/'
)

################################# Variation Value Comparison #################################
'''
    - compare variation over methods with variation over time.
'''

RESULTS = ALL_RESULTS['var_comparison']
scatter_data_across_func_conn = ALL_RESULTS['var_method_vs_time_method_pairs_across_func_conns'] 
scatter_data = ALL_RESULTS['var_method_vs_time_method_pairs']

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

################################# Randomization Tests #################################

metric = 'spearman'

'''
find the similarity between the dFC obtained by each method 
and a dFC which is a constant sequence of mean of all methods dFC
'''

########### Similarity with static FC ###########

RESULTS = ALL_RESULTS['randomization']['sim_with_static_FC']

############ VISUALIZE ############
cat_plot(data=RESULTS, x='dFC_method', y='sim', 
        kind='violin',
        title='similarity with constant static FC',
        save_image=save_image, output_root=output_root+'randomization/'
        )

########### Shuffled time ###########

'''
find the similarity between the dFC obtained by each method 
but with randomized temporal order
'''

RESULTS = ALL_RESULTS['randomization']['shuffled_time'] 

############ VISUALIZE ############
pairwise_cat_plots(RESULTS, x='', y='sim', z='actual_sim',
    title='randomized time similarity',
    save_image=save_image, output_root=output_root+'randomization/'
    )

########### Random state time course ###########

'''
find the similarity between dFC matrices created using
a random state time course but real FC patterns obtained 
by each method; for SW and TF all FC patterns of each 
subject are used
'''

RESULTS = ALL_RESULTS['randomization']['random_state_TC'] 

############ VISUALIZE ############
pairwise_cat_plots(RESULTS, x='', y='sim', z='actual_sim',
    title='random state time course dFC',
    save_image=save_image, output_root=output_root+'randomization/'
    )

################################# SIMILARITY OF ADJACENT TIME POINTS #################################

key_name = 'similarity of adjacent time points'
RESULTS = ALL_RESULTS['adjacent_time_points']

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

key_name = 'trans_freq'
RESULTS = ALL_RESULTS['transition_freq'] 

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

key_name = 'dwell_time'
RESULTS = ALL_RESULTS['dwell_time'] 

############ VISUALIZE ############

cat_plot(data=RESULTS, x='dFC_method', y=key_name, 
    kind='violin',
    title=key_name + ' ' + filter,
    save_image=save_image, output_root=output_root+'indiv_prop/'
    )

################################# TIME RECORD #################################

RESULTS = ALL_RESULTS['time_record'] 

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

if not save_image:
    plt.show()
#################################################################################
