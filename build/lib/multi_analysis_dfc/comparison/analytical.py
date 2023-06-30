# -*- coding: utf-8 -*-
"""
Analytical functions for the comparison framework.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import numpy as np
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
import pandas as pd
from copy import deepcopy

from ..dfc_utils import zip_name, mat_reorder, dFC_mat2vec 
 
################################# Analytical Functions ####################################

############## STAT Functions ##############

def make_sim_distribution(sim_mats_lst, name_lst, zip_names=True):
    '''
    each sim_mat in the sim_mat_lst corresponds
    to a subj. the name_lst must correspond to the
    columns and rows of sim_mats
    '''
    output = {}
    for sim_mat in sim_mats_lst:
        
        for i, name_i in enumerate(name_lst):
            for j, name_j in enumerate(name_lst):

                if j>=i:
                    continue

                if zip_names:
                    name_i_used = zip_name(name_i)
                    name_j_used = zip_name(name_j)
                else:
                    name_i_used = name_i
                    name_j_used = name_j

                if not name_i_used in output:
                    output[name_i_used] = {}
                if not name_j_used in output[name_i_used]:
                    output[name_i_used][name_j_used] = {'sim':list(), '':list()}
            
                output[name_i_used][name_j_used]['sim'].append(sim_mat[i, j])
                output[name_i_used][name_j_used][''].append('sim')
    return output

def two_way_anova(data):
    '''
    perform two-way anova
    target: sim
    factor1: session
    factor2: direction
    '''
    df = pd.DataFrame(data)

    # Performing two-way ANOVA
    model = ols('sim ~ C(session) + C(direction) +\
    C(session):C(direction)',
                data=df).fit()
    
    return sm.stats.anova_lm(model, type=2)
    
def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"

############## Randomization Functions ##############

def randomize_time(dFC_dict, N):
    '''
    '''
    output = {}
    for n in range(N):

        for i, measure_i_name in enumerate(dFC_dict):
            
            dFC_mat_i = dFC_dict[measure_i_name]

            # randomize the temporal order
            n_time = dFC_mat_i.shape[0]
            idx = np.random.choice(n_time, n_time, replace=False)
            dFC_mat_i = dFC_mat_i[idx, :, :]

            dFC_mat_i_vec = dFC_mat2vec(dFC_mat_i)

            for j, measure_j_name in enumerate(dFC_dict):
            
                if j>i:
                    continue
                if not measure_i_name in output:
                    output[measure_i_name] = {}
                if not measure_j_name in output[measure_i_name]:
                    output[measure_i_name][measure_j_name] = {'sim':list(), '':list()}

                dFC_mat_j = dFC_dict[measure_j_name]

                # randomize the temporal order
                n_time = dFC_mat_j.shape[0]
                idx = np.random.choice(n_time, n_time, replace=False)
                dFC_mat_j = dFC_mat_j[idx, :, :]

                dFC_mat_j_vec = dFC_mat2vec(dFC_mat_j)

                sim, p = stats.spearmanr(dFC_mat_i_vec.flatten(), dFC_mat_j_vec.flatten())
                output[measure_i_name][measure_j_name]['sim'].append(sim)
                output[measure_i_name][measure_j_name][''].append('sim')

    return output


def suffle_dFC(dFC_mat, mode):
    '''
    dFC_mat = ndarray(time, region, region)
    mode can be 'temporal', 'spatial',
    or 'all'
    '''
    new_dFC_mat = deepcopy(dFC_mat)
    if mode=='temporal':
        n_time = new_dFC_mat.shape[0]
        new_order = np.random.choice(n_time, n_time, replace=False)
        new_dFC_mat = new_dFC_mat[new_order, :, :]
    elif mode=='spatial':
        n_region = new_dFC_mat.shape[1]
        new_order = np.random.choice(n_region, n_region, replace=False)
        for k, mat in enumerate(new_dFC_mat):
            new_dFC_mat[k, :, :] = mat_reorder(new_dFC_mat[k, :, :], new_order)
    elif mode=='all':
        #spatial
        n_region = new_dFC_mat.shape[1]
        new_order_regions = np.random.choice(n_region, n_region, replace=False)
        for k, mat in enumerate(new_dFC_mat):
            new_dFC_mat[k, :, :] = mat_reorder(new_dFC_mat[k, :, :], new_order_regions)
        #temporal
        n_time = new_dFC_mat.shape[0]
        new_order_time = np.random.choice(n_time, n_time, replace=False)
        new_dFC_mat = new_dFC_mat[new_order_time, :, :]

    return new_dFC_mat


def randomized_dFC_sim(dFC_dict, N, mode):
    '''
    mode can be 'temporal', 'spatial',
    or 'all'
    'spatial': this will result in different methods having
    different spatial/region orders but still the
    same temporal order
    'temporal': this will result in different methods having
    different temporal orders but still the
    same spatial order
    'all': this will result in different methods having
    different temporal orders AND different spatial order
    '''
    output = {}
    for n in range(N):

        for i, measure_i_name in enumerate(dFC_dict):
            
            dFC_mat_i = dFC_dict[measure_i_name]

            # randomize the spatial (regions) order
            dFC_mat_i = suffle_dFC(dFC_mat_i, mode=mode)

            dFC_mat_i_vec = dFC_mat2vec(dFC_mat_i)

            for j, measure_j_name in enumerate(dFC_dict):
            
                if j>i:
                    continue
                if not measure_i_name in output:
                    output[measure_i_name] = {}
                if not measure_j_name in output[measure_i_name]:
                    output[measure_i_name][measure_j_name] = {'sim':list(), '':list()}

                dFC_mat_j = dFC_dict[measure_j_name]

                # randomize the temporal order
                dFC_mat_j = suffle_dFC(dFC_mat_j, mode=mode)

                dFC_mat_j_vec = dFC_mat2vec(dFC_mat_j)

                sim, p = stats.spearmanr(dFC_mat_i_vec.flatten(), dFC_mat_j_vec.flatten())
                output[measure_i_name][measure_j_name]['sim'].append(sim)
                output[measure_i_name][measure_j_name][''].append('sim')

    return output


def dFC_rand_generator(FCS, n_time):
    '''
    generate a dFC mat of length n_time 
    using spatial FC patterns in FCS = (num_pattern, ROI, ROI)
    '''
    dFC_rand = None
    idx = np.random.choice(FCS.shape[0], n_time, replace=True)
    dFC_rand = FCS[idx, :, :]
    return dFC_rand

def dFC_rand_sim(FCS_dict, n_time, N):
    '''
    for random state TC similarity assessment
    '''
    output = {}
    for n in range(N):

        for i, measure_i_name in enumerate(FCS_dict):
                
            dFC_rand = dFC_rand_generator(FCS_dict[measure_i_name], n_time=n_time)
            dFC_mat_i = dFC_rand
            dFC_mat_i_vec = dFC_mat2vec(dFC_mat_i)

            for j, measure_j_name in enumerate(FCS_dict):
            
                if j>i:
                    continue
                if not measure_i_name in output:
                    output[measure_i_name] = {}
                if not measure_j_name in output[measure_i_name]:
                    output[measure_i_name][measure_j_name] = {'sim':list(), '':list()}

                dFC_rand = dFC_rand_generator(FCS_dict[measure_j_name], n_time=n_time)
                dFC_mat_j = dFC_rand
                dFC_mat_j_vec = dFC_mat2vec(dFC_mat_j)

                sim, p = stats.spearmanr(dFC_mat_i_vec.flatten(), dFC_mat_j_vec.flatten())
                output[measure_i_name][measure_j_name]['sim'].append(sim)
                output[measure_i_name][measure_j_name][''].append('sim')
    return output

############## Hierarchical Clustering ##############

def correct_order(s):
    list = s.split('-')
    list = [int(item) for item in list]
    list.sort()
    return '-'.join(str(x) for x in list)

def open_trees(Z, num_leaf):
    '''
    replace trees in Z by their leaves
    '''
    Z_copy = deepcopy(Z)
    Z_new = []
    for tree in Z_copy:
        Z_new.append([tree[0], tree[1], tree[2], tree[3]])
    encode_dict = {}
    counter = num_leaf
    for tree in Z_new:
        if tree[0]>=num_leaf:
            tree[0] = encode_dict[tree[0]]
        else:
            tree[0] = str(int(tree[0]))
        if tree[1]>=num_leaf:
            tree[1] = encode_dict[tree[1]]
        else:
            tree[1] = str(int(tree[1]))
        encode_dict[counter] = tree[0]+'-'+tree[1]
        encode_dict[counter] = correct_order(encode_dict[counter])
        counter += 1
    return Z_new

def is_trees_equal(trees_1, trees_2):
    '''
    trees_2 is the reference
    '''
    for tree in trees_1:
        if (not [tree[0], tree[1]] in trees_2) \
            and (not [tree[1], tree[0]] in trees_2):
            return False
    return True

def is_in_Z_clstrs(trees, Z_clstrs, trees_key):
    for key in Z_clstrs:
        if is_trees_equal(trees, Z_clstrs[key][trees_key]):
            return key
    return None

def cluster_Z(Z_lst, num_leaf):
    '''
    Z_lst is the list of linkages of samples
    num_leaf is the number of objects in clustering
    '''
    Z_clstrs = {}
    counter = 0
    for Z in Z_lst:
        # replace trees in Z by their leaves
        Z_open = open_trees(Z, num_leaf)
        trees = [[tree[0], tree[1]] for tree in Z_open]
        distances = [tree[2] for tree in Z]
        clstr_idx = is_in_Z_clstrs(trees, Z_clstrs, trees_key='trees')
        
        if clstr_idx is None:
            Z_clstrs[counter] = {
                'Z': Z, 
                'trees': trees,
                'freq': 1, 
                'distance_lst': [distances]
            }
            counter += 1
        else:
            Z_clstrs[clstr_idx]['freq'] += 1
            Z_clstrs[clstr_idx]['distance_lst'].append(distances)
    return Z_clstrs