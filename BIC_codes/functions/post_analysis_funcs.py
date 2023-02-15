#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: mte
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import os
from copy import deepcopy
import sys

sys.path.append('./../git_codes/BIC_codes/')
from functions.dFC_funcs import *

################################# Parameters ####################################

fig_dpi = 120
fig_bbox_inches = 'tight'
fig_pad = 0.1

################################# Plotting Functions ####################################

def pairwise_cat_plots(data, x, y, z=None,
    title='', 
    save_image=False, output_root=None
    ):
    '''
    data is a dictionary with different vars as keys 
    if z is specidied, it will be used as a out of distribution 
    sample, e.g. actual similarity when plotting randomized 
    distribution.
    '''

    sns.set_context("paper", 
        font_scale=2.5, 
        rc={"lines.linewidth": 3.0}
    )

    row_keys = [key for key in data]
    n_rows = len(row_keys)
    column_keys = [key for key in data[row_keys[-1]]]
    n_columns = len(column_keys)

    sns.set_style('darkgrid')

    fig_width = n_columns * 5
    fig_height = n_rows * 5
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(fig_width, fig_height), \
        facecolor='w', edgecolor='k', sharex=True, sharey=True)
    
    axs_plotted = list()
    for i, key_i in enumerate(data):
        for j, key_j in enumerate(data[key_i]):
            df = pd.DataFrame(data[key_i][key_j])

            # statistical significance
            if not z is None:
                stat, pvalue = scipy.stats.ttest_1samp(
                        df[y], 
                        df[z][0],
                        alternative='less'
                    )

            if not z is None:
                sns.stripplot(ax=axs[i, j], data=df, x=x, y=z, color='red', jitter=False, size=10)
            sns.violinplot(ax=axs[i, j], data=df, x=x, y=y)

            if not z is None:
                y_position = 1.0
                text_kwargs = dict(ha='center', va='center')
                axs[i, j].text(x=0, y=y_position, s=convert_pvalue_to_asterisks(pvalue), **text_kwargs)

            axs[i, j].set_title(key_i+'-'+key_j)
            axs_plotted.append(axs[i, j])

    # remove extra subplots
    for ax in axs.ravel():
        if not ax in axs_plotted:
            ax.set_axis_off()
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
    
    plt.suptitle(title, fontsize=15, y=0.90)

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title+'.png', \
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad \
        ) 
        plt.close()
    else:
        plt.show()

################################# Analytical Functions ####################################

from scipy import stats

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

