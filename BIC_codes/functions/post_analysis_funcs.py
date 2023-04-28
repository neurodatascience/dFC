#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: mte
"""

import warnings
import numpy as np
import scipy.cluster.hierarchy as shc
import scipy.spatial.distance as ssd
import statsmodels.api as sm
from statsmodels.formula.api import ols

import matplotlib.pyplot as plt
import matplotlib as mpl
from nilearn.plotting import plot_markers
from matplotlib.colors import ListedColormap
import seaborn as sns
import pandas as pd
import os
from copy import deepcopy
import sys

sys.path.append('./../git_codes/BIC_codes/')
from functions.dFC_funcs import dFC_mat2vec, visualize_conn_mat_dict, mat_reorder

################################# Parameters ####################################

fig_dpi = 120
fig_bbox_inches = 'tight'
fig_pad = 0.1
show_title = False
save_fig_format = 'png' # pdf, png, 

################################# Plotting Functions ####################################

def title2file_name(title):
    '''
    change all spaces in the title to _
    the original string remains unchanged 
    '''
    return title.replace(" ", "_")

# test
def zip_name(name):
    # zip measure names
    if 'Clustering' in name:
        new_name = 'SWC' 
    if 'CAP' in name:
        new_name = 'CAP' 
    if 'ContinuousHMM' in name:
        new_name = 'CHMM' 
    if 'Windowless' in name:
        new_name = 'WL' 
    if 'DiscreteHMM' in name:
        new_name = 'DHMM' 
    if 'Time-Freq' in name:
        new_name = 'TF' 
    if 'SlidingWindow' in name:
        new_name = 'SW'
    return new_name

# test
# pear_corr problem
def unzip_name(name):
    # unzip measure names
    if 'SWC' in name:
        new_name = 'Clustering' 
    elif 'CAP' in name:
        new_name = 'CAP' 
    elif 'CHMM' in name:
        new_name = 'ContinuousHMM' 
    elif 'WL' in name:
        new_name = 'Windowless' 
    elif 'DHMM' in name:
        new_name = 'DiscreteHMM' 
    elif 'TF' in name:
        new_name = 'Time-Freq' 
    elif 'SW' in name:
        new_name = 'SlidingWindow'
    return new_name

def plot_sample_dFC(D, x,
    title='',
    cmap='seismic',
    normalize=False,
    disp_diag=True,
    save_image=False, output_root=None, 
    fix_lim=True, center_0=True, 
    node_networks=None, segmented=False 
    ):
    '''
    D is a dictionary of dFC samples. each
    key is the name of a dFC matrix (e.g. method
    used for assessing it), and D[key][x] contains the 
    the dFC matrix as a numpy ndarray 
    '''

    num_dFC = len(D)
    names_lst = [key for key in D]
    num_time = len(D[names_lst[0]][x])

    fig_width = 55*(num_time/10)
    fig_height = 55*(num_dFC/10)

    fig, axes = plt.subplots(num_dFC, num_time, figsize=(fig_width, fig_height), \
        facecolor='w', edgecolor='k')

    fig.subplots_adjust(
        bottom=0.1, \
        top=0.85, \
        left=0.1, \
        right=0.9,
        wspace=0.3, \
        hspace=0.8\
    )

    for i, dFC_mat_name in enumerate(D):
        visualize_conn_mat_dict(data=D[dFC_mat_name][x], 
            node_networks=node_networks, 
            title=dFC_mat_name, 
            cmap=cmap, center_0=center_0,
            normalize=normalize, fix_lim=fix_lim, 
            disp_diag=disp_diag,
            segmented=segmented,
            save_image=False, output_root=output_root,
            axes=axes[i, :], fig=fig, 
        )
        
    # set row names
    for i, dFC_mat_name in enumerate(D):
        axes[i, 0].set_ylabel(dFC_mat_name, fontdict={'fontsize': 18, 'fontweight': 'bold'}, rotation=90)
    
    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(output_root+title.replace(" ", "_")+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
        ) 
        plt.close()
    else:
        plt.show()

    
def pairwise_cat_plots(data=None, x=None, y=None, z=None,
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

            if not z is None:
                sns.stripplot(ax=axs[i, j], data=df, x=x, y=z, color='red', jitter=False, size=10)
            sns.violinplot(ax=axs[i, j], data=df, x=x, y=y)

            axs[i, j].set_title(key_i+'-'+key_j)
            axs_plotted.append(axs[i, j])

    # remove extra subplots
    for ax in axs.ravel():
        if not ax in axs_plotted:
            ax.set_axis_off()
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
    
    if show_title:
        plt.suptitle(title, fontsize=15, y=0.90)

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title2file_name(title)+'.'+save_fig_format, \
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format \
        ) 
        plt.close()
    else:
        plt.show()

def joint_dist_plot(data,
    title='',
    save_image=False, output_root=None
    ):
    '''
    data is a dictionary including list of dFC values
    of each dFC method
    '''
    df = pd.DataFrame(data)
    fig_width = 5*len(data)
    fig_height = 5*len(data)

    sns.set_context("paper", 
        font_scale=2.5, 
        rc={"lines.linewidth": 3.0}
    )
    
    sns.set_style('darkgrid')

    g = sns.PairGrid(df)

    g.map_diag(sns.histplot)
    g.map_offdiag(sns.histplot)

    g.fig.set_figwidth(fig_width)
    g.fig.set_figheight(fig_height)
    g.fig.subplots_adjust(top=0.95)
    if show_title:
        plt.suptitle(title, fontsize=50, y=0.98)

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title2file_name(title)+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
        ) 
        plt.close()
    else:
        plt.show()

def pairwise_scatter_plots(data, x, y,
    title='', hist=False,
    equal_axis_lim=False, show_x_equal_y=False,
    save_image=False, output_root=None
    ):
    '''
    data is a dictionary with different vars as keys 
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
    
    # equal x_lim and y_lim
    if equal_axis_lim or show_x_equal_y:
        min_lim = None
        max_lim = None
        for i, key_i in enumerate(data):
            for j, key_j in enumerate(data[key_i]):
                df = pd.DataFrame(data[key_i][key_j])
                m = np.minimum(df[x].min(), df[y].min())
                M = np.maximum(df[x].max(), df[y].max())
                if min_lim is None:
                    min_lim = m
                    max_lim = M
                else:
                    min_lim = np.minimum(m, min_lim)
                    max_lim = np.maximum(M, max_lim)
                
        lim_L = max_lim - min_lim
        min_lim = min_lim - lim_L*0.1
        max_lim = max_lim + lim_L*0.1

    axs_plotted = list()
    for i, key_i in enumerate(data):
        for j, key_j in enumerate(data[key_i]):
            df = pd.DataFrame(data[key_i][key_j])
            if hist:
                g = sns.histplot(ax=axs[i, j], data=df, x=x, y=y, bins=50)
            else:
                g = sns.scatterplot(ax=axs[i, j], data=df, x=x, y=y, s=50)
            axs[i, j].set_title(key_i+'-'+key_j)

            # equal x_lim and y_lim
            if equal_axis_lim:
                axs[i, j].set_xlim(min_lim, max_lim)
                axs[i, j].set_ylim(min_lim, max_lim)

            # y=x line
            if show_x_equal_y:
                X_plot = np.linspace(min_lim, max_lim, 100)
                Y_plot = X_plot
                axs[i, j].plot(X_plot, Y_plot, color='r')

            axs_plotted.append(axs[i, j])

    # remove extra subplots
    for ax in axs.ravel():
        if not ax in axs_plotted:
            ax.set_axis_off()
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
    
    if show_title:
        plt.suptitle(title, fontsize=15, y=0.90)

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title2file_name(title)+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format 
        ) 
        plt.close()
    else:
        plt.show()

def scatter_plot(data, x, y,
    labels=None, hue=None,
    title='', hist=False,
    equal_axis_lim=False, show_x_equal_y=False,
    save_image=False, output_root=None
    ):
    '''
    data is a dictionary with different vars as keys 
    '''
    df = pd.DataFrame(data)

    sns.set_context("paper", 
        font_scale=2.5, 
        rc={"lines.linewidth": 3.0}
    )

    fig_width = 20
    fig_height = 20 
    plt.figure(figsize=(fig_width, fig_height))
    sns.set_style('darkgrid')
    if hist:
        g = sns.histplot(data=df, x=x, y=y, hue=hue)
    else:
        g = sns.scatterplot(data=df, x=x, y=y, s=100, hue=hue)
    
    # equal x_lim and y_lim
    if equal_axis_lim:
        min_lim = np.minimum(df[x].min(), df[y].min())
        max_lim = np.maximum(df[x].max(), df[y].max())
        lim_L = max_lim - min_lim
        min_lim = min_lim - lim_L*0.1
        max_lim = max_lim + lim_L*0.1
        g.set_xlim(min_lim, max_lim)
        g.set_ylim(min_lim, max_lim)

    # y=x line
    if show_x_equal_y:
        min_lim = np.minimum(df[x].min(), df[y].min())
        max_lim = np.maximum(df[x].max(), df[y].max())
        lim_L = max_lim - min_lim
        min_lim = min_lim - lim_L*0.1
        max_lim = max_lim + lim_L*0.1
        X_plot = np.linspace(min_lim, max_lim, 100)
        Y_plot = X_plot
        plt.plot(X_plot, Y_plot, color='r')

    if (not labels is None) and (not hist):
        c = 0.015
        mid_x = (np.max(df[x]) + np.min(df[x]))/2
        mid_y = (np.max(df[y]) + np.min(df[y]))/2
        for i in range(len(df[x])):
            plt.text(
                x=df[x][i]-c*np.sign(df[x][i]-mid_x)-0.04, 
                y=df[y][i]-c*np.sign(df[y][i]-mid_y)-0.005, 
                s=df[labels][i], 
                fontdict=dict(color='black', size=20),
            )
    
    if show_title:
        plt.title(title, fontsize=15)

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title2file_name(title)+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
        ) 
        plt.close()
    else:
        plt.show()

def cat_plot(data, x, y, 
    kind='bar',
    scale_dist=False,
    log=False,
    title='',
    save_image=False, output_root=None
    ):
    '''
    data is a dictionary with different vars as keys 
    kind can be = box or violin or bar
    scale_dist is only for kind=='violin'
    '''

    sns.set_context("paper", 
        font_scale=1.0, 
        rc={"lines.linewidth": 1.0}
    )

    sns.set_style('darkgrid')

    df = pd.DataFrame(data)

    fig_width = 2*len(np.unique(data[x]))
    fig_height = 5 

    if kind=='violin' and scale_dist:
        g = sns.catplot(data=df, x=x, y=y, kind=kind,
            scale='width'
            # errorbar=("pi", 95)
        )
    elif kind=='bar':
        g = sns.catplot(data=df, x=x, y=y, kind=kind,
            # errorbar=("pi", 95)
        )
    else:
        g = sns.catplot(data=df, x=x, y=y, kind=kind,
            # errorbar=("pi", 95)
        )

    if log:
        plt.yscale('log')

    g.fig.set_figwidth(fig_width)
    g.fig.set_figheight(fig_height)
    if show_title:
        plt.title(title, fontsize=15)
    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title2file_name(title)+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
        ) 
        plt.close()
    else:
        plt.show()

def visualize_sim_mat(data, mat_key, title='', 
    name_lst_key=None, 
    cmap='viridis',
    annot=True, fmt=2, 
    show_diag=False, show_sig=False, no_color=False,
    save_image=False, output_root=None, axes=None, fig=None, 
    ):

    '''
    - name_lst_key is the key to list of names
    - data must be a dict of correlation/connectivity matrices
    - masks the nan values
    sample:
    Suptitle1
        corr_mat
            0.00 0.31 0.76 
            0.31 0.00 0.43 
            0.76 0.43 0.00 
        measure_lst
            ContinuousHMM
            Windowless
            Clustering_pear_corr
    Suptitle1
        corr_mat
            0.00 0.32 0.76 
            0.32 0.00 0.45 
            0.76 0.45 0.00 
        measure_lst
            ContinuousHMM
            Windowless
            Clustering_pear_corr
    '''

    sns.set_context("paper", 
        font_scale=1.0, 
        rc={"lines.linewidth": 1.0}
    )

    sns.set_style('white')

    if no_color:
        cmap = ListedColormap(['white'])
    
    if name_lst_key is None:
        fig_width = int(25*(len(data)/10))
    else:
        fig_width = int(60*(len(data)/10) + 1)
    fig_height = 5 

    fig_flag = True
    if axes is None or fig is None:
        fig_flag = False

    if not fig_flag:
        fig, axes = plt.subplots(1, len(data), figsize=(fig_width, fig_height), \
            facecolor='w', edgecolor='k', sharey=False)

    if not type(axes) is np.ndarray:
        axes = np.array([axes])

    if show_title:
        fig.suptitle(title, fontsize=20, y=0.98) #, fontsize=20, size=20

    axes = axes.ravel()

    # normalizing and scale
    sim_mats = list()
    for i, key in enumerate(data):
        sim_mats.append(data[key][mat_key])
    sim_mats = np.array(sim_mats)

    # plot
    for i, key in enumerate(data):

        C = sim_mats[i,:,:]

        name_lst = None
        if not name_lst_key is None:
            name_lst = data[key][name_lst_key]

        cbar_flag = False
        # if i==(len(data)-1):
        #     cbar_flag = True

        if annot:
            C_forlabels = C.copy()
            if not show_diag:
                np.fill_diagonal(C_forlabels, np.nan)
            df = pd.DataFrame(C_forlabels)
            if show_sig:
                annot_labels = df.applymap(lambda v: '' if np.isnan(v) else str(round(v, fmt))+''.join(['*' for t in [.05, .01, .001] if v<=t]))
            else:
                annot_labels = df.applymap(lambda v: '' if np.isnan(v) else str(round(v, fmt)))
        else:
            annot_labels = False

        # borderlines color
        if no_color:
            linecolor = 'black'
            annot_kws={'weight': 'bold'}
        else:
            linecolor = 'w'
            annot_kws={}

        im = sns.heatmap(C, 
            annot=annot_labels, annot_kws=annot_kws,
            fmt='', cmap=cmap, 
            xticklabels=name_lst, yticklabels=name_lst, 
            ax=axes[i], cbar=cbar_flag,
            square=True, linewidth=2, linecolor=linecolor
        )
        axes[i].set_title(key, fontdict= { 'fontsize': 17, 'fontweight':'bold'})
        im.set_xticklabels(im.get_xticklabels(), fontdict= { 'fontsize': 10, 'fontweight':'bold'}, rotation=90)
        im.set_yticklabels(im.get_yticklabels(), fontdict= { 'fontsize': 10, 'fontweight':'bold'}, rotation=0)

    if not fig_flag:
            
        fig.subplots_adjust(
            bottom=0.1, 
            top=0.85, 
            left=0.1, 
            right=0.9,
        )

        if not name_lst is None:
            fig.subplots_adjust(
                wspace=0.25 
            )

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title2file_name(title)+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
        ) 
        plt.close()
    else:
        plt.show()

def distance2Z(dist_mat, method='ward'):
    # convert the redundant n*n square matrix form into a condensed nC2 array
    distArray = ssd.squareform(dist_mat) 
    Z = shc.linkage(distArray, method=method)
    return Z

def dist_mat_dendo(Z, labels, 
    distances_CI=None,
    title='', \
    save_image=False, output_root=None, \
    ):
    '''
    if distances_CI is provided, confidence intervals (CI)
    of the distances will be shown. the order should be the same as 
    Z
    '''

    sns.set_context("paper", 
        font_scale=3.5, 
        rc={"lines.linewidth": 3.0}
    )

    sns.set_style('darkgrid')

    width = int(2.5*len(Z))
    fig = plt.figure(figsize=(width, 5))
    ax = fig.add_subplot(1, 1, 1)    
    with mpl.rc_context({'lines.linewidth': 3}):

        dend = shc.dendrogram(Z, distance_sort='ascending', no_plot=False, labels=labels)

        # show confidence interval of distances
        if not distances_CI is None:

            max_y_lim = None
            for i, d in zip(dend['icoord'], dend['dcoord']):
                
                # we have to match the distances in dcoord
                # with those in Z, because the orders are not
                # the same
                count = 0
                for idx, clstr in enumerate(Z):
                    if np.isclose(Z[idx][2], d[1]):
                        count += 1
                        Z_CI = distances_CI[idx]

                if count > 1 or count==0:
                    warnings.warn(
                        'Error in finding std of linkage.',
                        UserWarning
                    )

                x = 0.5 * sum(i[1:3])
                y = d[1]
                ci_line_y = np.linspace(y-Z_CI, y+Z_CI, 100)
                ci_line_x = x * np.ones(100)
                # cut start and the end for better
                # visualization
                ci_line_y = ci_line_y[5:-5]
                ci_line_x = ci_line_x[5:-5]

                plt.plot(ci_line_x, ci_line_y, 'black')
                plt.plot(x, y-Z_CI, 'b_', markersize=15, linewidth=15)
                plt.plot(x, y+Z_CI, 'b_', markersize=15, linewidth=15)
                plt.plot(x, y, 'ro', markersize=5)
                plt.annotate("%.2g" % y, (x, y), xytext=(15, 13),
                            fontsize = 11,
                            fontweight= 'bold',
                            textcoords='offset points',
                            va='top', ha='center')
                if max_y_lim is None:
                    max_y_lim = y+Z_CI
                else:
                    max_y_lim = max(y+Z_CI, max_y_lim)
            plt.ylim(0, max_y_lim*1.1)
                
    if show_title:
        plt.title(title, fontsize=15)
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='y', which='major', labelsize=15)    
    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title2file_name(title)+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
        ) 
        plt.close()

def plot_brain_act(act_vec, locs, axes,
    title='', save_image=False, output_root=''
    ):

    plot_markers(
        node_values=act_vec, node_coords=locs, 
        node_cmap='hot', 
        display_mode='z', 
        colorbar=False, axes=axes
    )

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title2file_name(title)+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
        ) 
        plt.close()

def visualize_state_TC(TC_lst, \
    TRs, \
    state_lst, \
    TC_name_lst, \
    title='', \
    save_image=None, output_root=None\
    ):

    color_lst = ['k', 'b', 'g', 'r']

    if 'on' in state_lst and 'off' in state_lst:
        ticks = range(2)
    else:
        ticks = range(1, len(state_lst)+1)

    plt.figure(figsize=(25, 5))
    for i, TC in enumerate(TC_lst):
        plt.plot(TRs, TC, color_lst[i], linewidth=2)
    plt.xlabel('TR')
    plt.yticks(ticks=ticks, labels=state_lst)
    plt.legend(TC_name_lst)
    if show_title:
        plt.title(title)
    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title2file_name(title)+'.'+save_fig_format, 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
        ) 
        plt.close()
    # else:
    #     plt.show()

    return

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

