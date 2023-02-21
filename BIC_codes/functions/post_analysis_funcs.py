#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 8 2023

@author: mte
"""

import numpy as np
import scipy.cluster.hierarchy as shc
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import matplotlib as mpl
from nilearn.plotting import plot_markers
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

            # # statistical significance
            # if not z is None:
            #     res = scipy.stats.ttest_1samp(
            #             df[y], 
            #             df[z][0],
            #             alternative='less'
            #         )

            #     ci = res.confidence_interval(confidence_level=0.95)

            if not z is None:
                sns.stripplot(ax=axs[i, j], data=df, x=x, y=z, color='red', jitter=False, size=10)
            sns.violinplot(ax=axs[i, j], data=df, x=x, y=y)

            # if not z is None:
            #     y_position = 1.0
            #     text_kwargs = dict(ha='center', va='center')
            #     axs[i, j].text(x=0, y=y_position, s=convert_pvalue_to_asterisks(pvalue), **text_kwargs)

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

def joint_dist_plot(data,
    title='',
    save_image=False, output_root=None
    ):
    '''
    data is a dictionary including list of dFC values
    of each dFC method
    '''
    title = 'distributions'
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
    plt.suptitle(title, fontsize=50, y=0.98)

    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title+'.png', 
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad
        ) 
        plt.close()
    else:
        plt.show()

def pairwise_scatter_plots(data, x, y,
    title='', hist=False,
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
    
    axs_plotted = list()
    for i, key_i in enumerate(data):
        for j, key_j in enumerate(data[key_i]):
            df = pd.DataFrame(data[key_i][key_j])
            if hist:
                g = sns.histplot(ax=axs[i, j], data=df, x=x, y=y, bins=50)
            else:
                g = sns.scatterplot(ax=axs[i, j], data=df, x=x, y=y, s=50)
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

def scatter_plot(data, x, y,
    labels=None, hue=None,
    title='', hist=False,
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
    plt.title(title, fontsize=15)

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

def cat_plot(data, x, y, 
    kind='bar',
    title='',
    save_image=False, output_root=None
    ):
    '''
    data is a dictionary with different vars as keys 
    kind can be = box or violin or bar
    '''

    sns.set_context("paper", 
        font_scale=1.0, 
        rc={"lines.linewidth": 1.0}
    )

    df = pd.DataFrame(data)

    fig_width = 2*len(np.unique(data[x]))
    fig_height = 5 
    g = sns.catplot(data=df, x=x, y=y, kind=kind,
        scale='width'
        # errorbar=("pi", 95)
    )
    g.fig.set_figwidth(fig_width)
    g.fig.set_figheight(fig_height)
    plt.title(title, fontsize=15)
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

def visualize_sim_mat(data, mat_key, title='', 
    name_lst_key=None, 
    cmap='viridis',
    annot=True, fmt=".2f",
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
            np.fill_diagonal(C_forlabels, np.nan)
            df = pd.DataFrame(C_forlabels)
            annot_labels = df.applymap(lambda v: '' if np.isnan(v) else str(round(v,2)))
        else:
            annot_labels = False

        im = sns.heatmap(C, 
            annot=annot_labels, fmt='', cmap=cmap, 
            xticklabels=name_lst, yticklabels=name_lst, 
            ax=axes[i], cbar=cbar_flag,
            square=True, linewidth=2, linecolor='w'
        )
        axes[i].set_title(key, fontdict= { 'fontsize': 17, 'fontweight':'bold'})
        im.set_xticklabels(im.get_xticklabels(), fontdict= { 'fontsize': 10, 'fontweight':'bold'})
        im.set_yticklabels(im.get_yticklabels(), fontdict= { 'fontsize': 10, 'fontweight':'bold'})

    if not fig_flag:
        fig.subplots_adjust(
            bottom=0.1, \
            top=0.85, \
            left=0.1, \
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
        plt.savefig(output_root+title+'.png', \
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad \
        ) 
        plt.close()
    else:
        plt.show()

def visualize_conn_mat(C, axis=None, title='', \
    cmap='seismic',\
    V_MIN=None, V_MAX=None, \
    node_networks=None \
    ):
    '''
    C is (regions, regions)
    '''

    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=(5, 5))

    if node_networks is None:
        axis.set_axis_off()

    if V_MAX is None:
        V_MAX = np.max(np.abs(C))
    if V_MIN is None:
        V_MIN = -1*V_MAX

    im = axis.imshow(C, interpolation='nearest', aspect='equal', cmap=cmap,    # 'viridis' or 'jet'
        vmin=V_MIN, vmax=V_MAX)
    
    # cluster node networks
    if not node_networks is None:
        
        # finding unique network names wrt order
        network_names = []
        for node in node_networks:
            if not node in network_names:
                network_names.append(node)
        network_labels = [network_names.index(node) for node in node_networks]

        network_borders = np.argwhere(np.diff(network_labels)!=0)
        ticks_position = []
        last_line_position = 0
        for i in network_borders:
            # 0.5 is the visualization offset of imshow
            line_position = i[0]+1-0.5
            axis.axvline(x=line_position, color='k', linewidth=1)
            axis.axhline(y=line_position, color='k', linewidth=1)
            ticks_position.append((line_position+last_line_position)/2)
            last_line_position = line_position
        line_position = len(node_networks)+1-0.5
        ticks_position.append((line_position+last_line_position)/2)

        axis.set_xticks(ticks_position)
        axis.set_yticks(ticks_position)
        axis.set_xticklabels(network_names, rotation=90, fontsize=13)
        axis.set_yticklabels(network_names, fontsize=13)
    
    axis.set_title(title, fontsize=18)

    return im

def visualize_conn_mat_dict(data, title='', \
    cmap='seismic',\
    normalize=False,\
    disp_diag=True,\
    save_image=False, output_root=None, axes=None, fig=None, \
    fix_lim=True, center_0=True, \
    node_networks=None, segmented=False \
    ):

    '''
    - data must be a dict of connectivity matrices
    sample:
    Suptitle1
        0.00 0.31 0.76 
        0.31 0.00 0.43 
        0.76 0.43 0.00 
    Suptitle1
        0.00 0.32 0.76 
        0.32 0.00 0.45 
        0.76 0.45 0.00 
    '''

    if node_networks is None:
        fig_width = 25*(len(data)/10)
    else:
        fig_width = 60*(len(data)/10)
    fig_height = 5

    fig_flag = True
    if axes is None or fig is None:
        fig_flag = False

    if not fig_flag:
        fig, axes = plt.subplots(1, len(data), figsize=(fig_width, fig_height), \
            facecolor='w', edgecolor='k')

    if not type(axes) is np.ndarray:
        axes = np.array([axes])

    fig.suptitle(title, fontsize=20, y=0.98) #, fontsize=20, size=20

    axes = axes.ravel()

    # normalizing and scale
    conn_mats = list()
    V_MAX_all = None
    for i, key in enumerate(data):
        
        if segmented:
            C = segment_FC(data[key], node_networks)
        else:
            C = data[key]

        if normalize:
            C = dFC_mat_normalize(C[None,:,:], global_normalization=False, threshold=0.0)[0]

        if not disp_diag:
            C = np.multiply(C, 1-np.eye(len(C)))
            C = C + np.mean(C.flatten()) * np.eye(len(C))

        if V_MAX_all is None:
            V_MAX_all = np.max(np.abs(C))
        else:
            V_MAX_all = max(V_MAX_all, np.max(np.abs(C)))

        conn_mats.append(C)
    conn_mats = np.array(conn_mats)

    if np.any(conn_mats<0) or center_0: 
        V_MIN = -1
        V_MAX = 1
    else: 
        V_MIN = 0
        V_MAX = 1

    if not fix_lim:
        V_MAX = V_MAX_all
        if np.any(conn_mats<0) or center_0:
            V_MIN = -1 * V_MAX_all
        else:
            V_MIN = 0

    # plot
    for i, key in enumerate(data):

        C = conn_mats[i,:,:]

        im = visualize_conn_mat(C, axis=axes[i], title=key, \
            cmap=cmap,\
            V_MIN=V_MIN, V_MAX=V_MAX, \
            node_networks=node_networks \
            )
    if not fig_flag:
        fig.subplots_adjust(
            bottom=0.1, \
            top=0.85, \
            left=0.1, \
            right=0.9,
            # wspace=0.02, \
            # hspace=0.02\
        )

        if not node_networks is None:
            fig.subplots_adjust(
                wspace=0.55 
            )
            
    l, b, w, h = axes[-1].get_position().bounds
    if fig_flag:
        cb_ax = fig.add_axes([0.91, b, 0.007, h])
    else:
        cb_ax = fig.add_axes([0.91, b, 0.01, h])
    cbar = fig.colorbar(im, cax=cb_ax, shrink=0.8) # shrink=0.8??

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


def visualize_conn_mat_2D_dict(data, title='', \
    cmap='seismic',\
    normalize=False,\
    disp_diag=True,\
    save_image=False, output_root=None, \
    fix_lim=True, center_0=True, \
    node_networks=None, segmented=False \
    ):

    '''
    - data must be a 2D dict of connectivity matrices
    sample:
    ROW1 (method_1)
        COLUMN1 (method_1)
            data[method_1][method_1]
                0.00 0.31 0.76 
                0.31 0.00 0.43 
                0.76 0.43 0.00 
        COLUMN2 (method_2)
            data[method_1][method_2]
                0.00 0.31 0.76 
                0.31 0.00 0.43 
                0.76 0.43 0.00 
    ROW2 (method_2)
        COLUMN1 (method_1)
            data[method_2][method_1]
                0.00 0.31 0.76 
                0.31 0.00 0.43 
                0.76 0.43 0.00 
    '''

    if node_networks is None:
        fig_width = 30*(len(data)/10)
    else:
        fig_width = 55*(len(data)/10) + 4
    fig_height = fig_width * 1.0

    fig, axs = plt.subplots(len(data), len(data), figsize=(fig_width, fig_height), \
        facecolor='w', edgecolor='k')

    if not type(axs) is np.ndarray:
        axs = np.array([axs])

    fig.suptitle(title, fontsize=25, y=0.98) #, fontsize=20, size=20

    # axs = axs.ravel()

    # normalizing and scale
    conn_mats = list()
    V_MAX_all = None
    for i, key_i in enumerate(data):
        for j, key_j in enumerate(data[key_i]):
            
            if segmented:
                C = segment_FC(data[key_i][key_j], node_networks)
            else:
                C = data[key_i][key_j]
            

            if normalize:
                C = dFC_mat_normalize(C[None,:,:], global_normalization=False, threshold=0.0)[0]

            if not disp_diag:
                C = np.multiply(C, 1-np.eye(len(C)))
                C = C + np.mean(C.flatten()) * np.eye(len(C))

            if V_MAX_all is None:
                V_MAX_all = np.max(np.abs(C))
            else:
                V_MAX_all = max(V_MAX_all, np.max(np.abs(C)))

            conn_mats.append(C)
    conn_mats = np.array(conn_mats)

    if np.any(conn_mats<0) or center_0: 
        V_MIN = -1
        V_MAX = 1
    else: 
        V_MIN = 0
        V_MAX = 1

    if not fix_lim:
        V_MAX = V_MAX_all
        if np.any(conn_mats<0) or center_0:
            V_MIN = -1 * V_MAX_all
        else:
            V_MIN = 0
    
    # plot
    axs_plotted = list()
    for i, key_i in enumerate(data):

        for j, key_j in enumerate(data[key_i]):

            if segmented:
                C = segment_FC(data[key_i][key_j], node_networks)
            else:
                C = data[key_i][key_j]

            if normalize:
                C = dFC_mat_normalize(C[None,:,:], global_normalization=False, threshold=0.0)[0]

            if not disp_diag:
                C = np.multiply(C, 1-np.eye(len(C)))
                C = C + np.mean(C.flatten()) * np.eye(len(C))

            im = visualize_conn_mat(C, axis=axs[i][j], title=key_i + ' and ' + key_j, \
                cmap=cmap,\
                V_MIN=V_MIN, V_MAX=V_MAX, \
                node_networks=node_networks \
                )

            axs_plotted.append(axs[i][j])

    # remove extra subplots
    for ax in axs.ravel():
        if not ax in axs_plotted:
            ax.set_axis_off()
            ax.xaxis.set_tick_params(which='both', labelbottom=True)

    fig.subplots_adjust(
        bottom=0.1, \
        top=0.95, \
        left=0.1, \
        right=0.9,
        wspace=0.001, \
        hspace=0.4\
    )

    if not node_networks is None:
        fig.subplots_adjust(
            wspace=0.45, 
            hspace=0.50
        )
        
    l, b, w, h = axs[-1][-1].get_position().bounds
    if node_networks is None:
        cb_ax = fig.add_axes([0.91, 0.5-h/2, 0.007, h])
    else:
        cb_ax = fig.add_axes([0.91, 0.5-h/2, 0.015, h])
    cbar = fig.colorbar(im, cax=cb_ax, shrink=0.8) # shrink=0.8??

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


def dist_mat_dendo(dist_mat, labels, title='', \
    save_image=False, output_root=None, \
    ):

    # convert the redundant n*n square matrix form into a condensed nC2 array
    distArray = ssd.squareform(dist_mat) 

    width = int(2.5*dist_mat.shape[0])
    fig = plt.figure(figsize=(width, 5))
    ax = fig.add_subplot(1, 1, 1)    
    with mpl.rc_context({'lines.linewidth': 3}):
        dend = shc.dendrogram(shc.linkage(distArray, method='single', metric='euclidean'), distance_sort='ascending', no_plot=False, labels=labels)
    plt.title(title, fontsize=15)
    ax.tick_params(axis='x', which='major', labelsize=15)
    ax.tick_params(axis='y', which='major', labelsize=15)    
    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+title+'.png', \
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad \
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
        plt.savefig(output_root+title+'.png', \
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad \
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
    plt.title(title)
    if save_image:
        folder = output_root[:output_root.rfind('/')]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(output_root+'.png', \
            dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad \
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

