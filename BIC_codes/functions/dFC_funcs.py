#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:34:49 2021

@author: mte
"""

# from tkinter import N
import numpy as np
from scipy import signal
import scipy.spatial.distance as ssd
import scipy.cluster.hierarchy as shc
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from nilearn.plotting import plot_markers
import networkx as nx
from scipy.spatial import distance
from scipy import stats
from joblib import Parallel, delayed
import os
import time
import hdf5storage
import scipy.io as sio
import pandas as pd
# np.seterr(invalid='ignore')

# ########## bundled brain graph visualizer ##########


# import panel as pn
# import datashader as ds
# import datashader.transfer_functions as tf
# from datashader.layout import random_layout, circular_layout, forceatlas2_layout
# from datashader.bundling import connect_edges, hammer_bundle
# from datashader import utils
# import holoviews as hv
# from itertools import chain

# import warnings

# warnings.simplefilter('ignore')

################################# Parameters ####################################

fig_dpi = 120
fig_bbox_inches = 'tight'
fig_pad = 0.1

################################# Other Functions ####################################

def find_new_order(old_list, new_list):
    '''
    new_order is a list of indices
    old_list = ['E', 'B', 'A', 'C', 'D']
    new_list = ['A', 'B', 'C', 'D', 'E']
    '''
    new_order = [old_list.index(a) for a in new_list]  
    return new_order

def mat_reorder(A, new_order):
    '''
    new_order must be a list of indices:
    old_list = ['E', 'B', 'A', 'C', 'D']
    new_list = ['A', 'B', 'C', 'D', 'E']
    new_order = find_new_order(old_list, new_list)
    A_sorted is a copy of A
    '''
    A_sorted = deepcopy(A)

    A_sorted = [[A_sorted[i][j] for j in new_order] for i in new_order]
    A_sorted = np.array(A_sorted)
    return A_sorted

# test
def get_subj_ts_dict(time_series_dict, subjs_id):
    subj_ts_dict = {}
    for session in time_series_dict:
        subj_ts_dict[session] = time_series_dict[session].get_subj_ts(subjs_id=subjs_id)
    return subj_ts_dict

# test
def filter_dFCM_lst(dFCM_lst, **param_dict):
    dFCM_lst2check = list()
    for dFCM in dFCM_lst:
        if dFCM.measure.param_match(**param_dict):
            dFCM_lst2check.append(dFCM) 
    return dFCM_lst2check

def SW_downsample(data, Fs, W, n_overlap, tapered_window=False):
    '''
    data = (n_time, ...)
    the time samples will be picked after 
    averaging over a window which slides
    W is in sec
    SWed_data = (n_time_new, ...)
    '''

    SWed_data = list()
    L = data.shape[0]
    # change W to timepoints
    W = int(W * Fs) 
    step = int((1-n_overlap)*W)
    if step == 0:
        step = 1

    window_taper = signal.windows.gaussian(W, std=3*W/22)

    TR_array = list()
    for l in range(0, L-W+1, step):

        ######### creating a rectangel window ############
        window = np.zeros((L))
        window[l:l+W] = 1
        
        ########### tapering the window ##############
        if tapered_window:
            window = signal.convolve(window, window_taper, mode='same') / sum(window_taper)

        # int(l-W/2):int(l+3*W/2) is the nonzero interval after tapering
        SWed_data.append(np.average(data, weights=window, axis=0))
        
        TR_array.append(int((l + (l+W)) / 2) )
        
    
    SWed_data = np.array(SWed_data)
    return SWed_data

def mutual_information(X, Y, N_bins=100):
    """ Mutual information for joint histogram
    https://matthew-brett.github.io/teaching/mutual_information.html#:~:text=Mutual%20information%20is%20a%20measure,signal%20intensity%20in%20the%20first.
    """

    # 2D histogram
    hist_2d, x_edges, y_edges = np.histogram2d(
                                                X,
                                                Y,
                                                bins=N_bins)
    
    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

# test
def normalizeAdjacency(W):
    """
    NormalizeAdjacency: Computes the [0, 1]-normalized adjacency matrix

    Input:

        W (np.array): adjacency matrix

    Output:

        W_norm (np.array): [0, 1] normalized adjacency matrix
    """
    W_norm = W - np.min(W)
    W_norm = np.divide(W_norm, np.max(W_norm))
    return W_norm 

# test
def normalized_euc_dist(x, y):
    # https://stats.stackexchange.com/questions/136232/definition-of-normalized-euclidean-distance#:~:text=The%20normalized%20squared%20euclidean%20distance,not%20related%20to%20Mahalanobis%20distance.

    if np.linalg.norm(x-np.mean(x))**2==0 and np.linalg.norm(y-np.mean(y))**2==0:
        return 0
    return 0.5*((np.linalg.norm((x-np.mean(x)) - (y-np.mean(y)))**2)/(np.linalg.norm(x-np.mean(x))**2 + np.linalg.norm(y-np.mean(y))**2))

def calc_graph_propoerty(A, property, threshold=False, binarize=False):
    """
    calc_graph_propoerty: Computes Graph-based properties 
    of adjacency matrix A
    A is converted to positive before calc
    property:
        - ECM: Computes Eigenvector Centrality Mapping (ECM) 
        - shortest_path
        - degree
        - clustering_coef

    Input:

        A (np.array): adjacency matrix (must be >0)

    Output:

        graph-property (np.array): a vector
    """
    N_edges = 200 # number of edges to keep 
    if property=='shortest_path' or property=='clustering_coef':
        threshold==True

    G = nx.from_numpy_matrix(np.abs(A)) 
    G.remove_edges_from(nx.selfloop_edges(G))
    # G = G.to_undirected()

    # pruning edges 
    if threshold:
        labels = [d["weight"] for (u, v, d) in G.edges(data=True)]
        labels.sort()
        threshold = labels[-1*N_edges]
        ebunch = [(u, v) for u, v, d in G.edges(data=True) if d['weight']<threshold]
        G.remove_edges_from(ebunch)

    if binarize:
        weight='None'
    else:
        weight='weight'

    if property=='ECM':
        graph_property = nx.eigenvector_centrality(G, weight=weight)
        graph_property = [graph_property[node] for node in graph_property]
        graph_property = np.array(graph_property)
    if property=='shortest_path':
        SHORTEST_PATHS = dict(nx.shortest_path_length(G, weight=weight))
        graph_property = np.zeros((A.shape[0], A.shape[0]))
        for node_i in SHORTEST_PATHS:
            for node_j in SHORTEST_PATHS[node_i]:
                graph_property[node_i, node_j] = SHORTEST_PATHS[node_i][node_j]
        graph_property = graph_property + graph_property.T
        graph_property = dFC_mat2vec(graph_property)
    if property=='degree':
        graph_property = [G.degree(weight=weight)[node] for node in G]
        graph_property = np.array(graph_property)
    if property=='clustering_coef':
        graph_property = nx.clustering(G, weight=weight)
        graph_property = [graph_property[node] for node in graph_property]
        graph_property = np.array(graph_property)

    return graph_property

def rank_norm(dFC_mat):
    '''
    dFC_mat = (n_time, n_region, n_region)
    dFC_mat_norm = rank_norm(dFC_mat)
    '''
    dFC_mat_new = deepcopy(dFC_mat)
    flag_dim = False
    if len(dFC_mat_new.shape)<3:
        dFC_mat_new = np.expand_dims(dFC_mat_new, axis=0)
        flag_dim = True
    assert dFC_mat_new.shape[1]==dFC_mat_new.shape[2], \
        'dimension mismatch.'
    n_region = dFC_mat_new.shape[1]
    dFC_vecs = dFC_mat2vec(dFC_mat_new)
    dFC_vecs_new = list()
    for i, vec in enumerate(dFC_vecs):
        dFC_vecs_new.append(stats.rankdata(vec))
    dFC_vecs_new = np.array(dFC_vecs_new)
    dFC_mat_new = dFC_vec2mat(dFC_vecs_new, N=n_region)
    if flag_dim:
        dFC_mat_new = np.squeeze(dFC_mat_new)
    return dFC_mat_new

def cat_data(X_t, N):
    '''
    X_t = (time, roi, roi)
    X_t is preferable to be ranked prior to cat_data
    '''
    X_t_new = list()
    for X in X_t:
        borders = np.linspace(1, np.max(X), N, endpoint=False)
        score_mat = np.zeros(X.shape)
        for border in borders:
            score_mat += X >= border
        X_t_new.append(score_mat)
    X_t_new = np.array(X_t_new)
    return X_t_new

def dFC_mask(dFC_mat, mask):
    '''
    dFC_mat and mask will be vectorized using dFC_mat2vec
    mask = (roi, roi)
    '''
    dFC_vecs = dFC_mat2vec(dFC_mat)
    mask_vec = dFC_mat2vec(mask)

    dFC_vec_new = list()
    for dFC_vec in dFC_vecs:
        dFC_vec_new.append(dFC_vec[mask_vec])
    dFC_vec_new = np.array(dFC_vec_new)

    return dFC_vec_new
    
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
    # zip measure names
    flag=False
    if not '_' in name:
        name = name + '_'
        flag=True
    if 'CAP' in name:
        new_name = 'CAP' + name[name.rfind('_'):]
    if 'SWC' in name:
        new_name = 'Clustering' + name[name.rfind('_'):]
    if 'CHMM' in name:
        new_name = 'ContinuousHMM' + name[name.rfind('_'):]
    if 'WL' in name:
        new_name = 'Windowless' + name[name.rfind('_'):]
    if 'DHMM' in name:
        new_name = 'DiscreteHMM' + name[name.rfind('_'):]
    if 'TF' in name:
        new_name = 'Time-Freq' + name[name.rfind('_'):]
    if 'SW_' in name:
        new_name = 'SlidingWindow' + name[name.rfind('_'):]
    if flag:
        new_name = new_name[:-1]
    return new_name

#test
def dFC_mat2vec(C_t):
    '''
    C_t must be an array of matrices or a single matrix
    diagonal values not included. if you want to include 
    them set k=0
    if C_t is a single matrix, F will be one dim
    changing F will not change C_t
    '''
    if len(C_t.shape)==2:
        assert C_t.shape[0]==C_t.shape[1],\
            'C is not a square matrix'
        return C_t[np.triu_indices(C_t.shape[1], k=1)]

    F = list()
    for t in range(C_t.shape[0]):
        C = C_t[t, : , :]
        assert C.shape[0]==C.shape[1],\
            'C is not a square matrix'
        F.append(C[np.triu_indices(C_t.shape[1], k=1)])

    F = np.array(F)
    return F

#test
def dFC_vec2mat(F, N):
    '''
    diagonal values are set to 1.0
    F shape is (observations, features)
    '''
    C = list()
    iu = np.triu_indices(N, k=1)
    for i in range(F.shape[0]):
        K = np.zeros((N, N))
        K[iu] = F[i,:]
        K = K + K.T
        K = K + np.eye(N)
        C.append(K)
    C = np.array(C)
    return C

# test
def common_subj_lst(time_series_dict):
    SUBJECTs = None
    for session in time_series_dict:
        if SUBJECTs is None:
            SUBJECTs = time_series_dict[session].subj_id_lst
        else:
            SUBJECTs = intersection(SUBJECTs, time_series_dict[session].subj_id_lst)
    return SUBJECTs

def intersection(lst1, lst2): # input is a list 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def TR_intersection(dFCM_lst): # input is a list of dFCM objs
    TRs_lst_old = dFCM_lst[0].TR_array
    common_Fs = dFCM_lst[0].TS_info['Fs']
    for dFCM in dFCM_lst:
        assert dFCM.TS_info['Fs'] == common_Fs, \
            'Fs mismatch. Cannot find the common TRs'
            
        TRs_lst_new = intersection(TRs_lst_old, dFCM.TR_array)
        TRs_lst_old = TRs_lst_new
    TRs_lst_old.sort()
    if len(TRs_lst_old)==0:
        print('No TR intersection.')
    return TRs_lst_old

def dFC_dict_slice(data, idx_lst):
    data_sliced = {}
    for i, k in enumerate(data):
        if i in idx_lst:
            data_sliced[k] = data[k]
    return data_sliced

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

def node_info2network(nodes_info):
    node_networks = []
    for info in nodes_info:
        if info[3]=='Network':
            continue
        node_networks.append(info[3])    
    return node_networks

def segment_FC(FC, node_networks):
    unique_node_networks = list(set(node_networks))
    segmented = np.zeros_like(FC)
    for network_i in unique_node_networks:
        node_id_i = [idx for idx, value in enumerate(node_networks) if value==network_i]
        for network_j in unique_node_networks:
            node_id_j = [idx for idx, value in enumerate(node_networks) if value==network_j]
            segmented[node_id_i[0]:node_id_i[-1]+1, node_id_j[0]:node_id_j[-1]+1] = np.mean(FC[node_id_i[0]:node_id_i[-1]+1, node_id_j[0]:node_id_j[-1]+1])
    return segmented
        
def segment_FC_dict(FC_dict, node_networks):
    segmented_dict = {}
    for key in FC_dict:
        segmented_dict[key] = segment_FC(FC_dict[key], node_networks)
    return segmented_dict
        
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
    g = sns.catplot(data=df, x=x, y=y, kind=kind
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

'''
########## bundled brain graph visualizer ##########

cvsopts = dict(plot_height=400, plot_width=400)

def thresh_G(G, threshold):
    
    G_copy = deepcopy(G)
    
    if threshold > 1:
        labels = [d["weight"] for (u, v, d) in G_copy.edges(data=True)]
        labels.sort()
        threshold = labels[-1*threshold]
    
    ebunch = [(u, v) for u, v, d in G_copy.edges(data=True) if np.abs(d['weight']) < threshold]
    G_copy.remove_edges_from(ebunch)
        
    return G_copy

def nodesplot(nodes, name=None, canvas=None, cat=None):
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    # aggregator=None if cat is None else ds.count_cat(cat)
    # agg=canvas.points(nodes,'x','y',aggregator)
    aggc = canvas.points(nodes, 'x', 'y', ds.count_cat('cat'))   #ds.by('cat', ds.count())
        
    color_key = dict(cat_normal='#FF3333', cat_sig='#00FF00')
    
    return tf.spread(tf.shade(aggc, color_key=color_key), px=4, name=name)


def edgesplot(edges, name=None, canvas=None):
    canvas = ds.Canvas(**cvsopts) if canvas is None else canvas
    return tf.shade(canvas.line(edges, 'x','y', agg=ds.count()), name=name)
    
def graphplot(nodes, edges, name="", canvas=None, cat=None):
    
    if canvas is None:
        xr = nodes.x.min(), nodes.x.max()
        yr = nodes.y.min(), nodes.y.max()
        canvas = ds.Canvas(x_range=xr, y_range=yr, **cvsopts)
        
    np = nodesplot(nodes, name + " nodes", canvas, cat)
    ep = edgesplot(edges, name + " edges", canvas)
    return tf.stack(ep, np, how="over", name=name)

def ng(graph,name):
    graph.name = name
    return graph

def nx_layout(graph, view_degree=0, threshold=0):
    # layout = nx.circular_layout(graph)
    
    # Get node positions
    pos = nx.get_node_attributes(graph, 'pos')
    for key in pos:
        if view_degree==0:    
            pos[key] = pos[key][:2]
        if view_degree==1:    
            pos[key] = pos[key][1:3]
        if view_degree==2:    
            pos[key] = pos[key][[0, 2]]
            
    # layout = pos
            
    cat = list()
    for key in graph.nodes():
        cat.append('cat_normal')
        # if key in find_sig_nodes(graph):   #
        #     cat.append( 'cat_sig')
        # else:
        #     cat.append('cat_normal')
            
    data = [[node]+pos[node].tolist()+[cat[i]] for i, node in enumerate(graph.nodes)]

    nodes = pd.DataFrame(data, columns=['id', 'x', 'y','cat'])
    nodes.set_index('id', inplace=True)
    nodes["cat"]=nodes["cat"].astype("category")

    graph_copy = thresh_G(graph, threshold=threshold)

    edges = pd.DataFrame(list(graph_copy.edges), columns=['source', 'target'])
    return nodes, edges

def nx_plot(graph, name="", view_degree=0, threshold=0):
    # print(graph.name, len(graph.edges))
    nodes, edges = nx_layout(graph, view_degree=view_degree, threshold=threshold)
    
    direct = connect_edges(nodes, edges)
    bundled_bw005 = hammer_bundle(nodes, edges)
    bundled_bw030 = hammer_bundle(nodes, edges, initial_bandwidth=0.30)
    bundled_bw100 = hammer_bundle(nodes, edges, initial_bandwidth=1)

    return [graphplot(nodes, direct,         graph.name, cat=None),
            graphplot(nodes, bundled_bw005, "Bundled bw=0.05", cat=None),
            graphplot(nodes, bundled_bw030, "Bundled bw=0.30", cat=None),
            graphplot(nodes, bundled_bw100, "Bundled bw=1.00", cat=None)]

def batch_Adj2Net(FCS, nodes_info, is_digraph=False):
  
    np.fill_diagonal(FCS, 0)
    if is_digraph:
        G = nx.from_numpy_matrix(FCS, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_matrix(FCS)

    mapping = {}
    for i, node_info in enumerate(nodes_info):
        mapping[i] = node_info[4]
    G = nx.relabel_nodes(G, mapping)

    return G

def set_locs_G(G, locs):
    
    G_copy = deepcopy(G)
    
    pos = nx.circular_layout(G_copy) 

    for i, key in enumerate(pos):
        pos[key] = locs[i]
        
    nx.set_node_attributes(G_copy, pos, "pos") 
      
    
    return G_copy 

def visulize_brain_graph(FCS, nodes_info, locs, num_edges2show, \
    title='', save_image=True, output_root=None \
    ):
    
    # EXAMPLE:
    # visulize_brain_graph(measure.FCS_dict[FCS], measure.TS_info['nodes_info'], \
    # measure.TS_info['nodes_locs'], num_edges2show=100, \
    # title=FCS+'_'+measure.measure_name, save_image=save_image, output_root=output_root \
    # )
    
    G = batch_Adj2Net(FCS=FCS, nodes_info=nodes_info, is_digraph=False)
    G = set_locs_G(G, locs=locs)   
    plots = [nx_plot(ng(G, name="dFC"), view_degree=0, threshold=num_edges2show)]

    if save_image:
        ds.utils.export_image(img=plots[0][2], filename=title+'_bundle_', 
                        fmt=".png", background='black', 
                        export_path=output_root)
    
    # return plots[0][0]


##############################
'''

def dFC_dict_normalize(D, global_normalization=False, threshold=0.0):

    C = list()
    for key in D:
        C.append(D[key])
    C = np.array(C)

    C_z = dFC_mat_normalize(C, \
        global_normalization=global_normalization, \
        threshold=threshold \
    )

    D_z = {}
    for i, key in enumerate(D):
        D_z[key] = C_z[i,:,:]

    return D_z

def dFC_mat_normalize(C_t, global_normalization=False, threshold=0.0):

    # threshold is ratio of connections wanted to be zero
    C_t_z = deepcopy(C_t)
    if len(C_t_z.shape)<3:
        C_t_z = np.expand_dims(C_t_z, axis=0)

    if global_normalization:

        # transform the whole abs(dFC mat) to [0, 1] 

        signs = np.sign(C_t_z)
        C_t_z = np.abs(C_t_z)

        miN = list()
        for i in range(C_t_z.shape[0]):
            slice = C_t_z[i,:,:]
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0],dtype=bool))]
            miN.append(np.min(slice_non_diag))

        C_t_z = C_t_z - np.min(miN)

        maX = list()
        for i in range(C_t_z.shape[0]):
            slice = C_t_z[i,:,:]
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0],dtype=bool))]
            maX.append(np.max(slice_non_diag))

        if np.max(maX) != 0:
            C_t_z = np.divide(C_t_z, np.max(maX))

        # thresholding
        d = deepcopy(np.ravel(C_t_z))
        d.sort()
        new_threshold = d[int(threshold*len(d))]
        C_t_z = np.multiply(C_t_z, (C_t_z>=new_threshold))
        C_t_z = np.multiply(C_t_z, signs)

    else:

        # transform abs of each time slice to [0, 1]

        signs = np.sign(C_t_z)
        C_t_z = np.abs(C_t_z)
        
        for i in range(C_t_z.shape[0]):
            slice = C_t_z[i,:,:]
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0],dtype=bool))]
            slice = slice - np.min(slice_non_diag)
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0],dtype=bool))]
            if np.max(slice_non_diag) != 0:
                slice = np.divide(slice, np.max(slice_non_diag))

            # thresholding
            d = deepcopy(np.ravel(slice))
            d.sort()
            new_threshold = d[int(threshold*len(d))]
            slice = np.multiply(slice, (slice>=new_threshold))

            C_t_z[i,:,:] = slice

        C_t_z = np.multiply(C_t_z, signs)

    # removing self connections
    for i in range(C_t_z.shape[1]):
        C_t_z[:, i, i] = np.mean(C_t_z) # ?????????????????

    return C_t_z

def print_mat(mat, s=0):
    if len(mat.shape)==1:
        mat = np.expand_dims(mat, axis=0)
    for i in mat:
        print('\t'*s,  end=" ")
        for j in i:
            print("{:.2f}".format(j), end=" ")
        print()

def print_dict(t, s=0):
    if not isinstance(t,dict) and not isinstance(t,list):
        if isinstance(t,np.ndarray):
            print_mat(t, s)
        else:
            if isinstance(t,float):
                print('\t'*s+"{:.2f}".format(t))
            else:
                print('\t'*s+str(t))
    else:
        for key in t:
            print('\t'*s+str(key))
            if not isinstance(t,list):
                print_dict(t[key],s+1)

############################# dFC Analyzer class ################################

"""

todo:
- 
"""

class DFC_ANALYZER:
    # if self.n_jobs is None => no parallelization

    def __init__(self, analysis_name='', **params):

        self.analysis_name = analysis_name
        
        self.params = params
        if not 'n_jobs' in self.params:
            self.params['n_jobs'] = -1
        if not 'verbose' in self.params:
            self.params['verbose'] = 1
        if not 'backend' in self.params:
            self.params['backend'] = 'loky'

        self.MEASURES_lst_ = None
        self.MEASURES_fit_lst_ = []
        self.MEASURES_name_lst = []
        self.params_methods = {}
        self.alter_hparams = {}
        self.hyper_param_info = {}

    @property
    def MEASURES_lst(self):
        assert not self.MEASURES_lst_ is None, \
            'first set the MEASURES_lst!'
        return self.MEASURES_lst_

    @property
    def MEASURES_fit_lst(self):
        return self.MEASURES_fit_lst_

    def set_MEASURES_lst(self, MEASURES_lst):
        self.MEASURES_lst_ = MEASURES_lst

    def set_MEASURES_fit_lst(self, MEASURES_fit_lst):
        self.MEASURES_fit_lst_ = MEASURES_fit_lst

    def measures_initializer(self, MEASURES_name_lst, params_methods, alter_hparams):

        '''
        - this will test values in hyper_params other than
            values already in self.params. values in self.params 
            will be considered the reference
        sample:
        hyper_params = { \
            'n_states': [6, 12, 16], \
            'normalization': [True], \
            'num_subj': [50, 100, 395], \
            'num_select_nodes': [50, 100, 333], \
            'num_time_point': [500, 800, 1200], \
            'Fs_ratio': [0.50, 1.00, 1.50], \
            'noise_ratio': [0.00, 0.50, 1.00], \
            'num_realization': [1, 2, 3], \
            }
            
            MEASURES_name_lst = ( \
                'SlidingWindow', \
                'Time-Freq', \
                'CAP', \
                'ContinuousHMM', \
                'Windowless', \
                'Clustering', \
                'DiscreteHMM' \
                )
        '''

        self.MEASURES_name_lst = MEASURES_name_lst
        self.params_methods = params_methods
        self.alter_hparams = alter_hparams

        # a list of MEASURES with default parameter values
        MEASURES_lst = self.create_measure_obj(MEASURES_name_lst=MEASURES_name_lst, **params_methods)

        # adding MEASURES with alternative parameter values
        hyper_param_info = {}
        hyper_param_info['default_values'] = params_methods
        for hyper_param in alter_hparams:
            for value in alter_hparams[hyper_param]:
                params = deepcopy(params_methods)
                params[hyper_param] = value
                hyper_param_info[hyper_param+'_'+str(value)] = deepcopy(params)
                new_MEASURES = self.create_measure_obj(MEASURES_name_lst=MEASURES_name_lst, **params)
                for new_measure in new_MEASURES:
                    flag=0
                    for MEASURE in MEASURES_lst:
                        if new_measure.issame(MEASURE):
                            flag=1
                    if flag==0:
                        MEASURES_lst.append(new_measure)

        self.hyper_param_info = hyper_param_info

        return MEASURES_lst

    def create_measure_obj(self, MEASURES_name_lst, **params):

        MEASURES_lst = list()
        for MEASURES_name in MEASURES_name_lst:

            ###### CAP ######
            if MEASURES_name=='CAP':
                measure = CAP(**params)

            ###### CONTINUOUS HMM ######
            if MEASURES_name=='ContinuousHMM':
                measure = HMM_CONT(**params)

            ###### WINDOW_LESS ######
            if MEASURES_name=='Windowless':
                measure = WINDOWLESS(**params)

            ###### SLIDING WINDOW ######
            if MEASURES_name=='SlidingWindow':
                measure = SLIDING_WINDOW(**params)

            ###### TIME FREQUENCY ######
            if MEASURES_name=='Time-Freq':
                measure = TIME_FREQ(**params)

            ###### SLIDING WINDOW + CLUSTERING ######
            if MEASURES_name=='Clustering':
                measure = SLIDING_WINDOW_CLUSTR(**params)

            ###### DISCRETE HMM ######
            if MEASURES_name=='DiscreteHMM':
                measure = HMM_DISC(**params)

            MEASURES_lst.append(measure)

        return MEASURES_lst

    def SB_MEASURES_lst(self, MEASURES_lst): # returns state_based measures
        SB_MEASURES = list()
        for measure in MEASURES_lst:
            if measure.is_state_based:
                SB_MEASURES.append(measure)
        return SB_MEASURES

    def DD_MEASURES_lst(self, MEASURES_lst): # returns data_driven measures
        DD_MEASURES = list()
        for measure in MEASURES_lst:
            if not measure.is_state_based:
                DD_MEASURES.append(measure)
        return DD_MEASURES

    ##################### FCS ESTIMATION ######################

    def estimate_group_FCS(self, time_series_dict):

        # time_series_dict is a dict of time_series

        for session in time_series_dict:

            time_series = time_series_dict[session]
            SB_MEASURES_lst = self.SB_MEASURES_lst(self.MEASURES_lst)
            if self.params['n_jobs'] is None:
                SB_MEASURES_lst_NEW = list()
                for measure in SB_MEASURES_lst:
                    SB_MEASURES_lst_NEW.append( \
                        measure.estimate_FCS(time_series=time_series) \
                        )
            else:
                SB_MEASURES_lst_NEW = Parallel( \
                    n_jobs=self.params['n_jobs'], verbose=self.params['verbose'], backend=self.params['backend'])( \
                    delayed(measure.estimate_FCS)(time_series=time_series) \
                        for measure in SB_MEASURES_lst)
            self.MEASURES_fit_lst_[session] = self.DD_MEASURES_lst(self.MEASURES_lst) + SB_MEASURES_lst_NEW

    ##################### dFCM ASSESSMENT ######################

    def group_dFCM_assess(self, time_series_dict):

        # time_series_dict is a dict of time_series

        SUBJ_s_dFCM_dict = {}
        
        SUBJECTs = common_subj_lst(time_series_dict) 

        if self.params['n_jobs'] is None:
            OUT = list()
            for subject in SUBJECTs:
                OUT.append( \
                    self.subj_lvl_dFC_assess( \
                    time_series_dict=get_subj_ts_dict(time_series_dict, subjs_id=subject), \
                    ))
        else:
            OUT = Parallel( \
                        n_jobs=self.params['n_jobs'], \
                        verbose=self.params['verbose'], \
                        backend=self.params['backend'])( \
                    delayed(self.subj_lvl_dFC_assess)( \
                        time_series_dict=get_subj_ts_dict(time_series_dict, subjs_id=subject), \
                        ) \
                        for subject in SUBJECTs)
        
        return OUT

    def subj_lvl_dFC_assess(self, time_series_dict):

        # time_series_dict is a dict of time_series

        dFCM_dict = {}
        # dFC_corr_assess_dict = {}

        if self.params['n_jobs'] is None:
            dFCM_lst = list()
            for measure in self.MEASURES_fit_lst_:
                dFCM_lst.append( \
                    measure.estimate_dFCM(time_series=time_series_dict[measure.params['session']]) \
                )
        else:
            dFCM_lst = Parallel( \
                n_jobs=self.params['n_jobs'], verbose=self.params['verbose'], backend=self.params['backend'])( \
                delayed(measure.estimate_dFCM)(time_series=time_series_dict[measure.params['session']]) \
                    for measure in self.MEASURES_fit_lst_)

        dFCM_dict['dFCM_lst'] = dFCM_lst

        return dFCM_dict

################################# Similarity_Assessment class ####################################

class SIMILARITY_ASSESSMENT:

    def __init__(self, dFCM_lst):
        self.dFCM_lst = dFCM_lst

    ##################### dFC FEATURES ######################

    def FO_calc(self, dFCM_lst, common_TRs=None):

        # returns, for each state the Fractional Occupancy (FO)
        # see Vidaurre et al., 2017
        # it only considers TRs in common_TRs

        if common_TRs is None:
            common_TRs = TR_intersection(dFCM_lst)

        FO_list = list()
        for dFCM in dFCM_lst:
        
            FO = {}

            if dFCM.measure.is_state_based:

                state_act_dict = dFCM.state_act_dict(TRs=common_TRs)
                
                for FCS_key in state_act_dict['state_TC']:
                    FO[FCS_key] = np.mean(state_act_dict['state_TC'][FCS_key]['act_TC'])

            FO_list.append(FO)

        return FO_list

    def transition_freq(self, dFCM_lst, common_TRs=None):
        # returns the number of total state transition within common_TRs -> trans_freq
        # and the number of total state transitions regardless of common_TRs
        # but normalized by total number of TRs -> trans_norm

        if common_TRs is None:
            common_TRs = TR_intersection(dFCM_lst)

        TRs_lst = list()
        for TR in common_TRs:
            TRs_lst.append('TR'+str(TR))

        trans_freq_lst = list()
        for dFCM in dFCM_lst:
        
            trans_freq_dict = {}

            if dFCM.measure.is_state_based:

                trans_freq = 0
                last_TR = None
                for TR in dFCM.FCS_idx:
                    if TR in TRs_lst:
                        if not last_TR is None:
                            if dFCM.FCS_idx[TR]!=dFCM.FCS_idx[last_TR]:
                                trans_freq += 1
                        last_TR = TR

                trans_freq_dict['trans_freq'] = trans_freq

                trans_norm = 0
                last_TR = None
                for TR in dFCM.FCS_idx:
                    if not last_TR is None:
                        if dFCM.FCS_idx[TR]!=dFCM.FCS_idx[last_TR]:
                            trans_norm += 1
                    last_TR = TR
                trans_norm = trans_norm / len(dFCM.FCS_idx)

                trans_freq_dict['trans_norm'] = trans_norm

            trans_freq_lst.append(trans_freq_dict)

        return trans_freq_lst 

    def feature_all(self, dFC_mat):
        vectorized_dFC = dFC_mat2vec(dFC_mat).flatten() # (time*connection, )
        vectorized_dFC = np.expand_dims(vectorized_dFC, axis=0) # (1, time*connection)
        return vectorized_dFC

    def feature_spatial(self, dFC_mat):
        conn_over_time = dFC_mat2vec(dFC_mat) # (time, connection)
        return conn_over_time

    def feature_temporal(self, dFC_mat):
        conn_over_time = dFC_mat2vec(dFC_mat) # (time, connection)
        time_over_conn = conn_over_time.T # (connection, time)
        return time_over_conn

    def feature_inter_time_corr(self, dFC_mat):
        '''
        returns correspondence of inter-time relation between results of dFC 
        measures in each subject
        '''
        conn_over_time = dFC_mat2vec(dFC_mat) # (time, connection)
        inter_time_corr = np.corrcoef(conn_over_time) # (time, time)
        inter_time_corr = np.nan_to_num(inter_time_corr)
        inter_time_corr = dFC_mat2vec(inter_time_corr) # (time*(time-1)/2, )
        inter_time_corr = np.expand_dims(inter_time_corr, axis=0) # (1, time*(time-1)/2)
        return inter_time_corr

    def feature_inter_conn_corr(self, dFC_mat):
        conn_over_time = dFC_mat2vec(dFC_mat) # (time, connection)
        time_over_conn = conn_over_time.T # (connection, time)
        inter_conn_corr = np.corrcoef(time_over_conn) # (connection, connection)
        inter_conn_corr = np.nan_to_num(inter_conn_corr)
        inter_conn_corr = dFC_mat2vec(inter_conn_corr) # (connection*(connection-1)/2, )
        inter_conn_corr = np.expand_dims(inter_conn_corr, axis=0) # (1, connection*(connection-1)/2)
        return inter_conn_corr

    def feature_dFC_avg(self, dFC_mat):
        dFC_avg = np.mean(dFC_mat, axis=0) # (ROI, ROI)
        vectorized_dFC_avg = dFC_mat2vec(dFC_avg) # (connection, )
        vectorized_dFC_avg = np.expand_dims(vectorized_dFC_avg, axis=0) # (1, connection)
        return vectorized_dFC_avg

    def feature_dFC_var(self, dFC_mat):
        dFC_var = np.var(dFC_mat, axis=0) # (ROI, ROI)
        vectorized_dFC_var = dFC_mat2vec(dFC_var) # (connection, )
        vectorized_dFC_var = np.expand_dims(vectorized_dFC_var, axis=0) # (1, connection)
        return vectorized_dFC_var

    def feature_graph_spatial(self, dFC_mat, graph_property):
        graph_feature_over_time = list()
        for FC_mat in dFC_mat:
            graph_feature = calc_graph_propoerty(FC_mat, property=graph_property, threshold=False, binarize=False)
            graph_feature_over_time.append(graph_feature)
        graph_feature_over_time = np.array(graph_feature_over_time) # (time, ROI)
        return graph_feature_over_time

    def feature_graph_temporal(self, dFC_mat, graph_property):
        graph_feature_over_time = list()
        for FC_mat in dFC_mat:
            graph_feature = calc_graph_propoerty(FC_mat, property=graph_property, threshold=False, binarize=False)
            graph_feature_over_time.append(graph_feature)
        graph_feature_over_time = np.array(graph_feature_over_time) # (time, ROI)
        graph_feature_over_node = graph_feature_over_time.T # (ROI, time)
        graph_feature_avg = np.mean(graph_feature_over_node, axis=0) # (time, )
        graph_feature_avg = np.expand_dims(graph_feature_avg, axis=0) # (1, time)
        return graph_feature_avg

    def extract_feature(self, dFC_mat, feature2extract, graph_property=None):
        '''
        feature2extract_list = [
            'all', 
            'spatial', 'temporal', 
            'inter_time_corr', 'inter_conn_corr', 
            'dFC_avg', 'dFC_var', 
            'graph_spatial', 'graph_temporal'
        ]
        '''
        feature = None
        if feature2extract=='all':
            feature = self.feature_all(dFC_mat)
        if feature2extract=='spatial':
            feature = self.feature_spatial(dFC_mat)
        if feature2extract=='temporal':
            feature = self.feature_temporal(dFC_mat)
        if feature2extract=='inter_time_corr':
            feature = self.feature_inter_time_corr(dFC_mat)
        if feature2extract=='inter_conn_corr':
            feature = self.feature_inter_conn_corr(dFC_mat)
        if feature2extract=='dFC_avg':
            feature = self.feature_dFC_avg(dFC_mat)
        if feature2extract=='dFC_var':
            feature = self.feature_dFC_var(dFC_mat)
        if feature2extract=='graph_spatial':
            feature = self.feature_graph_spatial(dFC_mat, graph_property=graph_property)
        if feature2extract=='graph_temporal':
            feature = self.feature_graph_temporal(dFC_mat, graph_property=graph_property)

        return feature

    def dFC_mat_lst_similarity(self, dFC_mat_lst, feature2extract, metric, graph_property=None):
        
        sim_mat_over_sample = None
        for i, dFC_mat_i in enumerate(dFC_mat_lst):
            for j, dFC_mat_j in enumerate(dFC_mat_lst):

                if j<=i:
                    continue

                assert dFC_mat_i.shape==dFC_mat_j.shape,\
                    'shape mismatch'

                feature_i = self.extract_feature(
                    dFC_mat_i, 
                    feature2extract=feature2extract, 
                    graph_property=graph_property
                ) # (samples, variables)
                feature_j = self.extract_feature(
                    dFC_mat_j, 
                    feature2extract=feature2extract, 
                    graph_property=graph_property
                ) # (samples, variables)

                sim_over_sample = list()
                for sample in range(feature_i.shape[0]):
                    if np.var(feature_i[sample, :])==0 or np.var(feature_j[sample, :])==0:
                        sim = 0
                    else:
                        if metric=='corr':
                            sim = np.corrcoef(feature_i[sample, :], feature_j[sample, :])[0,1]
                        elif metric=='spearman':
                            sim, p = stats.spearmanr(feature_i[sample, :], feature_j[sample, :])
                        elif metric=='MI':
                            sim = mutual_information(X=feature_i[sample, :], Y=feature_j[sample, :], N_bins=100)
                        elif metric=='euclidean_distance':
                            # normalized euclidean is used
                            sim = normalized_euc_dist(x=feature_i[sample, :], y=feature_j[sample, :])
                    sim_over_sample.append(sim)
                
                if sim_mat_over_sample is None:
                    sim_mat_over_sample = np.zeros((len(sim_over_sample), len(dFC_mat_lst), len(dFC_mat_lst)))
                sim_mat_over_sample[:, i, j] = np.array(sim_over_sample)
                sim_mat_over_sample[:, j, i] = sim_mat_over_sample[:, i, j]

        return sim_mat_over_sample

    def assess_similarity(self, dFCM_lst, downsampling_method='default', **param_dict):
        '''
        downsampling_method: 'default' picks FCs at common_TRs 
        while 'SWed' uses a sliding window to downsample
        '''
        methods_assess = {}

        # sort dFCM_lst according to methods names
        old_list = [dFCM.measure.measure_name for dFCM in dFCM_lst]
        new_list = deepcopy(old_list)
        new_list.sort()

        new_order = find_new_order(old_list, new_list)
        dFCM_lst = [dFCM_lst[i] for i in new_order]

        common_TRs = TR_intersection(dFCM_lst)

        measure_lst = list()
        TS_info_lst = list()
        dFC_mat_lst = list()
        for dFCM in dFCM_lst:
            measure_lst.append(dFCM.measure)
            TS_info_lst.append(dFCM.TS_info)
            if downsampling_method=='SWed':
                dFC_mat_lst.append( \
                    dFCM.SWed_dFC_mat( \
                        W=param_dict['W'], \
                        n_overlap=param_dict['n_overlap'], \
                        tapered_window=param_dict['tapered_window'] \
                    )
                )
            else:
                dFC_mat_lst.append(dFCM.get_dFC_mat(TRs=common_TRs))

        methods_assess['measure_lst'] = measure_lst
        methods_assess['TS_info_lst'] = TS_info_lst
        methods_assess['common_TRs'] = common_TRs

        ########## dFCM samples ##########

        dFCM_samples = {}
        for i, dFC_mat in enumerate(dFC_mat_lst):
            dFCM_samples[str(i)] = dFC_mat
        methods_assess['dFCM_samples'] = dFCM_samples

        ########## time record ##########
        
        time_record_dict = {}
        for i, dFCM in enumerate(dFCM_lst):
            time_record = {}
            time_record['FCS_fit'] = dFCM.measure.FCS_fit_time
            time_record['dFC_assess'] = dFCM.measure.dFC_assess_time
            time_record_dict[str(i)] = time_record
        methods_assess['time_record_dict'] = time_record_dict

        ########## subj_dFC_sim ##########
        # returns correlation/MI/spearman corr/euclidean distance between results of dFC 
        # measures in a subject
        feature2extract_list = [
            # 'all', 
            'spatial', 'temporal', 
            'inter_time_corr', 'inter_conn_corr', 
            'dFC_avg', 'dFC_var', 
            # 'graph_spatial', 'graph_temporal'
        ]
        metric_list = [
            'corr',
            'spearman',
            'MI',
            'euclidean_distance'
        ]
        graph_property_list = [
            'ECM',
            'shortest_path',
            'degree',
            'clustering_coef'
        ]
        methods_assess['all'] = {}
        for metric in metric_list:
            methods_assess['all'][metric] = self.dFC_mat_lst_similarity(
                                                dFC_mat_lst, 
                                                feature2extract='all', 
                                                metric=metric
                                                )
        methods_assess['feature_based'] = {}
        for feature2extract in feature2extract_list:
            methods_assess['feature_based'][feature2extract] = self.dFC_mat_lst_similarity(
                                                dFC_mat_lst, 
                                                feature2extract=feature2extract, 
                                                metric='spearman'
                                                )
        methods_assess['graph_based'] = {}
        methods_assess['graph_based']['graph_spatial'] = {}
        methods_assess['graph_based']['graph_temporal'] = {}
        for graph_property in graph_property_list:
            methods_assess['graph_based']['graph_spatial'][graph_property] = self.dFC_mat_lst_similarity(
                                                dFC_mat_lst, 
                                                feature2extract='graph_spatial', 
                                                metric='spearman',
                                                graph_property=graph_property
                                                )
            methods_assess['graph_based']['graph_temporal'][graph_property] = self.dFC_mat_lst_similarity(
                                                dFC_mat_lst, 
                                                feature2extract='graph_temporal', 
                                                metric='spearman',
                                                graph_property=graph_property
                                                )
        # ########## dFC temporal average and variance ##########

        methods_assess['dFC_avg'] = [self.feature_dFC_avg(dFC_mat) for dFC_mat in dFC_mat_lst]

        methods_assess['dFC_var'] = [self.feature_dFC_var(dFC_mat) for dFC_mat in dFC_mat_lst]
        
        ########## Fractional Occupancy ##########

        FO_lst = self.FO_calc(dFCM_lst, \
            common_TRs=common_TRs \
            )
        methods_assess['FO'] = FO_lst

        ########## transition frequency ##########

        trans_freq_lst = self.transition_freq(dFCM_lst, \
            common_TRs=common_TRs \
                )
        methods_assess['trans_freq'] = trans_freq_lst
                
        ##############################################
        return methods_assess

    def run(self, FILTERS, downsampling_method='default'):
        '''
        downsampling_method: 'default' picks FCs at common_TRs 
        while 'SWed' uses a sliding window to downsample
        '''
        parallelize = True
        output = {}
        if parallelize:
            out_lst = Parallel( \
                    n_jobs=4, verbose=0, backend='loky')( \
                    delayed(self.assess_similarity)(dFCM_lst=filter_dFCM_lst(self.dFCM_lst, **FILTERS[filter]), \
                    downsampling_method=downsampling_method, \
                    **FILTERS[filter]) \
                        for filter in FILTERS)
            for i, filter in enumerate(FILTERS):
                output[filter] = out_lst[i]
        else:
            for filter in FILTERS:
                param_dict = FILTERS[filter]
                dFCM_lst2check = filter_dFCM_lst(self.dFCM_lst, **param_dict)
                output[filter] = self.assess_similarity( \
                    dFCM_lst=dFCM_lst2check, \
                    downsampling_method=downsampling_method, \
                    **param_dict \
                    )

        return output

################################# dFC class ####################################

"""
todo:
- type annotation
"""

class dFC:

    TF_methods_name_lst = [ \
        'CWT_mag', \
        'CWT_phase_r', \
        'CWT_phase_a', \
        'WTC' \
    ]

    sw_methods_name_lst = [ \
        'pear_corr', \
        'MI', \
        'GraphLasso', \
    ]

    base_methods_name_lst = ['SlidingWindow', 'Time-Freq']

    def __init__(self):
        self.measure_name = ''
        self.is_state_based = bool()
        self._stat = []
        self.TPM = []
        self.params = {}
        self.TS_info_ = {}
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.logs_ = ''

    @property
    def FCS_fit_time(self):
        return self.FCS_fit_time_

    @property
    def dFC_assess_time(self):
        return self.dFC_assess_time_

    @property
    def TS_info(self):
        # info of the time series used to train/estimate FCSs
        return self.TS_info_

    @property
    def is_state_based(self):
        return self.params['is_state_based']

    @property
    def FCS(self):
        return self.FCS_

    # test
    @property
    def FCS_dict(self):
        # returns a dict including FCS matrices

        if not self.is_state_based:
            return None

        C_A = self.FCS
        FCSs = {}
        for k in range(C_A.shape[0]):
            FCSs['FCS'+str(k+1)] = C_A[k,:,:]
            
        return FCSs

    @property
    def info(self):
        return self.params

    @property
    def logs(self):
        print(self.logs_)

    def issame(self, dFC):
        if type(self)==type(dFC):
            for param_name in self.params:
                if self.params[param_name] != dFC.params[param_name]:
                    return False
        else:
            return False
        return True

    #test
    def param_match(self, **param_dict):
        for param in param_dict:
            if param in self.params:
                if type(param_dict[param]) is list:
                    if not self.params[param] in param_dict[param]:
                        return False
                else:
                    if self.params[param]!=param_dict[param]:
                        return False
        return True

    def set_FCS_fit_time(self, time):
        self.FCS_fit_time_ = time

    def set_dFC_assess_time(self, time):
        self.dFC_assess_time_ = time

    def set_mean_activity(self, time_series):
        # mean activity of regions at each state
        if self.is_state_based:
            if 'sw_method' in self.params_name_lst:
                SUBJECTs = time_series.subj_id_lst
                TS_data = None
                for subject in SUBJECTs:
                    subj_TS = time_series.get_subj_ts(subjs_id=subject).data
                    new_TS_data = SW_downsample(data=subj_TS.T, \
                        Fs=time_series.Fs, W=self.params['W'], \
                        n_overlap=self.params['n_overlap'], \
                        tapered_window=self.params['tapered_window'] \
                    ).T
                    if TS_data is None:
                        TS_data = new_TS_data
                    else:
                        TS_data = np.concatenate((TS_data, new_TS_data), axis=1)
            else:
                TS_data = time_series.data
            mean_act = list()
            for i in np.unique(self.Z):
                ids = np.array([int(state==i) for state in self.Z])
                mean_act.append(np.average(TS_data, weights=ids, axis=1))
            self.mean_act = np.array(mean_act)
        else:
            self.mean_act = None

    def estimate_FCS(self, time_series=None):
        pass

    def estimate_dFCM(self, time_series=None):
        pass

    def manipulate_time_series4FCS(self, time_series):

        new_time_series = deepcopy(time_series)

        # SUBJECTs
        new_time_series.select_subjs(num_subj=self.params['num_subj'])
        # SPATIAL RESOLUTION
        new_time_series.spatial_downsample(num_select_nodes=self.params['num_select_nodes'], rand_node_slct=False)
        # TEMPORAL RESOLUTION
        new_time_series.Fs_resample(Fs_ratio=self.params['Fs_ratio'])
        # NORMALIZE
        if self.params['normalization']:
            new_time_series.normalize()
        # NOISE
        new_time_series.add_noise(noise_ratio=self.params['noise_ratio'], mean_noise=0)
        # NUMBER OF TIME POINTS
        new_time_series.truncate(start_point=0, end_point=self.params['num_time_point']-1)

        self.TS_info_ = new_time_series.info_dict

        return new_time_series

    def manipulate_time_series4dFC(self, time_series):

        new_time_series = deepcopy(time_series)

        # SPATIAL RESOLUTION
        new_time_series.spatial_downsample(num_select_nodes=self.params['num_select_nodes'], rand_node_slct=False)
        # TEMPORAL RESOLUTION
        new_time_series.Fs_resample(Fs_ratio=self.params['Fs_ratio'])
        # NORMALIZE
        if self.params['normalization']:
            new_time_series.normalize()
        # NOISE
        new_time_series.add_noise(noise_ratio=self.params['noise_ratio'], mean_noise=0)
        # NUMBER OF TIME POINTS
        new_time_series.truncate(start_point=0, end_point=self.params['num_time_point']-1)

        return new_time_series
    
    def visualize_states(self):
        pass

    # todo : use FCS_dict func in this func
    def visualize_FCS(self,
                normalize=True, fix_lim=True, 
                save_image=False, output_root=None
                ):
        
        if self.FCS == []:
            return

        if normalize:
            D = dFC_dict_normalize(D=self.FCS_dict, global_normalization=False)
        else:
            D = self.FCS_dict

        fig_width = 55*(len(D)/10)
        fig_height = 8

        fig, axes = plt.subplots(2, len(D), figsize=(fig_width, fig_height), \
            facecolor='w', edgecolor='k')

        fig.subplots_adjust(
            bottom=0.1, \
            top=0.85, \
            left=0.1, \
            right=0.9,
            wspace=0.3, \
            hspace=0.8\
        )

        # plot mean activity
        for i, mean_act in enumerate(self.mean_act):
            plot_markers(
                node_values=mean_act, 
                node_coords=self.TS_info['nodes_locs'], 
                node_cmap='hot', 
                display_mode='z', 
                colorbar=False, axes=axes[1, i]
            )

        # plot FC pattern
        node_networks = node_info2network(self.TS_info['nodes_info'])

        visualize_conn_mat_dict(data=D, \
            node_networks=node_networks, \
            title=self.measure_name+' FCS', \
            save_image=save_image, \
            axes=axes[0, :], fig=fig, 
            output_root=output_root, \
            disp_diag=False, \
            fix_lim=fix_lim \
        )

    def visualize_TPM(self, normalize=True, save_image=False, output_root=None):
        
        if self.TPM == []:
            return
        if normalize:
            C = dFC_mat_normalize(C_t=np.expand_dims(self.TPM, axis=0), threshold=0.0)
        else:
            C = np.expand_dims(self.TPM, axis=0)

        plt.figure(figsize=(5, 5))
        plt.imshow(np.squeeze(C), interpolation='nearest', aspect='equal', cmap='jet')
        cb=plt.colorbar(shrink=0.8)
        plt.title(self.measure_name + ' TPM')
        
        if save_image:
            folder = output_root[:output_root.rfind('/')]
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(output_root+'.png', \
                dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad \
            ) 
            plt.close()
        else:
            plt.show()


################################## NEW METHOD ##################################

'''
by : web link

Reference: ##

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:

import needed_toolbox

class method_name(dFC):

    def __init__(self, **params):
        self.FCS_ = []
        self.logs_ = ''

        self.params_name_lst = ['measure_name', 'is_state_based', 'n_states', \
            'normalization', 'num_subj', 'num_select_nodes', 'num_time_point', \
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]

        self.params['specific_param'] = value
        self.params['measure_name'] = 'method_name'
        self.params['is_state_based'] = True/False
    
    @property
    def measure_name(self):
        return self.params['measure_name'] 

    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # start timing
        tic = time.time()

        # calc FCSs

        # calc self.Z

        # mean activation of states
        self.set_mean_activity(time_series)

        # record time
        self.set_FCS_fit_time(time.time() - tic)

        return self

    def estimate_dFCM(self, time_series):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        assert len(time_series.subj_id_lst)==1, \
            'this function takes only one subject as input.'

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        # calc FCSs and FCS_idx

        # record time
        self.set_dFC_assess_time(time.time() - tic)
            
        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=self.FCS_, FCS_idx=FCS_idx, TS_info=time_series.info_dict)
        return dFCM
'''

################################## CAP ##################################

'''
by : web link

Reference: ##

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
'''
from sklearn.cluster import KMeans

class CAP(dFC):

    def __init__(self, **params):
        self.logs_ = ''
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'n_states', \
            'n_subj_clstrs', 'normalization', 'num_subj', 'num_select_nodes', 'num_time_point', \
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]

        self.params['measure_name'] = 'CAP'
        self.params['is_state_based'] = True

    @property
    def measure_name(self):
        return self.params['measure_name'] 

    def act_vec2FCS(self, act_vecs):
        FCS_ = list()
        for act_vec in act_vecs:
            FCS_.append(np.multiply(act_vec[:, np.newaxis], act_vec[np.newaxis, :]))
        return np.array(FCS_)

    def cluster_act_vec(self, act_vecs, n_clusters):

        kmeans_ = KMeans(n_clusters=n_clusters, n_init=500).fit(act_vecs)
        act_centroids = kmeans_.cluster_centers_

        return act_centroids, kmeans_

    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # start timing
        tic = time.time()

        # 2-level clustering
        SUBJECTs = time_series.subj_id_lst
        act_center_1st_level = None
        for subject in SUBJECTs:
            
            act_vecs = time_series.get_subj_ts(subjs_id=subject).data.T

            # test
            if act_vecs.shape[0]<self.params['n_subj_clstrs']:
                print( \
                    'Number of subject-level clusters cannot be more than time samples! n_subj_clstrs was changed to ' \
                        + str(act_vecs.shape[0]))
                self.params['n_subj_clstrs'] = act_vecs.shape[0]

            act_centroids, _ = self.cluster_act_vec( \
                act_vecs = act_vecs, \
                n_clusters = self.params['n_subj_clstrs'] \
                )
            if act_center_1st_level is None:
                act_center_1st_level = act_centroids
            else:
                act_center_1st_level = np.concatenate((act_center_1st_level, act_centroids), axis=0)
        
        group_act_centroids, self.kmeans_= self.cluster_act_vec( \
            act_vecs=act_center_1st_level, \
            n_clusters = self.params['n_states'] \
            )
        self.FCS_ = self.act_vec2FCS(group_act_centroids)
        self.Z = self.kmeans_.predict(time_series.data.T)

        # mean activation of states
        self.set_mean_activity(time_series)

        # record time
        self.set_FCS_fit_time(time.time() - tic)

        return self

    def estimate_dFCM(self, time_series):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        assert len(time_series.subj_id_lst)==1, \
            'this function takes only one subject as input.'

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()
                    
        act_vecs = time_series.data.T

        Z = self.kmeans_.predict(act_vecs)

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=self.FCS_, \
            FCS_idx=Z, \
            TS_info=time_series.info_dict \
            )
        return dFCM

################################# HMM Continuous ###############################

"""
by hmmlearn

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
- number of iter?
- ValueError: 'covars' must be symmetric, positive-definite
"""

from hmmlearn import hmm

class HMM_CONT(dFC):

    def __init__(self, **params):
        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'n_states', 'hmm_iter', \
            'normalization', 'num_subj', 'num_select_nodes', 'num_time_point', \
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]

        self.params['measure_name'] = 'ContinuousHMM'
        self.params['is_state_based'] = True

    @property
    def measure_name(self):
        return self.params['measure_name'] 

    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        # start timing
        tic = time.time()

        time_series = self.manipulate_time_series4FCS(time_series)

        Models, Scores = [], []
        for i in range(self.params['hmm_iter']):
            model = hmm.GaussianHMM(n_components=self.params['n_states'], covariance_type="full")
            model.fit(time_series.data.T) 
            score = model.score(time_series.data.T)
            Models.append(model)
            Scores.append(score)
            
        self.hmm_model = Models[np.argmax(Scores)]
        self.Z = self.hmm_model.predict(time_series.data.T)
        self.means_ = self.hmm_model.means_
        self.FCS_ = self.hmm_model.covars_ 
        self.TPM = self.hmm_model.transmat_
        self.pi = self.hmm_model.startprob_

        # mean activation of states
        self.set_mean_activity(time_series)

        # record time
        self.set_FCS_fit_time(time.time() - tic)

        return self

    def estimate_dFCM(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."
        
        assert len(time_series.subj_id_lst)==1, \
            'this function takes only one subject as input.'

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        Z = self.hmm_model.predict(time_series.data.T)

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=self.FCS_, FCS_idx=Z, TS_info=time_series.info_dict)

        return dFCM

################################## Windowless ##################################

"""
by : https://github.com/nel215/ksvd

Reference: Rubinstein, R., Zibulevsky, M. and Elad, M., Efficient Implementation 
of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit Technical 
Report - CS Technion, April 2008

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
"""
from ksvd import ApproximateKSVD

class WINDOWLESS(dFC):

    def __init__(self, **params):
        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'n_states', \
            'normalization', 'num_subj', 'num_select_nodes', 'num_time_point', \
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]

        self.params['measure_name'] = 'Windowless'
        self.params['is_state_based'] = True
    
    @property
    def measure_name(self):
        return self.params['measure_name'] 

    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # start timing
        tic = time.time()

        # time_series ~ gamma.dot(dictionary)
        self.aksvd = ApproximateKSVD(n_components=self.params['n_states'], transform_n_nonzero_coefs=1)
        self.dictionary = self.aksvd.fit(time_series.data.T).components_
        self.gamma = self.aksvd.transform(time_series.data.T)

        self.FCS_ = np.zeros([self.params['n_states'], time_series.n_regions, time_series.n_regions])
        for i in range(self.params['n_states']):
            self.FCS_[i, :, :] = np.multiply(np.expand_dims(self.dictionary[i,:], axis=0).T, np.expand_dims(self.dictionary[i,:], axis=0))

        self.Z = list()
        for i in range(self.gamma.shape[0]):
            self.Z.append(np.argwhere(self.gamma[i, :] != 0)[0,0])

        # mean activation of states
        self.set_mean_activity(time_series)

        # record time
        self.set_FCS_fit_time(time.time() - tic)

        return self

    def estimate_dFCM(self, time_series):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        assert len(time_series.subj_id_lst)==1, \
            'this function takes only one subject as input.'

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        gamma = self.aksvd.transform(time_series.data.T)

        Z = list()
        for i in range(time_series.n_time):
            Z.append(np.argwhere(gamma[i, :] != 0)[0,0])

        # record time
        self.set_dFC_assess_time(time.time() - tic)
            
        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=self.FCS_, FCS_idx=Z, TS_info=time_series.info_dict)
        return dFCM

################################# Time-Frequency #################################

"""
PyCWT: 
Authors
Sebastian Krieger, Nabil Freij, Alexey Brazhe, Christopher Torrence, 
Gilbert P. Compo and contributors.

Disclaimer
This module is based on routines provided by C. Torrence and G. P. Compo available 
at http://paos.colorado.edu/research/wavelets/, on routines provided by A. Grinsted, 
J. Moore and S. Jevrejeva available at 
http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence, and on routines 
provided by A. Brazhe available at http://cell.biophys.msu.ru/static/swan/.

This software is released under a BSD-style open source license. Please read 
the license file for further information. This routine is provided as is 
without any express or implied warranties whatsoever.

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.
    dj : float, optional
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    s0 : float, optional
        Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
        Number of scales less one. Scales range from s0 up to
        s0 * 2**(J * dj), which gives a total of (J + 1) scales.
        Default is J = (log2(N*dt/so))/dj.
    sig : bool 
        set to compute signficance, default is True
    significance_level (float, optional) :
        Significance level to use. Default is 0.95.
    normalize (boolean, optional) :
        If set to true, normalizes CWT by the standard deviation of
        the signals.

- if n_jobs is None => no parallelization

todo:

- consider COI and edge effect in averaging:
    => should we truncate the time points having at less than 20 freqs as done in Savva et al. ?

"""
import pycwt as wavelet

class TIME_FREQ(dFC):

    def __init__(self, coi_correction=True, **params):
        
        # assert TF_method in self.TF_methods_name_lst, \
        #     "Time-frequency method not recognized."

        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'TF_method', 'coi_correction', \
            'n_jobs', 'verbose', 'backend', \
            'normalization', 'num_select_nodes', 'num_time_point', \
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
        
        self.params['measure_name'] = 'Time-Freq'
        self.params['is_state_based'] = False
        self.params['coi_correction'] = coi_correction

    @property
    def measure_name(self):
        return self.params['measure_name'] # + '_' + self.params['TF_method']

    def coi_correct(self, X, coi, freqs):
        # correct the edge effect in matrix X = [freqs, time] using coi
        # if self.coi_correction=True

        if not self.params['coi_correction']:
            return X
        periods = 1/freqs
        periods = np.repeat(periods[:, None], X.shape[1], axis=1)
        coi = np.repeat(coi[None, :], X.shape[0], axis=0)
        X_corrected = np.multiply(X, (coi>=periods))
        return X_corrected

    def WT_dFC(self, Y1, Y2, Fs, J, s0, dj):
        if self.params['TF_method']=='CWT_mag' or self.params['TF_method']=='CWT_phase_r' or self.params['TF_method']=='CWT_phase_a':
            # Cross Wavelet Transform
            WT_xy, coi, freqs, _ = wavelet.xwt(Y1, Y2, dt=1/Fs, dj=dj, s0=s0, J=J, 
                significance_level=0.95, wavelet='morlet', normalize=True)

            if self.params['TF_method']=='CWT_mag':
                WT_xy_corrected = self.coi_correct(WT_xy, coi, freqs)
                wt = np.abs(np.mean(WT_xy_corrected, axis=0))

            if self.params['TF_method']=='CWT_phase_r' or self.params['TF_method']=='CWT_phase_a':
                cosA = np.cos(np.angle(WT_xy))
                sinA = np.sin(np.angle(WT_xy))

                cosA_corrected = self.coi_correct(cosA, coi, freqs)
                sinA_corrected = self.coi_correct(sinA, coi, freqs)

                A = (cosA_corrected + sinA_corrected * 1j)

                if self.params['TF_method']=='CWT_phase_r':
                    wt = np.abs(np.mean(A, axis=0))
                else:
                    wt = np.angle(np.mean(A, axis=0))
        
        if self.params['TF_method']=='WTC':
            # Wavelet Transform Coherence
            WT_xy, _, coi, freqs, _ = wavelet.wct(Y1, Y2, dt=1/Fs, dj=dj, s0=s0, J=J, 
                sig=False, significance_level=0.95, wavelet='morlet', normalize=True)
            WT_xy_corrected = self.coi_correct(WT_xy, coi, freqs)
            wt = np.abs(np.mean(WT_xy_corrected, axis=0))

        return wt

    def estimate_dFCM(self, time_series):
        
        '''
        we assume calc is applied on subjects separately
        '''
        assert len(time_series.subj_id_lst)==1, \
            'this function takes only one subject as input.'

        # params
        J = 100 # -1
        s0 = 1 # -1
        dj = 1/12 # 1/12

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        WT = np.zeros((time_series.n_time, \
            time_series.n_regions, time_series.n_regions))

        for i in range(time_series.n_regions):
            if self.params['n_jobs'] is None:
                Q = list()
                for j in range(time_series.n_regions):
                    Q.append(self.WT_dFC( \
                        Y1=time_series.data[i, :], \
                        Y2=time_series.data[j, :], \
                        Fs=time_series.Fs, \
                        J=J, s0=s0, dj=dj))
            else:
                Q = Parallel( \
                    n_jobs=self.params['n_jobs'], verbose=self.params['verbose'], backend=self.params['backend'])( \
                    delayed(self.WT_dFC)( \
                                    Y1=time_series.data[i, :], \
                                    Y2=time_series.data[j, :], \
                                    Fs=time_series.Fs, \
                                    J=J, s0=s0, dj=dj) \
                                    for j in range(time_series.n_regions) \
                                                                )
            WT[:, i, :] = np.array(Q).T

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=WT, TS_info=time_series.info_dict)
        return dFCM

################################# Sliding-Window #################################

"""

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
"""

from sklearn.covariance import GraphicalLassoCV, graphical_lasso

class SLIDING_WINDOW(dFC):

    def __init__(self, **params):
        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'sw_method', 'tapered_window', \
            'W', 'n_overlap', 'normalization', \
            'num_select_nodes', 'num_time_point', 'Fs_ratio', \
            'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]

        self.params['measure_name'] = 'SlidingWindow'
        self.params['is_state_based'] = False

        assert self.params['sw_method'] in self.sw_methods_name_lst, \
            "sw_method not recognized."
        
    
    @property
    def measure_name(self):
        return self.params['measure_name'] #+ '_' + self.sw_method

    def shan_entropy(self, c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized* np.log2(c_normalized))  
        return H

    def calc_MI(self, X, Y):
        
        bins = 20
        
        c_XY = np.histogram2d(X,Y,bins)[0]
        c_X = np.histogram(X,bins)[0]
        c_Y = np.histogram(Y,bins)[0]
        
        H_X = self.shan_entropy(c_X)
        H_Y = self.shan_entropy(c_Y)
        H_XY = self.shan_entropy(c_XY)
        
        MI = H_X + H_Y - H_XY
        return MI

    def FC(self, time_series):
    
        if self.params['sw_method']=='GraphLasso':
            model = GraphicalLassoCV()
            model.fit(time_series.T)
            C = model.covariance_
        else:
            C = np.zeros((time_series.shape[0], time_series.shape[0]))
            for i in range(time_series.shape[0]):
                for j in range(i, time_series.shape[0]):
                    
                    X = time_series[i, :]
                    Y = time_series[j, :]

                    if self.params['sw_method']=='MI':
                        ########### Mutual Information ##############
                        C[j, i] = self.calc_MI(X, Y)
                    else:
                        ########### Pearson Correlation ##############
                        if np.var(X)==0 or np.var(Y)==0:
                            C[j, i] = 0
                        else:
                            C[j, i] = np.corrcoef(X, Y)[0, 1]

                    C[i, j] = C[j, i]   
                
        return C

    def dFC(self, time_series, W=None, n_overlap=None, tapered_window=False):
        # W is in time samples
        
        L = time_series.shape[1]
        step = int((1-n_overlap)*W)
        if step == 0:
            step = 1

        window_taper = signal.windows.gaussian(W, std=3*W/22)
        # C = DFCM(measure=self)
        FCSs = list()
        TR_array = list()
        for l in range(0, L-W+1, step):

            ######### creating a rectangel window ############
            window = np.zeros((L))
            window[l:l+W] = 1
            
            ########### tapering the window ##############
            if tapered_window:
                window = signal.convolve(window, window_taper, mode='same') / sum(window_taper)

            window = np.repeat(np.expand_dims(window, axis=0), time_series.shape[0], axis=0)

            # int(l-W/2):int(l+3*W/2) is the nonzero interval after tapering
            FCSs.append(self.FC( \
                        np.multiply(time_series, window)[ \
                            :,max(int(l-W/2),0):min(int(l+3*W/2),L) \
                                                        ] \
                                )
                        )
            TR_array.append(int((l + (l+W)) / 2) )

        return np.array(FCSs), np.array(TR_array)
    
    def estimate_dFCM(self, time_series):
        
        '''
        we assume calc is applied on subjects separately
        '''
        assert len(time_series.subj_id_lst)==1, \
            'this function takes only one subject as input.'

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        # W is converted from sec to samples
        FCSs, TR_array = self.dFC(time_series=time_series.data, \
            W=int(self.params['W'] * time_series.Fs) , \
            n_overlap=self.params['n_overlap'], \
            tapered_window=self.params['tapered_window'] \
            )

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.info_dict)

        return dFCM


########################### Sliding_Window + Clustering ############################

"""
- We used a tapered window as in Allen et al., created by convolving a rectangle (width = 22 TRs = 44s) 
  with a Gaussian (σ = 3 TRs) and slid in steps of 1 TR, resulting in W= 126 windows (Allen et al., 2014).
- Kmeans Clustering is repeated 500 times to escape local minima (Allen et al., 2014)
- for clustering, we have a 2-level kmeans clustering. First, we cluster FCSs of each subject. Then, we
    cluster all clustering centers from all subjects. the final estimate_dFCM is using the second kmeans
    model (Allen et al., 2014; Ou et al., 2015). 

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
- pyclustering(manhattan) has a problem when suing predict
"""

from sklearn.cluster import KMeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric

class SLIDING_WINDOW_CLUSTR(dFC):

    def __init__(self, clstr_distance='euclidean', **params):

        assert clstr_distance=='euclidean' or clstr_distance=='manhattan', \
            "Clustering distance not recognized. It must be either \
                euclidean or manhattan."
        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'clstr_base_measure', 'sw_method', 'tapered_window', \
            'clstr_distance', 'coi_correction', \
            'n_subj_clstrs', 'W', 'n_overlap', 'n_states', 'normalization', \
            'n_jobs', 'verbose', 'backend', \
            'num_subj', 'num_select_nodes', 'num_time_point', 'Fs_ratio', \
            'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
        
        self.params['measure_name'] = 'Clustering'
        self.params['is_state_based'] = True
        self.params['clstr_distance'] = clstr_distance

        assert self.params['clstr_base_measure'] in self.base_methods_name_lst, \
            "Base method not recognized."

    @property
    def measure_name(self):
        return self.params['measure_name'] #+ '_' + self.base_method

    def dFC_mat2vec(self, C_t):
        return dFC_mat2vec(C_t)
        # if len(C_t.shape)==2:
        #     assert C_t.shape[0]==C_t.shape[1],\
        #         'C is not a square matrix'
        #     return C_t[np.triu_indices(C_t.shape[1], k=0)]

        # F = list()
        # for t in range(C_t.shape[0]):
        #     C = C_t[t, : , :]
        #     assert C.shape[0]==C.shape[1],\
        #         'C is not a square matrix'
        #     F.append(C[np.triu_indices(C_t.shape[1], k=0)])

        # F = np.array(F)
        # return F

    def dFC_vec2mat(self, F, N):
        return dFC_vec2mat(F=F, N=N)
        # C = list()
        # iu = np.triu_indices(N, k=0)
        # for i in range(F.shape[0]):
        #     K = np.zeros((N, N))
        #     K[iu] = F[i,:]
        #     K = K + np.multiply(K.T, 1-np.eye(N))
        #     C.append(K)
        # C = np.array(C)
        # return C

    def clusters_lst2idx(self, clusters):
        Z = np.zeros((self.F.shape[0],))
        for i, cluster in enumerate(clusters):
            for sample in cluster:
                Z[sample] = i
        return Z.astype(int)

    def cluster_FC(self, FCS_raw, n_clusters, n_regions):

        F = self.dFC_mat2vec(FCS_raw)

        if self.params['clstr_distance']=='manhattan':
            pass
            # ########### Manhattan Clustering ##############
            # # Prepare initial centers using K-Means++ method.
            # initial_centers = kmeans_plusplus_initializer(F, self.n_states).initialize()
            # # create metric that will be used for clustering
            # manhattan_metric = distance_metric(type_metric.MANHATTAN)
            # # Create instance of K-Means algorithm with prepared centers.
            # kmeans_ = kmeans(F, initial_centers, metric=manhattan_metric)
            # # Run cluster analysis and obtain results.
            # kmeans_.process()
            # Z = self.clusters_lst2idx(kmeans_.get_clusters())
            # F_cent = np.array(kmeans_.get_centers())
        else:
            ########### Euclidean Clustering ##############
            kmeans_ = KMeans(n_clusters=n_clusters, n_init=500).fit(F)
            F_cent = kmeans_.cluster_centers_

        FCS_ = self.dFC_vec2mat(F_cent, N=n_regions)
        return FCS_, kmeans_

        
    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # start timing
        tic = time.time()

        base_dFC = None
        if self.params['clstr_base_measure']=='Time-Freq':
            base_dFC = TIME_FREQ(**self.params)
        if self.params['clstr_base_measure']=='SlidingWindow':
            base_dFC = SLIDING_WINDOW(**self.params)

        # 1-level clustering
        # dFCM_raw = base_dFC.estimate_dFCM( \
        #         time_series=time_series \
        #         )
        # self.FCS_, self.kmeans_ = self.cluster_FC( \
        # dFCM_raw.get_dFC_mat(TRs=self.dFCM_raw.TR_array), \
        # n_regions = dFCM_raw.n_regions \
        # )

        # 2-level clustering
        SUBJECTs = time_series.subj_id_lst
        FCS_1st_level = None
        SW_dFC = None
        for subject in SUBJECTs:
            
            dFCM_raw = base_dFC.estimate_dFCM( \
                time_series=time_series.get_subj_ts(subjs_id=subject) \
                )

            # test
            if dFCM_raw.n_time<self.params['n_subj_clstrs']:
                print( \
                    'Number of subject-level clusters cannot be more than SW dFCM samples! n_subj_clstrs was changed to ' \
                        + str(dFCM_raw.n_time))
                self.params['n_subj_clstrs'] = dFCM_raw.n_time

            FCS, _ = self.cluster_FC( \
                FCS_raw = dFCM_raw.get_dFC_mat(TRs=dFCM_raw.TR_array), \
                n_clusters = self.params['n_subj_clstrs'], \
                n_regions = dFCM_raw.n_regions \
                )
            
            if SW_dFC is None:
                SW_dFC = dFCM_raw.get_dFC_mat(TRs=dFCM_raw.TR_array)
            else:
                SW_dFC = np.concatenate((SW_dFC, dFCM_raw.get_dFC_mat(TRs=dFCM_raw.TR_array)), axis=0)
            if FCS_1st_level is None:
                FCS_1st_level = FCS
            else:
                FCS_1st_level = np.concatenate((FCS_1st_level, FCS), axis=0)
        
        self.FCS_, self.kmeans_ = self.cluster_FC( \
            FCS_raw=FCS_1st_level, \
            n_clusters = self.params['n_states'], \
            n_regions = dFCM_raw.n_regions \
            )
        self.Z = self.kmeans_.predict(self.dFC_mat2vec(SW_dFC))

        # mean activation of states
        self.set_mean_activity(time_series)

        # record time
        self.set_FCS_fit_time(time.time() - tic)

        return self

    def estimate_dFCM(self, time_series):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        assert len(time_series.subj_id_lst)==1, \
            'this function takes only one subject as input.'

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        base_dFC = None
        if self.params['clstr_base_measure']=='Time-Freq':
            base_dFC = TIME_FREQ(**self.params)
        if self.params['clstr_base_measure']=='SlidingWindow':
            base_dFC = SLIDING_WINDOW(**self.params)
                    
        dFCM_raw = base_dFC.estimate_dFCM(time_series=time_series)

        F = self.dFC_mat2vec(dFCM_raw.get_dFC_mat(TRs=dFCM_raw.TR_array))

        if self.params['clstr_distance']=='manhattan':
            pass
            # ########### Manhattan Clustering ##############
            # self.kmeans_.predict(F)
            # Z = self.clusters_lst2idx(self.kmeans_.get_clusters())
        else:
            ########### Euclidean Clustering ##############
            Z = self.kmeans_.predict(F)

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=self.FCS_, \
            FCS_idx=Z, \
            TS_info=time_series.info_dict, \
            TR_array=dFCM_raw.TR_array \
            )

        return dFCM

################################# HMM Discrete #################################

"""

Parameters
    ----------
    Z : numpy.ndarray
        state time course
    M : int
        (num of observations/n_state) of 16
    N : int
        (num of hidden states) of 24
    self.FCC : 
        dFCM estimated by Clustering which is then used to fit Discrete HMM 
    self.FCS_ : 
        collection FCS pattern coded in numbers for Discrete HMM

todo:
- two-level hierarchical clustering ?
- find a better name for FCC
"""
# from HMM_discrete import *
from hmmlearn import hmm

class HMM_DISC(dFC):

    def __init__(self, **params):
        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.swc = None
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        
        self.params_name_lst = ['measure_name', 'is_state_based', 'clstr_base_measure', 'sw_method', 'tapered_window', \
            'dhmm_obs_state_ratio', 'coi_correction', 'hmm_iter', \
            'n_jobs', 'verbose', 'backend', \
            'n_subj_clstrs', 'W', 'n_overlap', 'n_states', 'normalization', \
            'num_subj', 'num_select_nodes', 'num_time_point', 'Fs_ratio', \
            'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
        
        self.params['measure_name'] = 'DiscreteHMM'
        self.params['is_state_based'] = True

        assert self.params['clstr_base_measure'] in self.base_methods_name_lst, \
            "Base measure not recognized."

    @property
    def measure_name(self):
        return self.params['measure_name'] #+ '_' + self.base_method

    def estimate_FCS(self, time_series):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # start timing
        tic = time.time()

        # change n_states of swc to n_observations which is dhmm_obs_state_ratio*n_states
        params = deepcopy(self.params)
        params['n_states'] = int(self.params['dhmm_obs_state_ratio'] * self.params['n_states'])

        self.swc = SLIDING_WINDOW_CLUSTR(**params)
        self.swc.estimate_FCS(time_series=time_series)

        SUBJECTs = time_series.subj_id_lst
        SWC_dFC = None
        Obs_seq = None
        for subject in SUBJECTs:
            new_dFCM = self.swc.estimate_dFCM(time_series=time_series.get_subj_ts(subjs_id=subject))
            new_dFC_mat = new_dFCM.get_dFC_mat()
            new_Obs = new_dFCM.FCS_idx_array.reshape(-1, 1)
            if SWC_dFC is None:
                SWC_dFC = new_dFC_mat
                Obs_seq = new_Obs
            else:
                SWC_dFC = np.concatenate((SWC_dFC, new_dFC_mat), axis=0)
                Obs_seq = np.concatenate((Obs_seq, new_Obs), axis=0)

        Models, Scores = [], []
        for i in range(self.params['hmm_iter']):
            model = hmm.MultinomialHMM(n_components=self.params['n_states'])
            model.fit(Obs_seq) 
            score = model.score(Obs_seq)
            Models.append(model)
            Scores.append(score)
            
        self.hmm_model = Models[np.argmax(Scores)]
        self.Z = self.hmm_model.predict(Obs_seq)
        self.TPM = self.hmm_model.transmat_
        self.EPM = self.hmm_model.emissionprob_ 

        if len(np.unique(self.Z)) < self.params['n_states']:
            self.logs_ += 'Less number of states were fitted than n_states. \n'

        self.FCS_ = np.zeros((self.params['n_states'], \
            time_series.n_regions, time_series.n_regions))
        for i in range(self.params['n_states']):
            ids = np.array([int(state==i) for state in self.Z])
            if np.any(ids>0):
                self.FCS_[i,:,:] = np.average(SWC_dFC, weights=ids, axis=0)

        # mean activation of states
        self.set_mean_activity(time_series)

        # record time
        self.set_FCS_fit_time(time.time() - tic)

        return self

    def estimate_dFCM(self, time_series):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        assert len(time_series.subj_id_lst)==1, \
            'this function takes only one subject as input.'

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        FCC = self.swc.estimate_dFCM(time_series=time_series)
        Obs_seq = FCC.FCS_idx_array.reshape(-1, 1)

        Z = self.hmm_model.predict(Obs_seq)

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=self.FCS_, \
            FCS_idx=Z, \
            TS_info=time_series.info_dict, \
            TR_array=FCC.TR_array \
                )

        return dFCM
    
###################################################################################

################################# TIME_SERIES class ######################################

"""

todo:
- select nodes for visualizer
- default node list is chosen by arange !
"""

class TIME_SERIES():
    def __init__(self, data=None, subj_id=None, Fs=None, time_array=None, \
                locs=None, nodes_info=None, TS_name='', session_name=''):
        
        '''
        subj_id is an id to identify the subjects
        all properties are applied to every subject separately
        for instance interval applies to TS of each subj separately

        time_array of all subjects must be equal
        '''

        assert (not data is None) and (not Fs is None) and (not subj_id is None), \
            "data, subj_id, and Fs args must be provided."

        self.data_dict_ = {}
        self.data_dict_[subj_id] = {}
        self.data_dict_[subj_id]['data'] = data 
        self.data_ = None
        self.Fs_ = Fs
        self.Fs_ratio_ = 1.00
        self.noise_ratio = 0.0
        self.normalized = False
        self.TS_name_ = TS_name
        self.session_name_ = session_name
        self.n_regions_ = data.shape[0]
        self.n_time_ = data.shape[1]

        # assert self.n_regions_ < self.n_time_, \
        #     "Probably you have to transpose the time_series."

        if time_array is None:
            self.time_array_ = 1/self.Fs_ + np.arange(0, data.shape[1]/self.Fs_, 1/self.Fs_)
        else:
            self.time_array_ = time_array

        self.locs_ = locs
        self.nodes_info_ = nodes_info

        self.interval_ = np.arange(0, self.n_time_, dtype=int)
        self.nodes_selection_ = list(range(self.n_regions_))

    @property
    def info(self):
        print_dict(self.info_dict)

    @property
    def info_dict(self):
        info_dict = {}
        info_dict['n_time'] = self.n_time
        info_dict['n_regions'] = self.n_regions
        info_dict['Fs'] = self.Fs
        info_dict['Fs_ratio'] = self.Fs_ratio_
        info_dict['noise_ratio'] = self.noise_ratio
        info_dict['nodes_lst'] = self.nodes_lst
        info_dict['nodes_info'] = self.nodes_info
        info_dict['nodes_locs'] = self.locs
        info_dict['subj_id_lst'] = self.subj_id_lst
        info_dict['interval'] = self.interval
        info_dict['time'] = self.time

        return info_dict

    @property
    def data_dict(self):
        return self.data_dict_

    @property
    def data(self):
        if self.data_ is None:
            self.updatae_data()
        return self.data_

    def updatae_data(self):
        # after any change in data_dict, self.data_ is 
        # set to None and needs an update before being used
        data = None
        for subj in self.data_dict:
            if data is None:
                data = self.data_dict[subj]['data']
            else:
                data = np.concatenate((data, self.data_dict[subj]['data']), axis=1)
        self.data_ = data
        return

    @property
    def subj_id_lst(self):
        return [subj_id for subj_id in self.data_dict]

    @property
    def nodes_lst(self):
        # output shape is (n_region,) 
        return np.array(self.nodes_selection_)
        
    @property
    def interval(self):
        # output shape is (n_time,) 
        return self.interval_

    @property
    def locs(self):
        if self.locs_ is None:
            return None
        else:
            return self.locs_[self.nodes_lst, :]

    @property
    def nodes_info(self):
        if self.nodes_info_ is None:
            return None
        else:
            return [self.nodes_info_[0]] + [self.nodes_info_[i+1] for i in self.nodes_lst] 

    @property
    def Fs(self):
        return self.Fs_ * self.Fs_ratio_

    @property
    def n_time(self):
        return len(self.time)

    @property
    def n_regions(self):
        return len(self.nodes_lst)

    @property
    def time(self):
        return self.time_array_[self.interval]

    @property
    def TS_name(self):
        return self.TS_name_

    def get_subj_ts(self, subjs_id=None):
        """
        you can select time samples by their subj_id
        ! be careful about the original properties of TS hidden in new TS
        node selection will be kept.

        subjs_id is one str or a list of str
        """

        flag = 0
        if not type(subjs_id) is list:
            subjs_id = [subjs_id]
            flag = 1

        new_TS = deepcopy(self)

        SUBJECTS = [subj_id for subj_id in new_TS.data_dict_]
        for subj in SUBJECTS:
            if not subj in subjs_id:
                new_TS.data_dict_.pop(subj, None)

        if flag == 1:
            new_TS.TS_name_ = self.TS_name+' subject '+subjs_id[0]

        return new_TS


    def append_ts(self, new_time_series, time_array=None, subj_id=None):
        # append new time series to existing ones
        # truncate and node selection , etc will be automatically applied to new TS;
        # However, at first the new TS must have the same properties as the original properties of 
        # the existing TSs 

        assert self.n_regions_ == new_time_series.shape[0], \
            "Number of nodes mismatch."

        assert not subj_id is None, \
            "subj_id must be provided."

        assert not subj_id in self.data_dict_, \
            "subj_id already exists."

        self.data_dict_[subj_id] = {}

        if not time_array is None:
            assert self.time_array_ == time_array, \
                'time array mismatch!'

        self.data_dict_[subj_id]['data'] = new_time_series

        self.data_ = None


    def truncate(self, start_time=None, end_time=None, start_point=None, end_point=None):

        # truncates TS of every subj separately
        # based on either time or samples
        # if all None -> whole time_series
        #check if not out of total interval

        start = 0
        end = self.n_time

        if not start_point is None:
            start = start_point
        
        if not end_point is None:
            end = end_point + 1

        if not start_time is None:
            start = np.argwhere(self.time_array_>=start_time)[0,0]

        if not end_time is None:
            end = np.argwhere(self.time_array_<=end_time)[-1,0] + 1

        if start > self.interval_[0] or end < self.interval_[-1]:
            # make sure the interval is not out of range
            start = max(start, 0)
            end = min(end, self.n_time)

            self.interval_ = np.arange(start, end, dtype=int)
            
            for subj in self.data_dict_:
                self.data_dict_[subj]['data'] = self.data_dict_[subj]['data'][:, self.interval]

            self.data_ = None

    def normalize(self):
        # normalization
        if not self.normalized:
            for subj_id in self.data_dict:
                new_time_series = self.data_dict[subj_id]['data']
                for n in range(new_time_series.shape[0]):
                    new_time_series[n, :] = new_time_series[n, :] - np.mean(new_time_series[n, :])
                    new_time_series[n, :] = np.divide(new_time_series[n, :], np.std(new_time_series[n, :]))
                self.data_dict_[subj_id]['data'] = new_time_series

            self.normalized = True
            self.data_ = None

    def spatial_downsample(self, num_select_nodes, rand_node_slct=False):
        if num_select_nodes < self.n_regions:
            if rand_node_slct:
                np.random.seed(0)
                nodes_idx = np.random.choice(range(self.n_regions), size=num_select_nodes, replace=False)
                nodes_idx.sort()
            else:
                # nodes_idx = np.array(list(range(self.num_select_nodes)))
                nodes_idx = np.arange(0, self.n_regions, self.n_regions/num_select_nodes, dtype=int)
            self.select_nodes(nodes_idx=nodes_idx)

            self.data_ = None

    def Fs_resample(self, Fs_ratio=None):
        # downsample frequency
        if Fs_ratio != 1 and self.Fs_ratio_ == 1:
            for subj_id in self.data_dict:
                new_time_series = self.data_dict[subj_id]['data']
                downsampled_time_series = np.zeros((new_time_series.shape[0], int(new_time_series.shape[1]*Fs_ratio)))
                for n in range(new_time_series.shape[0]):
                    downsampled_time_series[n, :] =  signal.resample(new_time_series[n, :], int(new_time_series.shape[1]*Fs_ratio))
                self.data_dict_[subj_id]['data'] = downsampled_time_series
            self.Fs_ratio_ = Fs_ratio

            start_time = self.time_array_[self.interval_[0]]
            end_time = self.time_array_[self.interval_[-1]]

            _, self.time_array_ =  signal.resample(new_time_series[n, :], int(new_time_series.shape[1]*Fs_ratio), t=self.time_array_)

            start = np.argwhere(self.time_array_>=start_time)[0,0]
            end = np.argwhere(self.time_array_<=end_time)[-1,0] + 1
            self.interval_ = np.arange(start, end, dtype=int)

            self.data_ = None

    def add_noise(self, noise_ratio, mean_noise=0):
        # adding noise perturbation 
        if noise_ratio > 0 and self.noise_ratio == 0 :
            for subj_id in self.data_dict:
                new_time_series = self.data_dict[subj_id]['data']
                power_signal = np.mean(new_time_series ** 2)
                power_noise = power_signal * noise_ratio
                new_time_series += np.random.normal(mean_noise, np.sqrt(power_noise), (new_time_series.shape[0], new_time_series.shape[1]))
                self.data_dict_[subj_id]['data'] = new_time_series

            self.noise_ratio = noise_ratio
            self.data_ = None

    def select_subjs(self, num_subj):
        # selects the first num_subj subjects in self.subj_id_lst

        SUBJECTS = [subj_id for subj_id in self.data_dict_]
        if num_subj < len(SUBJECTS):
            for subj_id in SUBJECTS:
                if not subj_id in SUBJECTS[:num_subj]:
                    self.data_dict_.pop(subj_id, None)

            self.data_ = None

    def select_nodes(self, nodes_idx=None):
        # select the nodes indexed by numbers in nodes_idx. nodes_idx is a numpy 1D array
        # if nodes_idx is None -> all the nodes will be considered (resets node selection)
        # if nodes_idx is not sorted, it can be used to reorder the nodes
        # this function can be used only once (you cannot select the nodes again)

        if nodes_idx is None:
            self.nodes_selection_ = np.arange(0, self.n_regions_, 1, dtype=int)
        else:
            self.nodes_selection_ = nodes_idx  

        for subj in self.data_dict_:
            self.data_dict_[subj]['data'] = self.data_dict_[subj]['data'][self.nodes_lst, :]

        self.data_ = None

        

    def visualize(self, start_time=None, end_time=None, \
        nodes_lst=None, \
        save_image=False, output_root=None):

        start = 0
        end = self.n_time

        if not start_time is None:
            start = np.argwhere(self.time>=start_time)[0,0]

        if not end_time is None:
            end = np.argwhere(self.time<=end_time)[-1,0] + 1
        
        interval = list(range(start, end))

        if nodes_lst is None:
            nodes_lst = self.nodes_lst
        else:
            nodes_lst = np.array(nodes_lst)[:, np.newaxis]

        plt.figure(figsize=(15, 5))
        plt.plot(self.time[interval], self.data[nodes_lst, interval].T)
        plt.xlabel('time (sec)')
        plt.title(self.TS_name_ + ' ' + self.session_name_ )
        if save_image:
            folder = output_root[:output_root.rfind('/')]
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(output_root+'.png', \
                dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad \
            ) 
            plt.close()
        else:
            plt.show()


################################# dFCM class ######################################

"""
Parameters
    ----------
    TR_array : an array labeling 
        timepoints by their TRs

Variables
    ----------
    FCSs : Functional Connecitivity 
        States patterns
    FCS_idx : the  index of the 
        FCS that corresponds to each 
        timepoint
    

todo:
- 
"""

class DFCM():
    def __init__(self, measure=None):

        # assert not measure is None, \
        #     "measure arg must be provided."
        self.measure_ = measure
        self.FCSs_ = None # is a dict
        self.FCS_idx_ = None # is a dict
        # info of the time series used for dFC estimation
        self.TS_info_ = None
        self.TR_array_ = None
        self.n_regions_ = None
        self.n_time_ = -1
    
    @classmethod
    def from_numpy(cls, array=None):
        pass

    @property
    def measure(self):
        return self.measure_

    @property
    def TR_array(self):
        return self.TR_array_.astype(int)

    @property
    def TR_keys(self):
        TRs_lst = list()
        for TR in self.TR_array:
            TRs_lst.append('TR'+str(TR))
        return TRs_lst

    @property
    def n_regions(self):
        return self.n_regions_

    @property
    def n_time(self):
        return self.n_time_

    # test this
    @property
    def FCSs(self):
        return self.FCSs_

    # test this
    @property
    def FCS_idx(self):
        return self.FCS_idx_

    # test this
    @property
    def FCS_idx_array(self):
        return np.array([int(self.FCS_idx[TR][self.FCS_idx[TR].find('S')+1:])-1 for TR in self.FCS_idx])

    @property
    def TS_info(self):
        # info of the time series used for dFC estimation
        return self.TS_info_

    
    # test
    def state_TC(self, TRs=None, \
        state_match=False, state_match_dict=None \
        ):
        # returns a np array of state indices over TRs in TRs

        if TRs is None:
            TRs = self.TR_array

        if not type(TRs[0]) is str:
            TRs_lst = list()
            for TR in TRs:
                TRs_lst.append('TR'+str(TR))
        else:
            TRs_lst = TRs

        state_TC = list()
        for key in self.FCS_idx:
            if key in TRs_lst:
                state = self.FCS_idx[key]
                if state_match:
                    match = state_match_dict['FCS_match'][state]['match']
                    state_TC.append(int(match[match.find('FCS')+3:]))
                else:
                    state_TC.append(int(state[state.find('FCS')+3:]))

        state_TC = np.array(state_TC)
        return state_TC
    
    # test
    def state_act_dict(self, TRs=None):
        # returns a dict including each FCS and its activation times
        # the TRs arg can be used to set a common set of TRs

        if TRs is None:
            TRs = self.TR_array

        TRs_lst = list()
        for TR in TRs:
            TRs_lst.append('TR'+str(TR))

        state_act_dict = {}
        state_act_dict['state_TC'] = {}
        state_act_dict['TR_array'] = TRs
        for FCS_key in self.FCSs:
            state_act_dict['state_TC'][FCS_key] = {}
            state_act_dict['state_TC'][FCS_key]['FCS'] = self.FCSs[FCS_key]
            state_act_dict['state_TC'][FCS_key]['act_TC'] = np.zeros((len(TRs),))
        t=0
        for TR in self.FCS_idx:
            if TR in TRs_lst:
                state_act_dict['state_TC'][self.FCS_idx[TR]]['act_TC'][t] = 1
                t=t+1
        assert t==len(TRs), 'error!'

        return state_act_dict

    # test
    def dFC2dict(self, TRs=None):
        # return dFC samples as a dictionary
        if TRs is None:
            TRs = self.TR_array
        if type(TRs) is list:
            TRs = np.array(TRs)
        TRs = TRs.astype(int)
        dFC_mat = self.get_dFC_mat(TRs=TRs)
        dFC_dict = {}
        for k, TR in enumerate(TRs):
            dFC_dict['TR'+str(TR)] = dFC_mat[k, :, :]
        return dFC_dict

    # test this
    def get_dFC_mat(self, TRs=None, num_samples=None):
        '''
        get dFC matrices corresponding to 
        the specified TRs 
        TRs should be list/ndarray not necessarily in order ?
        if num_samples specified, it will downsample 
        TRs to reach that number of samples and will also
        return picked TRs
        if num_samples > len(TRs) -> picks all TRs
        '''

        if TRs is None:
            TRs = self.TR_array

        if type(TRs) is np.int32 or type(TRs) is np.int64 or type(TRs) is int:
            TRs = [TRs]

        if not num_samples is None:
            if num_samples < len(TRs):
                TRs = TRs[np.linspace(0, len(TRs), num_samples, endpoint=False, dtype=int)]

        dFC_mat = list()
        for TR in TRs:
            dFC_mat.append(self.FCSs[self.FCS_idx['TR'+str(TR)]])

        dFC_mat = np.array(dFC_mat)

        if num_samples is None:
            return dFC_mat
        else:
            return dFC_mat, TRs

    def SWed_dFC_mat(self, W=None, n_overlap=None, tapered_window=False):
        '''
        the time samples will be picked after 
        averaging over a window which slides
        W is in sec
        '''
        dFC_mat = self.get_dFC_mat()

        # method not applicable to SW-based methods
        if 'sw_method' in self.measure.info:
            return dFC_mat

        dFC_mat_new = SW_downsample(data=dFC_mat, \
            Fs=self.TS_info['Fs'], W=W, n_overlap=n_overlap, tapered_window=tapered_window \
        )
        
        return dFC_mat_new


    def set_dFC(self, FCSs, FCS_idx=None, TS_info=None, TR_array=None):
        
        if len(FCSs.shape)==2:
            FCSs = np.expand_dims(FCSs, axis=0)

        if FCS_idx is None:
            FCS_idx = np.arange(start=0, stop=FCSs.shape[0], step=1, dtype=int)

        if type(FCS_idx) is list:
            FCS_idx = np.array(FCS_idx)

        if len(FCS_idx.shape)>1:
            FCS_idx = np.squeeze(FCS_idx)
        
        assert FCSs.shape[1] == FCSs.shape[2], \
                "FC matrices must be square."

        assert self.n_time==-1, \
            'why n_time is not -1 ?'
        
        if TR_array is None:
            TR_array = np.arange(start=self.n_time+1, stop=self.n_time+len(FCS_idx)+1, step=1, dtype=int)

        assert np.sum(np.abs(np.sort(TR_array)-TR_array))==0.0, \
            'TRs not sorted !'

        # the input FCS_idx is ranged from 0 to len(FCS)-1 but we shift it to 1 to len(FCS)
        self.FCSs_ = {}
        for i, FCS in enumerate(FCSs):
            self.FCSs_['FCS'+str(i+1)] = FCS

        self.FCS_idx_ = {}
        for i, idx in enumerate(FCS_idx):
            self.FCS_idx_['TR'+str(TR_array[i])] = 'FCS'+str(idx+1)

        self.TS_info_ = TS_info
        self.n_regions_ = FCSs.shape[1]
        self.n_time_ = len(self.FCS_idx_)
        self.TR_array_ = TR_array


    def visualize_dFC(self, TRs=None, normalize=True, \
        threshold=0.0, save_image=False, fig_name=None, fix_lim=True):

        assert not self.measure is None, \
            'Measure is not provided.'

        if TRs is None:
            TRs = self.TR_array

        if normalize:
            C = dFC_mat_normalize(C_t=self.get_dFC_mat(TRs=TRs), \
                global_normalization=True, threshold=threshold)
        else:
            C = self.get_dFC_mat(TRs=TRs)

        dFC_dict = {}
        for i, TR in enumerate(TRs):
            dFC_dict['TR'+str(TR)] = C[i]

        visualize_conn_mat(data=dFC_dict, \
            title=self.measure.measure_name+' dFC', \
            save_image=save_image, \
            output_root=fig_name, \
            fix_lim=fix_lim \
        )


################################# DATA LOADER class ######################################

"""
Parameters
    ----------
    DATA_type : ICA , Gordon , sample , 
        or simulated 

Variables
    ----------
    Var1 : Functional Connecitivity 
        States patterns
    Var2 : the  index of the 
        FCS that corresponds to each 
        timepoint
    

todo:
- add SUBJECTS of sample and simulation DATA_type
- factors are implemented only for gordon data type
"""

class DATA_LOADER():
    def __init__(self, **params):

        self.params = params

        if self.params['DATA_type']=='ICA': 
            self.BOLD_Fs_ = 1/0.72
            self.SUBJECTS = self.find_subj_list()

        if self.params['DATA_type']=='Gordon': 
            self.BOLD_Fs_ = 1/0.72
            self.SUBJECTS = self.find_subj_list()

        if self.params['DATA_type']=='sample': 
            self.BOLD_Fs_ = 1/0.5

        if self.params['DATA_type']=='simulated': 
            self.BOLD_Fs_ = 1/0.5

    
    @property
    def BOLD_Fs(self):
        return self.BOLD_Fs_

    def load(self, subj_id2load=None):

        BOLD = {}
        if self.params['DATA_type']=='ICA': 
            BOLD = self.load_ica(subj_id2load)

        if self.params['DATA_type']=='Gordon': 
            BOLD = self.load_gordon(subj_id2load)

        if self.params['DATA_type']=='sample': 
            BOLD = self.load_sample(subj_id2load)

        if self.params['DATA_type']=='simulated': 
            BOLD = self.load_simulated(subj_id2load)

        return BOLD
    
    def find_subj_list(self):
        '''
        if any of data_root_gordon or data_root_ica
        is None, it will be excluded.
        '''

        # ICA
        SUBJECTS_ica = None
        if not self.params['data_root_ica'] is None:
            ALL_RECORDS = os.listdir(self.params['data_root_ica'])
            ALL_RECORDS = [i for i in ALL_RECORDS if '.txt' in i]
            ALL_RECORDS.sort()
            SUBJECTS_ica = list()
            for s in ALL_RECORDS:
                num = s[:s.find('.')]
                SUBJECTS_ica.append(num)
            SUBJECTS_ica = list(set(SUBJECTS_ica))
            SUBJECTS_ica.sort()

        # GORDON
        SUBJECTS_gordon = None
        if not self.params['data_root_gordon'] is None:
            ALL_RECORDS = os.listdir(self.params['data_root_gordon'])
            ALL_RECORDS = [i for i in ALL_RECORDS if 'Rest' in i]
            ALL_RECORDS.sort()
            SUBJECTS_gordon = list()
            for s in ALL_RECORDS:
                num = s[:s.find('_')]
                SUBJECTS_gordon.append(num)
            SUBJECTS_gordon = list(set(SUBJECTS_gordon))
            SUBJECTS_gordon.sort()

        SUBJECTS = list()
        if SUBJECTS_gordon is None:
            SUBJECTS = SUBJECTS_ica
        if SUBJECTS_ica is None:
            SUBJECTS = SUBJECTS_gordon
        if (not SUBJECTS_gordon is None) and (not SUBJECTS_ica is None):
            SUBJECTS = intersection(SUBJECTS_gordon, SUBJECTS_ica)

        print( str(len(SUBJECTS)) + ' subjects were found. ')

        # print( str(len(SUBJECTS)) + ' subjects were found. ' + str(self.params['num_subj']) + ' subjects were selected.')

        # SUBJECTS = SUBJECTS[0:self.params['num_subj']]

        return SUBJECTS

    def load_gordon(self, subj_id2load=None):

        SESSIONs = self.params['SESSIONs'] #['Rest1_LR' , 'Rest1_RL', 'Rest2_LR', 'Rest2_RL']
        if subj_id2load is None:
            SUBJECTS = self.SUBJECTS
        else:
            SUBJECTS = [subj_id2load]

        # LOAD Region Location DATA

        locs = sio.loadmat(self.params['data_root_gordon']+'Gordon333_LOCS.mat')
        locs = locs['locs']

        # LOAD Region Data

        file = self.params['data_root_gordon']+'Gordon333_Key.txt'
        f = open(file, 'r')

        atlas_data = []
        for line in f:
            row = line.split()
            atlas_data.append(row)

        # apply networks2include
        nodes2include = [i-1 for i, x in enumerate(atlas_data) if x[3] in self.params['networks2include']]
        locs = locs[nodes2include, :]
        atlas_data = [x for node, x in enumerate(atlas_data) if node-1 in nodes2include]

        BOLD = {}
        for session in SESSIONs:
            BOLD[session] = None
            for subject in SUBJECTS:

                subj_fldr = subject + '_' + session

                # LOAD BOLD Data

                DATA = hdf5storage.loadmat(self.params['data_root_gordon']+subj_fldr+'/ROI_data_Gordon_333_surf.mat')
                time_series = DATA['ROI_data']

                # change time_series.shape to (nodes, time)
                time_series = time_series.T

                # apply networks2include
                time_series = time_series[nodes2include, :]

                if BOLD[session] is None:
                    BOLD[session] = TIME_SERIES(data=time_series, subj_id=subject, \
                                        Fs=self.BOLD_Fs, \
                                        locs=locs, nodes_info=atlas_data, \
                                        TS_name='BOLD Real', session_name=session \
                                    )
                else:
                    BOLD[session].append_ts(new_time_series=time_series, subj_id=subject)

            print( '*** Session ' + session + ': ' )
            print( 'number of regions= '+str(BOLD[session].n_regions) + ', number of TRs= ' + str(BOLD[session].n_time) )

        return BOLD

    def load_ica(self, subj_id2load=None):

        SESSIONs = self.params['SESSIONs'] #['session_1']
        if subj_id2load is None:
            SUBJECTS = self.SUBJECTS
        else:
            SUBJECTS = [subj_id2load]

        BOLD = {}
        for session in SESSIONs:
            BOLD[session] = None
            for subject in SUBJECTS:
                time_series = np.loadtxt( \
                    self.params['data_root_ica'] + subject + '.txt', dtype='float64' \
                    )
                time_series = time_series.T
                
                # time_series = time_series - np.repeat(np.mean(time_series, axis=1)[:,None], time_series.shape[1], axis=1) # ???????????????????????

                if BOLD[session] is None:
                    BOLD[session] = TIME_SERIES(data=time_series, subj_id=subject, Fs=self.BOLD_Fs, TS_name='BOLD ICA', session_name=session)
                else:
                    BOLD[session].append_ts(new_time_series=time_series, subj_id=subject)

            print(BOLD[session].n_regions, BOLD[session].n_time)

        return BOLD

    def load_sample(self, subj_id2load=None):

        ###### BOLD DATA ######
        time_BOLD = np.load(self.params['data_root_sample']+'bold_time.npy')/1e3    
        time_series = np.load(self.params['data_root_sample']+'bold_data.npy')

        time_series = time_series.T

        BOLD = None
        for subject in range(5):
            if BOLD is None:
                BOLD = TIME_SERIES( \
                    data=time_series[:, (subject)*1200:(subject+1)*1200], \
                    subj_id=str(subject+1), Fs=self.BOLD_Fs, \
                    time_array=time_BOLD[(subject)*1200:(subject+1)*1200], \
                    TS_name='BOLD Sample' \
                )
            else:
                BOLD.append_ts( \
                    new_time_series=time_series[:, (subject)*1200:(subject+1)*1200], \
                    time_array=time_BOLD[(subject)*1200:(subject+1)*1200],
                    subj_id=str(subject+1) \
                )

        print(BOLD.n_regions, BOLD.n_time)

        return BOLD

    def load_simulated(self, subj_id2load=None):

        ################################# Load Simulated BOLD data #################################

        time_BOLD = np.load(self.params['data_root_simul']+'bold_time.npy')/1e3    
        time_series_BOLD = np.load(self.params['data_root_simul']+'bold_data.npy')

        BOLD = TIME_SERIES(data=time_series_BOLD.T, subj_id='1', Fs=self.BOLD_Fs, time_array=time_BOLD, TS_name='BOLD Simulation')

        # ################################# Load Simulated Tavg data #################################

        # time_Tavg = np.load(self.params['data_root_simul']+'TVB data/tavg_time.npy')/1e3    
        # time_series_Tavg = np.load(self.params['data_root_simul']+'TVB data/tavg_data.npy')

        # TAVG = TIME_SERIES(data=time_series_Tavg.T, subj_id='1', Fs=200, time_array=time_Tavg, TS_name='Tavg Simulation')

        return BOLD
