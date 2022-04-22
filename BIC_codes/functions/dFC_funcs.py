#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:34:49 2021

@author: mte
"""

import numpy as np
from scipy import signal
from copy import deepcopy
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import distance
from joblib import Parallel, delayed
import os
import hdf5storage
import scipy.io as sio
from sklearn.preprocessing import power_transform

# ########## bundled brain graph visualizer ##########

# import pandas as pd
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

# test
def get_subj_ts_dict(time_series_dict, subj_id):
    subj_ts_dict = {}
    for session in time_series_dict:
        subj_ts_dict[session] = time_series_dict[session].get_subj_ts(subj_id=subj_id)
    return subj_ts_dict

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

    return 0.5*((np.linalg.norm((x-np.mean(x))-(y-np.mean(y)))**2)/(np.linalg.norm(x-np.mean(x))**2+np.linalg.norm(y-np.mean(y))**2))

def calc_ECM(A):
    """
    calc_ECM: Computes Eigenvector Centrality Mapping (ECM) 
    of adjacency matrix A

    Input:

        A (np.array): adjacency matrix

    Output:

        centrality (np.array): ECM vector
    """
    G = nx.from_numpy_matrix(A) 
    G.remove_edges_from(nx.selfloop_edges(G))
    # G = G.to_undirected()
    centrality = nx.eigenvector_centrality(G, weight='weight')
    # centrality = nx.pagerank(G, alpha=0.85)

    centrality = [centrality[node] for node in centrality]

    return centrality

# test
def zip_name(name_lst):
    # zip measure names
    new_name_lst = list()
    for name in name_lst:
        if 'Clustering' in name:
            new_name = 'SWC' + name[name.rfind('_'):]
        if 'ContinuousHMM' in name:
            new_name = 'CHMM' + name[name.rfind('_'):]
        if 'Windowless' in name:
            new_name = 'WL' + name[name.rfind('_'):]
        if 'DiscreteHMM' in name:
            new_name = 'DHMM' + name[name.rfind('_'):]
        if 'Time-Freq' in name:
            new_name = 'TF' + name[name.rfind('_'):]
        if 'SlidingWindow' in name:
            new_name = 'SW' + name[name.rfind('_'):]
        new_name_lst.append(new_name)
    return new_name_lst

# test
# pear_corr problem
def unzip_name(name):
    # zip measure names
    flag=False
    if not '_' in name:
        name = name + '_'
        flag=True
    if 'SWC' in name:
        new_name = 'Clustering_pear_corr' + name[name.rfind('_'):]
    if 'CHMM' in name:
        new_name = 'ContinuousHMM' + name[name.rfind('_'):]
    if 'WL' in name:
        new_name = 'Windowless' + name[name.rfind('_'):]
    if 'DHMM' in name:
        new_name = 'DiscreteHMM_pear_corr' + name[name.rfind('_'):]
    if 'TF' in name:
        new_name = 'Time-Freq' + name[name.rfind('_'):]
    if 'SW_' in name:
        new_name = 'SlidingWindow_pear_corr' + name[name.rfind('_'):]
    if flag:
        new_name = new_name[:-1]
    return new_name

#test
def dFC_mat2vec(C_t):
    '''
    C_t must be an array of matrices or a single matrix
    '''
    if len(C_t.shape)==2:
        assert C_t.shape[0]==C_t.shape[1],\
            'C is not a square matrix'
        return C_t[np.triu_indices(C_t.shape[1])]

    F = list()
    for t in range(C_t.shape[0]):
        C = C_t[t, : , :]
        assert C.shape[0]==C.shape[1],\
            'C is not a square matrix'
        F.append(C[np.triu_indices(C_t.shape[1])])

    F = np.array(F)
    return F

#test
def dFC_vec2mat(F, N):
    C = list()
    iu = np.triu_indices(N)
    for i in range(F.shape[0]):
        K = np.zeros((N, N))
        K[iu] = F[i,:]
        K = K + np.multiply(K.T, 1-np.eye(N))
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
    for dFCM in dFCM_lst:
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
    else:
        plt.show()

    return

def visualize_conn_mat(data, title='', \
    name_lst_key=None, mat_key=None, \
    cmap='viridis',\
    normalize=False,\
    disp_diag=True,\
    save_image=False, output_root=None, \
        fix_lim=True, lim_val=1.0 \
    ):

    '''
    - name_lst_key can be a list of names or the key to list of names
    - data must be a dict of correlation/connectivity matrices
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

    if name_lst_key is None:
        fig_width = 25*(len(data)/10)
    else:
        fig_width = 35*(len(data)/10) + 4
    fig_height = 10

    fig, axs = plt.subplots(1, len(data), figsize=(fig_width, fig_height), \
        facecolor='w', edgecolor='k')

    if not type(axs) is np.ndarray:
        axs = np.array([axs])

    fig.suptitle(title) #, fontsize=20, size=20

    axs = axs.ravel()

    for i, key in enumerate(data):

        name_lst = None
        if not name_lst_key is None:
            if type(name_lst_key) is str:
                name_lst = data[key][name_lst_key]
            if type(name_lst_key) is list:
                name_lst = name_lst_key

        if mat_key is None:
            C = data[key]
        else:
            C = data[key][mat_key]

        # C = np.abs(C) # ?????? should we do this?

        if normalize:
            C = dFC_mat_normalize(C[None,:,:], global_normalization=False, threshold=0.0)[0]

        if not disp_diag:
            C = np.multiply(C, 1-np.eye(len(C)))
            C = C + np.mean(C.flatten()) * np.eye(len(C))

        if np.any(C<0): # ?????? should we do this?
            V_MIN = -1
            V_MAX = 1
        else: # ?????? should we do this?
            V_MIN = 0
            V_MAX = lim_val

        if not fix_lim:
            V_MAX = np.max(C)
            V_MIN = np.min(C)

        if name_lst is None:
            axs[i].set_axis_off()

        im = axs[i].imshow(C, interpolation='nearest', aspect='equal', cmap=cmap,    # 'viridis' or 'jet'
            vmin=V_MIN, vmax=V_MAX)
        
        if not name_lst is None:
            axs[i].set_xticks(np.arange(len(name_lst)))
            axs[i].set_yticks(np.arange(len(name_lst)))
            axs[i].set_xticklabels(name_lst, rotation=90, fontsize=9)
            axs[i].set_yticklabels(name_lst, fontsize=9)
        axs[i].set_title(key)

    fig.subplots_adjust(
        bottom=0.1, \
        top=1.5, \
        left=0.1, \
        right=0.9,
        # wspace=0.02, \
        # hspace=0.02\
    )

    if not name_lst is None:
        fig.subplots_adjust(
            wspace=0.85 
        )
        
    if name_lst is None:
        cb_ax = fig.add_axes([0.91, 0.75, 0.007, 0.1])
    else:
        cb_ax = fig.add_axes([0.91, 0.75, 0.02, 0.1])
    cbar = fig.colorbar(im, cax=cb_ax, shrink=0.8) # shrink=0.8??

    # # set the colorbar ticks and tick labels
    # cbar.set_ticks(np.arange(0, 1.1, 0.5))
    # cbar.set_ticklabels(['0', '0.5', '1'])

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

'''
########## bundled brain graph visualizer ##########

cvsopts = dict(plot_height=400, plot_width=400)

def thresh_G(G, threshold=None):
    
    G_copy = deepcopy(G)
    
    if threshold==None:
        sig_edges = find_sig_edges(G_copy, min_num_edge=0)
        threshold = G.edges()[sig_edges[-1]]['weight']
    else:
        if threshold > 1:
            labels = [d["weight"] for (u, v, d) in G_copy.edges(data=True)]
            labels.sort()
            threshold = labels[-1*threshold]
            # sig_edges = find_sig_edges(G_copy, min_num_edge=threshold)
            # threshold = G.edges()[sig_edges[-1]]['weight']
    
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
        pos[key] = locs[i][0]
        
    nx.set_node_attributes(G_copy, pos, "pos") 
      
    
    return G_copy 

def visulize_brain_graph(FCS, nodes_info, locs, num_edges2show):
    G = batch_Adj2Net(FCS=FCS, nodes_info=nodes_info, is_digraph=False)
    G = set_locs_G(G, locs=locs)   
    plots = [nx_plot(ng(G, name="dFC"), view_degree=0, threshold=num_edges2show) ]
    
    return plots[0][0]


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

import time

class DFC_ANALYZER:
    # if self.n_jobs is None => no parallelization

    def __init__(self, analysis_name='', **params):

        self.analysis_name = analysis_name
        
        self.params = params
        if not 'vis_TR_idx' in self.params:
            self.params['vis_TR_idx'] = None
        if not 'save_image' in self.params:
            self.params['save_image'] = False
        if not 'output_root' in self.params:
            self.params['output_root'] = None
        if not 'n_jobs' in self.params:
            self.params['n_jobs'] = -1
        if not 'verbose' in self.params:
            self.params['verbose'] = 1
        if not 'backend' in self.params:
            self.params['backend'] = 'loky'

        self.MEASURES_lst_ = None
        self.MEASURES_fit_lst_ = {}

        # self.sim_assess_params = {}
        # if 'sim_assess_params' in params:
        #     self.sim_assess_params = params['sim_assess_params']

        # self.dyn_conn_det_params = {}
        # if 'dyn_conn_det_params' in params:
            # self.dyn_conn_det_params = params['dyn_conn_det_params']

        self.methods_assess_dict_lst_ = list()

    @property
    def methods_corr(self):
        # it assumes all subjects have all sessions and the order of their measures is the same ?
        
        methods_corr_dict = {}
        for session in self.methods_assess_dict_lst[0]:
            methods_corr_dict[session] = {}
            methods_corr = list()
            
            # dFC corr
            for methods_assess_subj_dict in self.methods_assess_dict_lst:
                methods_corr.append(methods_assess_subj_dict[session]['corr_mat'])
            
            methods_corr = np.array(methods_corr)
            methods_corr_dict[session]['corr_mat'] = np.mean(methods_corr, axis=0)
            methods_corr_dict[session]['measure_lst'] = methods_assess_subj_dict[session]['measure_lst']

        return methods_corr_dict

    @property
    def methods_assess_dict_lst(self):
        # methods_assess_dict_lst:
        # -> subject (list)
        #   -> session
        #       -> methods_corr
        #       -> measure_lst
        #       -> state_match
        #           -> [measure_i][measure_j][FCS_i]['trans_sim_vec'] 
        #       -> FO
        #           -> [measure][FCS]
        #       -> trans_freq
        #           -> [measure] -> trans_freq
        #           -> [measure] -> trans_norm

        return self.methods_assess_dict_lst_

    @property
    def MEASURES_lst(self):
        assert not self.MEASURES_lst_ is None, \
            'first set the MEASURES_lst!'
        return self.MEASURES_lst_

    # test
    @property
    def MEASURES_dict(self):
        dict = {}
        for measure in self.MEASURES_lst:
            assert not measure.measure_name in dict, \
                'duplicate measure name.'
            dict[measure.measure_name] = measure
        return dict

    @property
    def MEASURES_fit_lst(self):
        return self.MEASURES_fit_lst_

    # test
    @property
    def MEASURES_fit_dict(self):
        dict = {}
        for session in self.MEASURES_fit_lst:
            dict[session] = {}
            for measure in self.MEASURES_fit_lst[session]:
                assert not measure.measure_name in dict[session], \
                    'duplicate measure name.'
                dict[session][measure.measure_name] = measure
        return dict

    @property
    def FCS_dict(self):
        #FCS_dict[session][measure.measure_name]['state_TC'][FCS_i]['FCS']
        FCS_dict = {}
        for session in self.MEASURES_fit_lst:
            FCS_dict[session] = {}
            for measure in self.SB_MEASURES_lst(self.MEASURES_fit_lst[session]):
                FCS_dict[session][measure.measure_name] = measure.FCS_dict
        return FCS_dict

    @property
    def FCS_sim_dict(self):
        ## normalize ??
        normalize = False
        # FCS_sim_dict[session][measure_i][measure_j][FCS_i]
        FCS_sim_dict = {}
        for session in self.FCS_dict:
            FCS_sim_dict[session] = {}
            for measure_i in self.FCS_dict[session]:
                FCS_sim_dict[session][measure_i] = {}
                for measure_j in self.FCS_dict[session]:

                    FCS_sim_dict[session][measure_i][measure_j] = self.FCS_sim_calc( \
                        FCS_dict_i = self.FCS_dict[session][measure_i], \
                        FCS_dict_j = self.FCS_dict[session][measure_j], \
                        normalize=normalize \
                    )

        return FCS_sim_dict

    def set_MEASURES_lst(self, MEASURES_lst):
        self.MEASURES_lst_ = MEASURES_lst

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
                'ContinuousHMM', \
                'Windowless', \
                'Clustering', \
                'DiscreteHMM' \
                )
        '''

        # a list of MEASURES with default parameter values
        MEASURES_lst = self.create_measure_obj(MEASURES_name_lst=MEASURES_name_lst, **params_methods)

        # adding MEASURES with alternative parameter values
        for hyper_param in alter_hparams:
            params = deepcopy(params_methods)
            for value in alter_hparams[hyper_param]:
                params[hyper_param] = value
                new_MEASURES = self.create_measure_obj(MEASURES_name_lst=MEASURES_name_lst, **params)
                for new_measure in new_MEASURES:
                    flag=0
                    for MEASURE in MEASURES_lst:
                        if new_measure.issame(MEASURE):
                            flag=1
                    if flag==0:
                        MEASURES_lst.append(new_measure)

        return MEASURES_lst

    def create_measure_obj(self, MEASURES_name_lst, **params):

        MEASURES_lst = list()
        for MEASURES_name in MEASURES_name_lst:

            ###### CONTINUOUS HMM ######
            if MEASURES_name=='ContinuousHMM':
                measure = HMM_CONT(**params)

            ###### WINDOW_LESS ######
            if MEASURES_name=='Windowless':
                measure = WINDOWLESS(**params)

            ###### SLIDING WINDOW ######
            if MEASURES_name=='SlidingWindow':
                measure = SLIDING_WINDOW(sw_method='pear_corr', **params)

            ###### TIME FREQUENCY ######
            if MEASURES_name=='Time-Freq':
                measure = TIME_FREQ(TF_method='WTC', **params)

            ###### SLIDING WINDOW + CLUSTERING ######
            if MEASURES_name=='Clustering':
                measure = SLIDING_WINDOW_CLUSTR(base_method='pear_corr', **params)

            ###### DISCRETE HMM ######
            if MEASURES_name=='DiscreteHMM':
                measure = HMM_DISC(base_method='pear_corr', **params)

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

    ##################### MEASURE CHARACTERISTICS ######################

    def time_analyze(self, time_series=None):

        print('Total Time = estimate_FCS Time + estimate_dFCM Time')
        
        time_lst = list()
        estimate_FCS_time_lst = list()
        for measure in self.MEASURES_lst:

            tic = time.time()
            if measure.is_state_based:
                tic_FCS = time.time()
                measure.estimate_FCS(time_series=time_series)
                estimate_FCS_time_lst.append(time.time() - tic_FCS)
            else:
                estimate_FCS_time_lst.append(0.0)

            measure.estimate_dFCM(time_series=time_series)

            time_lst.append(time.time() - tic)

        for i, measure in enumerate(self.MEASURES_lst):
            if measure.is_state_based:
                print('Measure '+measure.measure_name+' time = ' + \
                    str(time_lst[i]) + ' = ' + \
                    str(estimate_FCS_time_lst[i]) + ' + ' + \
                    str(time_lst[i] - estimate_FCS_time_lst[i]) \
                    )
            else:
                print('Measure '+measure.measure_name+' time = ' + \
                    str(time_lst[i]) \
                    )

    def dFCM_var(self, MEASURES_dFCM):

        MEASURES_dFC_var = {}
        for measure in MEASURES_dFCM:
            dFC_mat = MEASURES_dFCM[measure].get_dFC_mat(TRs = MEASURES_dFCM[measure].TR_array)
            V = np.var(dFC_mat, axis=0)
            MEASURES_dFC_var[measure] = V
        return MEASURES_dFC_var

    ##################### POST ANALYSIS ######################

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

        SUBJ_s_output = {}
        
        SUBJECTs = common_subj_lst(time_series_dict) 

        if self.params['n_jobs'] is None:
            OUT = list()
            for subject in SUBJECTs:
                OUT.append( \
                    self.subj_lvl_dFC_assess( \
                    time_series_dict=get_subj_ts_dict(time_series_dict, subj_id=subject), \
                    ))
        else:
            OUT = Parallel( \
                        n_jobs=self.params['n_jobs'], \
                        verbose=self.params['verbose'], \
                        backend=self.params['backend'])( \
                    delayed(self.subj_lvl_dFC_assess)( \
                        time_series_dict=get_subj_ts_dict(time_series_dict, subj_id=subject), \
                        ) \
                        for subject in SUBJECTs)
        
        SUBJ_s_output['dFC_assess'] = [out['dFC_corr_assess_dict'] for out in OUT]

        return SUBJ_s_output

    def subj_lvl_dFC_assess(self, time_series_dict):

        # time_series_dict is a dict of time_series

        SUBJ_output = {}

        dFCM_dict = {}
        dFC_corr_assess_dict = {}
        for session in time_series_dict:
            time_series = time_series_dict[session]
            if self.params['n_jobs'] is None:
                dFCM_lst = list()
                for measure in self.MEASURES_fit_lst_[session]:
                    dFCM_lst.append( \
                        measure.estimate_dFCM(time_series=time_series) \
                    )
            else:
                dFCM_lst = Parallel( \
                    n_jobs=self.params['n_jobs'], verbose=self.params['verbose'], backend=self.params['backend'])( \
                    delayed(measure.estimate_dFCM)(time_series=time_series) \
                        for measure in self.MEASURES_fit_lst_[session])

            MEASURES_dFCM = {}
            for dFCM in dFCM_lst:
                # test if self.MEASURES_lst[m].measure_name=dFCM.measure.measure_name
                MEASURES_dFCM[dFCM.measure.measure_name] = dFCM

            dFCM_dict[session] = MEASURES_dFCM

            # self.dFC_corr_assess returns a dict with 'corr_mat', 
            # 'measure_lst', 'sb_measure_lst', and 'state_match' keys
            dFC_corr_assess_dict[session] = self.dFC_corr_assess(dFCM_lst=dFCM_lst)

        # dFC_session_sim_dict = self.dFC_session_similarity(dFCM_dict)

        # SUBJ_output['dFC_session_sim_dict'] = dFC_session_sim_dict
        SUBJ_output['dFC_corr_assess_dict'] = dFC_corr_assess_dict

        return SUBJ_output


    ##################### dFC CHARACTERISTICS ######################

    def dFC_corr(self, dFCM_i, dFCM_j, TRs=None):

        # returns correlation of dFC measures over time

        if TRs is None:
            TRs = TR_intersection([dFCM_i, dFCM_j])
        dFC_mat_i = dFCM_i.get_dFC_mat(TRs=TRs)
        dFC_mat_j = dFCM_j.get_dFC_mat(TRs=TRs)
        corr = list()
        for t in range(len(TRs)):
            corr.append(np.corrcoef(dFC_mat_i[t,:,:].flatten(), dFC_mat_j[t,:,:].flatten())[0,1])
        corr= np.array(corr)
        return corr
    
    def FO_calc(self, dFCM, common_TRs=None):

        # returns, for each state the Fractional Occupancy (FO)
        # see Visaure et al., 2017
        # it only considers TRs in common_TRs

        if common_TRs is None:
            common_TRs = dFCM.TR_array

        state_act_dict = dFCM.state_act_dict(TRs=common_TRs)

        FO = {}
        for FCS_key in state_act_dict['state_TC']:
            FO[FCS_key] = np.mean(state_act_dict['state_TC'][FCS_key]['act_TC'])

        return FO

    def COM_calc(self, dFCM_lst, common_TRs=None, lag=0):
        # returns co-occurance (CO) with specified lag, a dict with:
        # CO['Obs_seq']
        # CO['FCS_name']
        # CO['COM']
        # automatically ignores DD methods

        # TODO:  DOWNSAMPLING problem and ignoring the between state transitions

        if common_TRs is None:
            common_TRs = TR_intersection(dFCM_lst)

        TRs_lst = list()
        for TR in common_TRs:
            TRs_lst.append('TR'+str(TR))

        # list of FCS names
        ############## when combining COMs check if the order of FCS names is the same
        FCS_name_lst = list()
        for dFCM in dFCM_lst:
            if dFCM.measure.is_state_based:
                for FCS in dFCM.FCSs:
                    FCS_name_lst.append(dFCM.measure.measure_name+'_'+FCS)
        # print(FCS_name_lst)

        # building the observation sequence
        Obs_seq = list()
        for TR in TRs_lst:
            Obs_vec = list()
            for dFCM in dFCM_lst:
                if dFCM.measure.is_state_based:
                    Obs_vec.append(dFCM.measure.measure_name + '_' + dFCM.FCS_idx[TR])
            Obs_seq.append(Obs_vec)
        # print(Obs_seq)

        # computing COM
        flag = 0
        CO = {}
        CO['Obs_seq'] = Obs_seq
        CO['FCS_name'] = FCS_name_lst
        CO['COM'] = np.zeros((len(FCS_name_lst), len(FCS_name_lst)))
        for i, TR in enumerate(TRs_lst):
            if i>=lag:
                for last_FCS in CO['Obs_seq'][i-lag]:
                    for current_FCS in CO['Obs_seq'][i]:
                        CO['COM'][CO['FCS_name'].index(last_FCS), CO['FCS_name'].index(current_FCS)] += 1

        return CO


    def transition_freq(self, dFCM, common_TRs=None):
        # returns the number of total state transition within common_TRs -> trans_freq
        # and the number of total state transitions regardless of common_TRs
        # but normalized by total number of TRs -> trans_norm

        if common_TRs is None:
            common_TRs = dFCM.TR_array

        TRs_lst = list()
        for TR in common_TRs:
            TRs_lst.append('TR'+str(TR))

        trans_freq_dict = {}

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

        return trans_freq_dict 

    def dFC_distance(self, FC_t_i, FC_t_j, metric, normalize=True):
        '''
        FC_t_i and FC_t_j must be an 
        array of FC matrices = (n_time, n_regions, n_regions)
        metric options: correlation, euclidean, ECM (Eigenvector Centrality Mapping)
        normalize option is for ECM and euclidean metrics since correlation is already 
        normalized.
        for ECM, the input can be an array of ECM_vecs and of shape (n_time, n_regions) 
        or array of FC matrices = (n_time, n_regions, n_regions)
        '''
        assert len(FC_t_i)==len(FC_t_j),\
            'the inputs must of the same number of samples'

        distance_out = list()
        for t in range(FC_t_i.shape[0]):

            if metric=='correlation' or metric=='euclidean':
                assert FC_t_i[t].shape[0]==FC_t_i[t].shape[1],\
                    'Matrices are not square'
                assert FC_t_j[t].shape[0]==FC_t_j[t].shape[1],\
                    'Matrices are not square'

            if metric=='correlation':
                FC_vec_i = dFC_mat2vec(FC_t_i[t])
                FC_vec_j = dFC_mat2vec(FC_t_j[t])
                distance_out.append(distance.correlation(FC_vec_i, FC_vec_j))

            if metric=='euclidean':
                FC_vec_i = dFC_mat2vec(FC_t_i[t])
                FC_vec_j = dFC_mat2vec(FC_t_j[t])
                if normalize:
                    distance_out.append(normalized_euc_dist(FC_vec_i, FC_vec_j))
                else:
                    distance_out.append(distance.euclidean(FC_vec_i, FC_vec_j))

            if metric=='ECM':
                if len(FC_t_i[t].shape)==2:
                    assert FC_t_i[t].shape[0]==FC_t_i[t].shape[1],\
                        'Matrices are not square'
                    assert FC_t_j[t].shape[0]==FC_t_j[t].shape[1],\
                        'Matrices are not square'
                    ECM_i = calc_ECM(np.abs(FC_t_i[t]))
                    ECM_j = calc_ECM(np.abs(FC_t_j[t]))
                else:
                    ECM_i = FC_t_i[t]
                    ECM_j = FC_t_j[t]
                if normalize:
                    distance_out.append(normalized_euc_dist(ECM_i, ECM_j))
                else:
                    distance_out.append(distance.euclidean(ECM_i, ECM_j))

        return np.array(distance_out)

    # #regression
        # y = dFC_vec_j[t]
        # xx = FCS_vecs_new_order
        # reg = LinearRegression().fit(xx.T, y.T)
        # reg_dist.append(reg.coef_)

    def dFCM_lst_distance(self, dFCM_lst, metric, normalize=True):

        TRs = TR_intersection(dFCM_lst)
        
        distance_mat = np.zeros((len(TRs), len(dFCM_lst), len(dFCM_lst)))
        for i, dFCM_i in enumerate(dFCM_lst):
            for j, dFCM_j in enumerate(dFCM_lst):
                dFC_mat_i = dFCM_i.get_dFC_mat(TRs=TRs)
                dFC_mat_j = dFCM_j.get_dFC_mat(TRs=TRs)
                distance_mat[:, i, j] = self.dFC_distance(\
                    FC_t_i=dFC_mat_i, \
                    FC_t_j=dFC_mat_j, \
                    metric=metric, \
                    normalize=normalize\
                        )
        return distance_mat

    def dFCM_lst_var(self, dFCM_lst, metric, normalize=True):

        TRs = TR_intersection(dFCM_lst)

        dFC_mat_avg = None
        for i, dFCM_i in enumerate(dFCM_lst):
            dFC_mat_i = dFCM_i.get_dFC_mat(TRs=TRs)
            if dFC_mat_avg is None:
                dFC_mat_avg = dFC_mat_normalize(C_t=dFC_mat_i, global_normalization=True) 
            else:
                dFC_mat_avg += dFC_mat_normalize(C_t=dFC_mat_i, global_normalization=True) 
        dFC_mat_avg = np.divide(dFC_mat_avg, len(dFCM_lst))

        distance_var_mat = np.zeros((len(TRs), len(dFCM_lst)))
        for i, dFCM_i in enumerate(dFCM_lst):
            dFC_mat_i = dFCM_i.get_dFC_mat(TRs=TRs)
            distance_var_mat[:, i] = self.dFC_distance(\
                FC_t_i=dFC_mat_i, \
                FC_t_j=dFC_mat_avg, \
                metric=metric, \
                normalize=normalize\
                    )
        return distance_var_mat

    def dFC_corr_assess(self, dFCM_lst):

        ########## dFCM corr ##########
        # returns averaged correlation of dFC measures in a dict format
        # with 'corr_mat' and 'measure_lst' keys

        a = 0.1 # portion of the dFCs to ignore from the beginning and the end

        measure_lst = list()
        common_TRs = dFCM_lst[0].TR_array
        for dFCM in dFCM_lst:
            measure_lst.append(dFCM.measure.measure_name)
            common_TRs = intersection(common_TRs, dFCM.TR_array)
        common_TRs.sort()
        assert len(common_TRs)!=0, \
            'No TR intersection.'

        methods_assess = {}
        corr_mat = np.zeros((len(measure_lst), len(measure_lst)))
        for i in range(len(measure_lst)):
            for j in range(i+1, len(measure_lst)):

                # assert dFCM_lst[i].measure.measure_name==self.MEASURES_lst[i].measure_name and \
                #     dFCM_lst[j].measure.measure_name==self.MEASURES_lst[j].measure_name, \
                #     'mismatch in MEASURES_lst order'

                corr_ij = self.dFC_corr( \
                    dFCM_lst[i], dFCM_lst[j], \
                    TRs=common_TRs \
                        )
                corr_mat[i,j] = np.mean(corr_ij[ \
                    int(len(corr_ij)*a) : int(len(corr_ij)*(1-a)) \
                        ])
                corr_mat[j,i] = corr_mat[i,j] 

        ########## distance calc ##########

        dFC_distance = {}
        dFC_distance['euclidean'] = self.dFCM_lst_distance(\
            dFCM_lst, \
            metric='euclidean', \
            normalize=True \
            )
        dFC_distance['correlation'] = self.dFCM_lst_distance(\
            dFCM_lst, \
            metric='correlation', \
            normalize=True \
            )
        dFC_distance['ECM'] = self.dFCM_lst_distance(\
            dFCM_lst, \
            metric='ECM', \
            normalize=True \
            )

        ########## distance var calc ##########

        dFC_distance_var = {}
        dFC_distance_var['euclidean'] = self.dFCM_lst_var(\
            dFCM_lst, \
            metric='euclidean', \
            normalize=True \
            )
        dFC_distance_var['correlation'] = self.dFCM_lst_var(\
            dFCM_lst, \
            metric='correlation', \
            normalize=True \
            )
        dFC_distance_var['ECM'] = self.dFCM_lst_var(\
            dFCM_lst, \
            metric='ECM', \
            normalize=True \
            )

        ########## state coactivation corr ##########

        sb_dFCM_dict = {}
        for dFCM in dFCM_lst:
            if dFCM.measure.is_state_based:
                sb_dFCM_dict[dFCM.measure.measure_name] = dFCM

        # ## using the same common_TRs as dFC corr

        # state_match = {}
        # for sb_measure_i in sb_dFCM_dict:
        #     state_match[sb_measure_i] = {}
        #     for sb_measure_j in sb_dFCM_dict:

        #         state_match[sb_measure_i][sb_measure_j] = self.dFCM_trans_sim( \
        #             dFCM_i=sb_dFCM_dict[sb_measure_i], \
        #             dFCM_j=sb_dFCM_dict[sb_measure_j], \
        #             common_TRs=common_TRs \
        #         )

        ########## Fractional Occupancy ##########

        FO = {}
        for sb_measure in sb_dFCM_dict:
            FO[sb_measure] = self.FO_calc(
                dFCM=sb_dFCM_dict[sb_measure], \
                common_TRs=common_TRs \
            )

        ########## Co-Occurance Matrix and Transition Probability Matrix ##########

        CO = self.COM_calc(dFCM_lst, \
            common_TRs=common_TRs, \
            lag=0 \
            )

        TP = self.COM_calc(dFCM_lst, \
            common_TRs=common_TRs, \
            lag=1 \
            )

        ########## transition frequency ##########

        trans_freq = {}
        for sb_measure in sb_dFCM_dict:
            trans_freq[sb_measure] = self.transition_freq(
                dFCM=sb_dFCM_dict[sb_measure], \
                common_TRs=common_TRs \
            )
                
        ##############################################

        methods_assess['corr_mat'] = corr_mat
        methods_assess['dFC_distance'] = dFC_distance
        methods_assess['dFC_distance_var'] = dFC_distance_var
        methods_assess['measure_lst'] = measure_lst
        # methods_assess['state_match'] = state_match
        methods_assess['FO'] = FO
        methods_assess['CO'] = CO
        methods_assess['TP'] = TP
        methods_assess['trans_freq'] = trans_freq

        return methods_assess

    def visualize_dFCMs(self, dFCM_lst=None, TR_idx=None, normalize=True, threshold=0.0, \
                            fix_lim=True, subj_id=''):
        
        # TR_idx is not TR values, but their indices!

        TRs = TR_intersection(dFCM_lst)
        if not TR_idx is None:
            assert not np.any(np.array(TR_idx)>=len(TRs)), \
                'TR_idx out of range.'
            TRs = [TRs[i] for i in TR_idx]

        for dFCM in dFCM_lst:
            if self.params['save_image']:
                output_root = self.params['output_root']+'dFC/'
                dFCM.visualize_dFC(TRs=TRs, normalize=normalize, threshold=threshold, \
                    fix_lim=fix_lim, \
                    save_image=self.params['save_image'], \
                    fig_name= output_root+'subject'+subj_id+'_'+dFCM.measure.measure_name+'_dFC')
            else:
                dFCM.visualize_dFC(TRs=TRs, normalize=normalize, threshold=threshold, fix_lim=fix_lim)

    def visualize_FCS(self, normalize=True, threshold=0.0):

        for session in self.MEASURES_fit_lst:
            for measure in self.MEASURES_fit_lst[session]:  
                if self.params['save_image']:
                    output_root = self.params['output_root'] + 'FCS/'
                    measure.visualize_FCS(normalize=normalize, threshold=threshold, save_image=True, \
                        fig_name= output_root + measure.measure_name+'_FCS_'+session)
                    # measure.visualize_TPM(normalize=normalize)
                else:
                    measure.visualize_FCS(normalize=normalize, threshold=threshold) # normalize?
                    # measure.visualize_TPM(normalize=normalize)

################################# dFC class ####################################

"""

todo:
- separate the matrix visualizing function
- brain or brain graph class
- add an updating behavior -> we can segment subjects and time_series and update the model gradually ?
- type annotation
- remove sliding window type dFC visualization 
- normalization: C_t_z[:, i, i] = np.mean(C_t_z) # ?????????????????
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

    base_methods_name_lst = sw_methods_name_lst + TF_methods_name_lst

    def __init__(self):
        self.measure_name = ''
        self.is_state_based = bool()
        self._stat = []
        self.TPM = []
        self.params = {}

    @property
    def FCS(self):
        return self.FCS_

    # test
    @property
    def FCS_dict(self):
        # returns a dict inclusing each FCS to be fed to similarity assess

        if not self.is_state_based:
            return None

        C_A = self.FCS
        state_act_dict = {}
        state_act_dict['state_TC'] = {}
        for k in range(C_A.shape[0]):
            state_act_dict['state_TC']['FCS'+str(k+1)] = {}
            state_act_dict['state_TC']['FCS'+str(k+1)]['FCS'] = C_A[k,:,:]
            
        return state_act_dict

    @property
    def info(self):
        print_dict(self.params)

    def issame(self, dFC):
        pass

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
        new_time_series.truncate(start_point=0, end_point=self.params['num_time_point'])

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
        new_time_series.truncate(start_point=0, end_point=self.params['num_time_point'])

        return new_time_series
    
    def visualize_states(self):
        pass

    # todo : use FCS_dict func in this func
    def visualize_FCS(self, normalize=True, threshold=0.0, save_image=False, fig_name=None):
        
        if self.FCS == []:
            return

        if normalize:
            C = dFC_mat_normalize(C_t=self.FCS, threshold=threshold)
        else:
            C = self.FCS

        FCS_dict = {}
        for i in range(C.shape[0]):
            FCS_dict['FCS '+str(i+1)] = C[i]

        visualize_conn_mat(data=FCS_dict, \
            title=self.measure_name+' FCS', \
            save_image=save_image, \
            output_root=fig_name, \
            fix_lim=True \
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
        self.measure_name = 'ContinuousHMM'
        self.is_state_based = True
        self.TPM = []
        self.FCS_ = []

        self.params_name_lst = ['n_states', \
            'normalization', 'num_subj', 'num_select_nodes', 'num_time_point', \
            'Fs_ratio', 'noise_ratio', 'num_realization']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]

    def issame(self, dFC):
        if type(self)==type(dFC):
            for param_name in self.params:
                if self.params[param_name] != dFC.params[param_name]:
                    return False
        else:
            return False
        return True


    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        time_series = self.manipulate_time_series4FCS(time_series)

        Models, Scores = [], []
        for i in range(100):
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

        return self

    def estimate_dFCM(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        Z = self.hmm_model.predict(time_series.data.T)
        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=self.FCS_, FCS_idx=Z, TS_info=time_series.TS_info)

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
        self.measure_name = 'Windowless'
        self.is_state_based = True
        self.TPM = []
        self.FCS_ = []

        self.params_name_lst = ['n_states', \
            'normalization', 'num_subj', 'num_select_nodes', 'num_time_point', \
            'Fs_ratio', 'noise_ratio', 'num_realization']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
    
    def issame(self, dFC):
        if type(self)==type(dFC):
            for param_name in self.params:
                if self.params[param_name] != dFC.params[param_name]:
                    return False
        else:
            return False
        return True

    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        # time_series ~ gamma.dot(dictionary)
        self.aksvd = ApproximateKSVD(n_components=self.params['n_states'], transform_n_nonzero_coefs=1)
        self.dictionary = self.aksvd.fit(time_series.data.T).components_
        self.gamma = self.aksvd.transform(time_series.data.T)

        self.FCS_ = np.zeros([self.params['n_states'], time_series.n_regions, time_series.n_regions])
        for i in range(self.params['n_states']):
            self.FCS_[i, :, :] = np.multiply(np.expand_dims(self.dictionary[i,:], axis=0).T, np.expand_dims(self.dictionary[i,:], axis=0))

        self.Z = list()
        for i in range(time_series.n_time):
            self.Z.append(np.argwhere(self.gamma[i, :] != 0)[0,0])

        return self

    def estimate_dFCM(self, time_series):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        gamma = self.aksvd.transform(time_series.data.T)

        Z = list()
        for i in range(time_series.n_time):
            Z.append(np.argwhere(gamma[i, :] != 0)[0,0])
            
        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=self.FCS_, FCS_idx=Z, TS_info=time_series.TS_info)
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

    def __init__(self, TF_method='WTC', coi_correction=True, **params):
        
        assert TF_method in self.TF_methods_name_lst, \
            "Time-frequency method not recognized."

        self.measure_name_ = 'Time-Freq'
        self.is_state_based = False
        self.TPM = []
        self.FCS_ = []

        self.params_name_lst = ['TF_method', 'coi_correction', \
            'n_jobs', 'verbose', 'backend', \
            'normalization', 'num_subj', 'num_select_nodes', 'num_time_point', \
            'Fs_ratio', 'noise_ratio', 'num_realization']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
        
        self.params['TF_method'] = TF_method
        self.params['coi_correction'] = coi_correction

    @property
    def measure_name(self):
        return self.measure_name_ # + '_' + self.params['TF_method']

    def issame(self, dFC):
        if type(self)==type(dFC):
            for param_name in self.params:
                if self.params[param_name] != dFC.params[param_name]:
                    return False
        else:
            return False
        return True

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

        # params
        J = 50 # -1
        s0 = 1 # -1
        dj = 1/8 # 1/12

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

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

        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=WT, TS_info=time_series.TS_info)
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

from sklearn.covariance import GraphicalLassoCV

class SLIDING_WINDOW(dFC):

    def __init__(self, sw_method='pear_corr', tapered_window=True, **params):

        assert sw_method in self.sw_methods_name_lst, \
            "sw_method not recognized."

        self.measure_name_ = 'SlidingWindow'
        self.is_state_based = False
        self.sw_method_ = sw_method
        self.TPM = []
        self.FCS_ = []

        self.params_name_lst = ['sw_method', 'tapered_window', \
            'W', 'n_overlap', 'normalization', \
            'num_subj', 'num_select_nodes', 'num_time_point', 'Fs_ratio', \
            'noise_ratio', 'num_realization']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
        
        self.params['sw_method'] = sw_method
        self.params['tapered_window'] = tapered_window
        
    
    @property
    def measure_name(self):
        return self.measure_name_ #+ '_' + self.sw_method

    def issame(self, dFC):
        if type(self)==type(dFC):
            for param_name in self.params:
                if self.params[param_name] != dFC.params[param_name]:
                    return False
        else:
            return False
        return True

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
            TR_array.append(np.array( [ int((l + (l+W)) / 2) ] ))

        return FCSs, TR_array
    
    def estimate_dFCM(self, time_series):
        
        '''
        we assume calc is applied on subjects separately
        '''
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        # W is converted from sec to samples
        FCSs, TR_array = self.dFC(time_series=time_series.data, \
            W=int(self.params['W'] * time_series.Fs) , \
            n_overlap=self.params['n_overlap'], \
            tapered_window=self.params['tapered_window'] \
            )

        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.TS_info)

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

    def __init__(self, base_method='pear_corr', clstr_distance='euclidean',
    tapered_window=True, **params):

        assert clstr_distance=='euclidean' or clstr_distance=='manhattan', \
            "Clustering distance not recognized. It must be either \
                euclidean or manhattan."

        assert base_method in self.base_methods_name_lst, \
            "Base method not recognized."
    
        self.measure_name_ = 'Clustering'
        self.is_state_based = True
        self.TPM = []
        self.FCS_ = []

        self.params_name_lst = ['base_method', 'tapered_window', 'clstr_distance', \
            'coi_correction', \
            'n_subj_clstrs', 'W', 'n_overlap', 'n_states', 'normalization', \
            'n_jobs', 'verbose', 'backend', \
            'num_subj', 'num_select_nodes', 'num_time_point', 'Fs_ratio', \
            'noise_ratio', 'num_realization']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
        
        self.params['base_method'] = base_method
        self.params['clstr_distance'] = clstr_distance
        self.params['tapered_window'] = tapered_window

    @property
    def measure_name(self):
        return self.measure_name_ #+ '_' + self.base_method

    def issame(self, dFC):
        if type(self)==type(dFC):
            for param_name in self.params:
                if self.params[param_name] != dFC.params[param_name]:
                    return False
        else:
            return False
        return True

    def dFC_mat2vec(self, C_t):
        return dFC_mat2vec(C_t)

    def dFC_vec2mat(self, F, N):
        return dFC_vec2mat(F=F, N=N)

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
            Z = kmeans_.predict(F)
            F_cent = kmeans_.cluster_centers_

        FCS_ = self.dFC_vec2mat(F_cent, N=n_regions)
        return FCS_, kmeans_

        
    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        if self.params['base_method']=='CWT_mag' or self.params['base_method']=='CWT_phase_r' \
            or self.params['base_method']=='CWT_phase_a' or self.params['base_method']=='WTC':
            # params = {'n_jobs': self.params['n_jobs'], 'verbose': self.params['verbose'], 'backend': self.params['backend']}
            base_dFC = TIME_FREQ(TF_method=self.params['base_method'], **self.params)
        else:
            # params = {'W': self.W, 'n_overlap': self.n_overlap}
            base_dFC = SLIDING_WINDOW(sw_method=self.params['base_method'], \
                    tapered_window=self.params['tapered_window'], **self.params)

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
        for subject in SUBJECTs:
            
            dFCM_raw = base_dFC.estimate_dFCM( \
                time_series=time_series.get_subj_ts(subj_id=subject) \
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
            if FCS_1st_level is None:
                FCS_1st_level = FCS
            else:
                FCS_1st_level = np.concatenate((FCS_1st_level, FCS), axis=0)
        
        self.FCS_, self.kmeans_ = self.cluster_FC( \
            FCS_raw=FCS_1st_level, \
            n_clusters = self.params['n_states'], \
            n_regions = dFCM_raw.n_regions \
            )

        return self

    def estimate_dFCM(self, time_series):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        if self.params['base_method']=='CWT_mag' or self.params['base_method']=='CWT_phase_r' \
            or self.params['base_method']=='CWT_phase_a' or self.params['base_method']=='WTC':
            params = {'n_jobs': self.params['n_jobs'], 'verbose': self.params['verbose'], 'backend': self.params['backend']}
            base_dFC = TIME_FREQ(TF_method=self.params['base_method'], **params)
        else:
            params = {'W': self.params['W'], 'n_overlap': self.params['n_overlap']}
            base_dFC = SLIDING_WINDOW(sw_method=self.params['base_method'], \
                    tapered_window=self.params['tapered_window'], **params)
                    
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

        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=self.FCS_, \
            FCS_idx=Z, \
            TS_info=time_series.TS_info, \
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
    self.FCC_ : 
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

    def __init__(self, base_method='pear_corr', tapered_window=True, **params):
        
        assert base_method in self.base_methods_name_lst, \
            "Base method not recognized."
            
        self.measure_name_ = 'DiscreteHMM'
        self.is_state_based = True
        self.TPM = []
        self.FCS_ = []
        self.swc = None
        
        self.params_name_lst = ['base_method', 'tapered_window', 'n_hid_states', \
            'coi_correction', \
            'n_jobs', 'verbose', 'backend', \
            'n_subj_clstrs', 'W', 'n_overlap', 'n_states', 'normalization', \
            'num_subj', 'num_select_nodes', 'num_time_point', 'Fs_ratio', \
            'noise_ratio', 'num_realization']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
        
        self.params['base_method'] = base_method
        self.params['n_hid_states'] = self.params['n_states']
        self.params['tapered_window'] = tapered_window

    @property
    def measure_name(self):
        return self.measure_name_ #+ '_' + self.base_method

    def issame(self, dFC):
        if type(self)==type(dFC):
            for param_name in self.params:
                if self.params[param_name] != dFC.params[param_name]:
                    return False
        else:
            return False
        return True

    def estimate_FCS(self, time_series):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        # params = {'W': self.W, 'n_overlap': self.n_overlap, \
        #     'n_subj_clstrs': self.n_subj_clstrs, 'n_states': self.n_states, \
        #     'n_jobs': self.params['n_jobs'], 'verbose': self.params['verbose'], 'backend': self.params['backend']}
        self.swc = SLIDING_WINDOW_CLUSTR(base_method=self.params['base_method'], \
            tapered_window=self.params['tapered_window'], **self.params)
        self.swc.estimate_FCS(time_series=time_series)
        self.FCC_ = self.swc.estimate_dFCM(time_series=time_series)

        Models, Scores = [], []
        for i in range(100):
            model = hmm.MultinomialHMM(n_components=self.params['n_hid_states'])
            model.fit(self.FCC_.FCS_idx_array.reshape(-1, 1)) 
            score = model.score(self.FCC_.FCS_idx_array.reshape(-1, 1))
            Models.append(model)
            Scores.append(score)
            
        self.hmm_model = Models[np.argmax(Scores)]
        self.Z = self.hmm_model.predict(self.FCC_.FCS_idx_array.reshape(-1, 1))
        self.TPM = self.hmm_model.transmat_
        self.EPM = self.hmm_model.emissionprob_ 

        # self.hmm_model = hmm.MultinomialHMM(n_components=self.params['n_hid_states'])
        # self.hmm_model.fit(self.FCC_.FCS_idx_array.reshape(-1, 1))

        # self.Z = self.hmm_model.predict(self.FCC_.FCS_idx_array.reshape(-1, 1))
        # self.TPM = self.hmm_model.transmat_
        # self.EPM = self.hmm_model.emissionprob_ 

        self.FCS_ = np.zeros((self.params['n_hid_states'], \
            time_series.n_regions, time_series.n_regions))
        for i in range(self.params['n_hid_states']):
            if len(np.argwhere(self.Z==i))>0:
                self.FCS_[i,:,:] = np.mean(self.FCC_.get_dFC_mat(\
                    TRs=self.FCC_.TR_array[np.squeeze(np.argwhere(self.Z==i))]\
                        ), axis=0)  # III

        return self

    def estimate_dFCM(self, time_series):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        FCC = self.swc.estimate_dFCM(time_series=time_series)

        Z = self.hmm_model.predict(FCC.FCS_idx_array.reshape(-1, 1))

        dFCM = DFCM(measure=self)
        dFCM.set_dFC(FCSs=self.FCS_, \
            FCS_idx=Z, \
            TS_info=time_series.TS_info, \
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

        self.interval_ = np.arange(0, self.n_time_)
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
        info_dict['Fs_ratio'] = self.Fs_ratio
        info_dict['noise_ratio'] = self.noise_ratio
        info_dict['selected_nodes'] = self.select_nodes
        info_dict['nodes_info'] = self.nodes_info
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
            return [self.nodes_info_[0]] + [self.nodes_info_[i[0]+1] for i in self.nodes_lst] 

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

            self.interval_ = np.arange(start, end)
            
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
            self.interval_ = np.arange(start, end)

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

        if nodes_idx is None:
            self.nodes_selection_ = np.arange(0, self.n_regions_, 1)
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
    def dFC2dict(self, TRs=None, num_samples=None):
        # return dFC samples as a dictionary
        if TRs is None:
            TRs = self.TR_array
        if type(TRs) is list:
            TRs = np.array(TRs)
        TRs = TRs.astype(int)
        dFC_mat, TRs_new = self.get_dFC_mat(TRs=TRs, num_samples=num_samples)
        dFC_dict = {}
        for k, TR in enumerate(TRs_new):
            dFC_dict['TR'+str(TR)] = dFC_mat[k, :, :]
        return dFC_dict

    # test this
    def get_dFC_mat(self, TRs=None, num_samples=None):
        # get dFC matrices corresponding to 
        # the specified TRs 
        # TRs should be list not necessarily in order ?
        # if num_samples specified, it will downsample 
        # TRs to reach that number of samples
        # if num_samples > len(TRs) -> picks all TRs

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

    # def concat(self, dFCM):

    #     # test this method

    #     assert type(dFCM) is DFCM, \
    #             "The input must be of DFCM class"

    #     if self.FCPs_ is None:
    #         self.FCPs_ = dFCM.FCPs
    #         self.FCP_idx_ = dFCM.FCP_idx
    #         self.n_regions_ = dFCM.n_regions
    #         self.n_time_ = dFCM.n_time
    #         self.TR_array_ = dFCM.TR_array
    #     else:
    #         assert self.n_regions== dFCM.n_regions, \
    #             "dFCM region numbers missmatch."
    #         FCP_idx = dFCM.FCP_idx + self.FCPs.shape[0]
    #         self.FCPs_ = np.concatenate((self.FCPs_, dFCM.FCPs), axis=0)
    #         self.FCP_idx_ = np.concatenate((self.FCP_idx_, FCP_idx), axis=0)
    #         self.n_time_ = self.FCP_idx.shape[0]
    #         self.TR_array_ = np.concatenate((self.TR_array, dFCM.TR_array))

    def set_dFC(self, FCSs, FCS_idx=None, TS_info=None, TR_array=None):
        
        if len(FCSs.shape)==2:
            FCSs = np.expand_dims(FCSs, axis=0)

        if FCS_idx is None:
            FCS_idx = np.arange(start=0, stop=FCSs.shape[0], step=1)

        if type(FCS_idx) is list:
            FCS_idx = np.array(FCS_idx)

        if len(FCS_idx.shape)>1:
            FCS_idx = np.squeeze(FCS_idx)
        
        assert FCSs.shape[1] == FCSs.shape[2], \
                "FC matrices must be square."

        assert self.n_time==-1, \
            'why n_time is not -1 ?'
        
        if TR_array is None:
            TR_array = np.arange(start=self.n_time+1, stop=self.n_time+len(FCS_idx)+1, step=1)

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

        # ICA
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
        ALL_RECORDS = os.listdir(self.params['data_root_gordon'])
        ALL_RECORDS = [i for i in ALL_RECORDS if 'Rest' in i]
        ALL_RECORDS.sort()
        SUBJECTS_gordon = list()
        for s in ALL_RECORDS:
            num = s[:s.find('_')]
            SUBJECTS_gordon.append(num)
        SUBJECTS_gordon = list(set(SUBJECTS_gordon))
        SUBJECTS_gordon.sort()

        SUBJECTS = intersection(SUBJECTS_gordon, SUBJECTS_ica)

        print( str(len(SUBJECTS)) + ' subjects were found. ')

        # print( str(len(SUBJECTS)) + ' subjects were found. ' + str(self.params['num_subj']) + ' subjects were selected.')

        # SUBJECTS = SUBJECTS[0:self.params['num_subj']]

        return SUBJECTS

    def load_gordon(self, subj_id2load=None):

        SESSIONs = ['Rest1_LR' , 'Rest1_RL', 'Rest2_LR', 'Rest2_RL']
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

        BOLD = {}
        for session in SESSIONs:
            BOLD[session] = None
            for subject in SUBJECTS:

                subj_fldr = subject + '_' + session

                # LOAD BOLD Data

                DATA = hdf5storage.loadmat(self.params['data_root_gordon']+subj_fldr+'/ROI_data_Gordon_333_surf.mat')
                time_series = DATA['ROI_data']

                time_series = time_series.T

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

        SESSIONs = ['session_1']
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




'''
################################# dFC_analyzer #################################

def FCS_sim_calc(self, FCS_dict_i, FCS_dict_j, normalize=False):
    # FCS_sim_dict[FCS_i] = FCS_similarity_vec

    FCS_sim_dict = {}
    for FCS_i in FCS_dict_i['state_TC']:
        FCS_similarity_vec = np.zeros((len(FCS_dict_j['state_TC'])))
        for j, FCS_j in enumerate(FCS_dict_j['state_TC']):
            if normalize:
                FCS_similarity_vec[j] = self.FC_mat_corr( \
                    dFC_mat_normalize(C_t=FCS_dict_i['state_TC'][FCS_i]['FCS'], \
                        global_normalization=False), \
                    dFC_mat_normalize(C_t=FCS_dict_j['state_TC'][FCS_j]['FCS'], \
                        global_normalization=False) \
                ) 
            else:
                FCS_similarity_vec[j] = self.FC_mat_corr( \
                    FCS_dict_i['state_TC'][FCS_i]['FCS'], \
                    FCS_dict_j['state_TC'][FCS_j]['FCS']\
                ) 
        FCS_sim_dict[FCS_i] = FCS_similarity_vec

    return FCS_sim_dict

def states_FO_vec(self):
    # it assumes all subjects have all sessions and the order of their measures is the same ?
    # FO_vec_dict[session][measure][state] is the FO vector; each element is the FO for a subject
    
    FO_vec_dict = {}
    for s, methods_assess_subj_dict in enumerate(self.methods_assess_dict_lst):
        for session in methods_assess_subj_dict:
            if s==0:
                FO_vec_dict[session] = {}
            for measure in methods_assess_subj_dict[session]['FO']:
                if s==0:
                    FO_vec_dict[session][measure] = {}
                for state in methods_assess_subj_dict[session]['FO'][measure]:
                    if s==0:
                        FO_vec_dict[session][measure][state] = np.zeros((len(self.methods_assess_dict_lst),))
                    FO_vec_dict[session][measure][state][s] = methods_assess_subj_dict[session]['FO'][measure][state]

    return FO_vec_dict

def state_FO_corr(self):

    FO_vec_dict = self.states_FO_vec()
    FO_corr_dict = {}
    for session in FO_vec_dict:
        FO_corr_dict[session] = {}
        for measure_i in FO_vec_dict[session]:
            FO_corr_dict[session][measure_i] = {}
            for measure_j in FO_vec_dict[session]:
                FO_corr_dict[session][measure_i][measure_j] = {}

                FO_corr = np.zeros((len(FO_vec_dict[session][measure_i]), len(FO_vec_dict[session][measure_j])))
                for i, FCS_i in enumerate(FO_vec_dict[session][measure_i]):
                    for j, FCS_j in enumerate(FO_vec_dict[session][measure_j]):
                        FO_corr[i, j] = np.corrcoef( \
                                            FO_vec_dict[session][measure_i][FCS_i], \
                                            FO_vec_dict[session][measure_j][FCS_j] \
                                        )[0,1]

                FO_corr_dict[session][measure_i][measure_j]['FO_corr'] = FO_corr

    return FO_corr_dict

# todo: add save image
def similarity_analyze(self, SUBJs_dFC_session_sim_dict, verb=False, show_all_subj=False):

    ############ inter session dFC similarity averaged ############
    
    # dFC_session_sim_dict contains similarity between different sessions 
    # and in different measures in each subject
    avg_dFC_session_sim_dict = {}
    SUBJECTs = [subject for subject in SUBJs_dFC_session_sim_dict]
    for measure in SUBJs_dFC_session_sim_dict[SUBJECTs[0]]:
        avg_dFC_session_sim_dict[measure] = {}
        avg_dFC_session_sim_dict[measure]['sim_mat'] = 0
        for subject in SUBJECTs:
            avg_dFC_session_sim_dict[measure]['sim_mat'] += np.divide(SUBJs_dFC_session_sim_dict[subject][measure]['sim_mat'], len(SUBJECTs))
            avg_dFC_session_sim_dict[measure]['session_lst'] = SUBJs_dFC_session_sim_dict[subject][measure]['session_lst']
        
    visualize_conn_mat(avg_dFC_session_sim_dict, \
        title='avg inter session dFC similarity', \
        name_lst_key='session_lst', mat_key='sim_mat', \
        save_image=self.params['save_image'], output_root=self.params['output_root']+'similarity/inter_session_dFC' \
    )

    avg_measure_repro = {}
    for measure in avg_dFC_session_sim_dict:
        avg_measure_repro[measure] = np.mean(avg_dFC_session_sim_dict[measure]['sim_mat'])

    if verb:
        print('average inter session dFC similarity:')
        print_dict(avg_measure_repro)

    ############ inter session dFC similarity for all subjects ############

    if show_all_subj:
        if verb:
            print('inter session dFC similarity dict:')
            print_dict(SUBJs_dFC_session_sim_dict)

        for subject in SUBJs_dFC_session_sim_dict:
            visualize_conn_mat(SUBJs_dFC_session_sim_dict[subject], \
                title='inter session dFC similarity subj '+subject, \
                name_lst_key='session_lst', mat_key='sim_mat', \
                save_image=self.params['save_image'], output_root=self.params['output_root']+'similarity/inter_session_dFC'+subject \
                )

    ############ inter session measures FCS similarity ############

    FCS_session_sim_dict = self.FCS_session_similarity()
    if verb:
        print_dict(FCS_session_sim_dict)

    visualize_conn_mat(FCS_session_sim_dict, \
        title='inter session measures FCS similarity', \
        name_lst_key='session_lst', mat_key='sim_mat', \
        save_image=self.params['save_image'], output_root=self.params['output_root']+'similarity/inter_session_FCS' \
    )

    avg_measure_repro = {}
    for measure in FCS_session_sim_dict:
        avg_measure_repro[measure] = np.mean(FCS_session_sim_dict[measure]['sim_mat'])

    if verb:
        print('average measures FCS reproducibility:')
        print_dict(avg_measure_repro)

    ############ intra session measures FCS similarity ############

    # FCS_measure_sim_dict = self.FCS_measure_similarity()
    # if verb:
    #     print_dict(FCS_measure_sim_dict)
    # visualize_conn_mat(FCS_measure_sim_dict, title='measures FCS similarity matrix', name_lst_key='measure_lst', mat_key='sim_mat')

    return

def dFCM_state_match(self, trans_sim_dict, FCS_dict_i, FCS_dict_j, FCS_sim, matching_method='score'):

    # matching_method can be 'FCS', 'transition' or 'score' (combination of both)
    # trans_sim_dict[FCS_i]['trans_sim_vec'] = list(transition_similarity_vec) of subjs
    # returns 

    state_match_dict= {}
    state_match_dict['FCS_match'] = {}
    state_match_dict['orig_FCSs'] = {}
    state_match_dict['matched_FCSs'] = {}
    
    FCS_j_lst = [FCS for FCS in FCS_dict_j['state_TC']]
    for FCS_i in trans_sim_dict:

        # trans_sim_dict[FCS_i]['trans_sim_vec'] is list of trans_sim_vec of all subjects

        score = list()
        for trans_sim_vec in trans_sim_dict[FCS_i]['trans_sim_vec']:
            score.append(np.multiply(FCS_sim[FCS_i], trans_sim_vec))
        
        trans_sim = np.array(trans_sim_dict[FCS_i]['trans_sim_vec'])
        score = np.array(score)

        trans_sim_avg = np.nanmean(trans_sim, axis=0)
        score_avg = np.nanmean(score, axis=0)

        # choose match based on score/FCS/transition 
        if matching_method=='score':
            FCS_j_max = np.nanargmax(score_avg)
        if matching_method=='transition':
            FCS_j_max = np.nanargmax(trans_sim_avg)
        if matching_method=='FCS':
            FCS_j_max = np.argmax(FCS_sim[FCS_i])

        state_match_dict['FCS_match'][FCS_i] = {}
        state_match_dict['FCS_match'][FCS_i]['match'] = FCS_j_lst[FCS_j_max]
        state_match_dict['FCS_match'][FCS_i]['trans_corr'] = trans_sim_avg[FCS_j_max]
        state_match_dict['FCS_match'][FCS_i]['FCS_corr'] = FCS_sim[FCS_i][FCS_j_max]
        state_match_dict['FCS_match'][FCS_i]['score'] = score_avg[FCS_j_max]

        state_match_dict['orig_FCSs'][FCS_i] = FCS_dict_i['state_TC'][FCS_i]['FCS']
        state_match_dict['matched_FCSs'][FCS_i+'->'+FCS_j_lst[FCS_j_max]] = FCS_dict_j['state_TC'][FCS_j_lst[FCS_j_max]]['FCS']

    # avg over all FCSs of measure_i
    FCS_match_dict = state_match_dict['FCS_match']
    state_match_dict['avg_score'] = np.nanmean([FCS_match_dict[FCS_i]['score'] for FCS_i in FCS_match_dict])
    state_match_dict['avg_trans_corr'] = np.nanmean([FCS_match_dict[FCS_i]['trans_corr'] for FCS_i in FCS_match_dict])
    state_match_dict['avg_FCS_corr'] = np.mean([FCS_match_dict[FCS_i]['FCS_corr'] for FCS_i in FCS_match_dict])

    return state_match_dict

def state_match(self):
    # this function matches states of two different method
    # matching_method can be 'FCS', 'transition' or 'score' (combination of both)

    state_match = {}
    for session in self.FCS_sim_dict:

        FCS_dict = self.FCS_dict[session]
        FCS_sim_dict = self.FCS_sim_dict[session]

        # collecting similarity scores across subjects
        score_dict = {}
        for subj_i, subj_dict in enumerate(self.methods_assess_dict_lst_):
            for measure_i in subj_dict[session]['state_match']:
                if subj_i==0:
                    score_dict[measure_i] = {}
                for measure_j in subj_dict[session]['state_match'][measure_i]:
                    if subj_i==0:
                        score_dict[measure_i][measure_j] = {}
                    for FCS_i in subj_dict[session]['state_match'][measure_i][measure_j]:
                        trans_sim_vec = subj_dict[session]['state_match'][measure_i][measure_j][FCS_i]['trans_sim_vec']
        
                        if subj_i==0:
                            score_dict[measure_i][measure_j][FCS_i] = {}
                            score_dict[measure_i][measure_j][FCS_i]['trans_sim_vec'] = list()

                        score_dict[measure_i][measure_j][FCS_i]['trans_sim_vec'].append(trans_sim_vec)

        # averaging collected similarity scores
        state_match_dict = {}
        state_match_dict['final'] = {}
        state_match_dict['method_pairs'] = {}
        state_match_dict['final']['score'] = {}
        state_match_dict['final']['trans_corr'] = {}
        state_match_dict['final']['FCS_corr'] = {}
        state_match_dict['final']['score']['corr_mat'] = np.zeros((len(score_dict), len(score_dict)))
        state_match_dict['final']['trans_corr']['corr_mat'] = np.zeros((len(score_dict), len(score_dict)))
        state_match_dict['final']['FCS_corr']['corr_mat'] = np.zeros((len(score_dict), len(score_dict)))
        for measure_i_iter, measure_i in enumerate(score_dict):
            state_match_dict['method_pairs'][measure_i] = {}
            for measure_j_iter, measure_j in enumerate(score_dict):

                trans_sim_dict = score_dict[measure_i][measure_j]
                state_match_dict['method_pairs'][measure_i][measure_j] = self.dFCM_state_match( \
                    trans_sim_dict=trans_sim_dict, \
                    FCS_dict_i=FCS_dict[measure_i], \
                    FCS_dict_j=FCS_dict[measure_j], \
                    FCS_sim=FCS_sim_dict[measure_i][measure_j], \
                    matching_method=self.sim_assess_params['matching_method'] \
                )
                    
                # avg over all FCSs of measure_i as a matrix
                state_match_dict['final']['score']['corr_mat'][measure_i_iter, measure_j_iter] = state_match_dict['method_pairs'][measure_i][measure_j]['avg_score']
                state_match_dict['final']['trans_corr']['corr_mat'][measure_i_iter, measure_j_iter] = state_match_dict['method_pairs'][measure_i][measure_j]['avg_trans_corr']
                state_match_dict['final']['FCS_corr']['corr_mat'][measure_i_iter, measure_j_iter] = state_match_dict['method_pairs'][measure_i][measure_j]['avg_FCS_corr']

        state_match[session] = state_match_dict

    ###### results visualization ######

    for session in state_match:
        visualize_conn_mat(state_match[session]['final'], \
            title='intra session state match results ('+session+')', \
            name_lst_key=[measure for measure in state_match[session]['method_pairs']], \
            mat_key='corr_mat', \
            cmap='viridis',\
            save_image=self.params['save_image'], output_root=self.params['output_root']+'state_match/results_'+session, \
            fix_lim=True \
        )
    
    return state_match


def state_transition_analyze(self, dFCM_i, dFCM_j, \
    state_match_dict=None, \
    matching_method='score', \
    TRs=None, \
    subject='', \
    session='', \
    verb=False \
    ):

    normalize=False

    output_root = self.params['output_root']+'post_analysis/'+ \
        subject+'/state_match_'+ \
        dFCM_i.measure.measure_name+'_'+dFCM_j.measure.measure_name+ \
        '_'+session+'_'

    if verb:
        print('***** Subject '+subject+' Session '+session+' *****')

    if TRs is None:
        TRs = TR_intersection([dFCM_i, dFCM_j])
    TRs_lst = list()
    for TR in TRs:
        TRs_lst.append('TR'+str(TR))

    TC_name_lst = list()
    TC_name_lst.append(dFCM_i.measure.measure_name)
    TC_name_lst.append(dFCM_j.measure.measure_name)

    num_FCS = max(len(dFCM_i.FCSs), len(dFCM_j.FCSs))
    FCS_lst = ['FCS'+str(i+1) for i in range(num_FCS)]

    ############ state transition without matching ############

    state_TC_i = dFCM_i.state_TC(TRs=TRs_lst, \
                    state_match=False, state_match_dict=None \
                )

    state_TC_j = dFCM_j.state_TC(TRs=TRs_lst, \
                    state_match=False, state_match_dict=None \
                )

    visualize_state_TC([state_TC_i, state_TC_j], \
        TRs=TRs, TC_name_lst=TC_name_lst , \
        state_lst=FCS_lst, \
        title='state transition without matching', \
        save_image=self.params['save_image'], output_root=output_root+'without_match' \
    )

    ############ state matching ############
        
    # here the FCSs are matched using state_match function which can be 
    # by FCS similarity, transition similarity, or combination of both

    if state_match_dict is None:

        #trans_sim[FCS_i]['trans_sim_vec'] = transition_similarity_vec
        trans_sim = self.dFCM_trans_sim( \
            dFCM_i=dFCM_i, \
            dFCM_j=dFCM_j, \
            common_TRs=None \
        )

        trans_sim_dict = {}
        for FCS_i in trans_sim:
            trans_sim_dict[FCS_i] = {}
            trans_sim_vec = trans_sim[FCS_i]['trans_sim_vec']
            trans_sim_dict[FCS_i]['trans_sim_vec'] = [trans_sim_vec]

        FCS_dict_i = {}
        FCS_dict_i['state_TC'] = {}
        for FCS in dFCM_i.FCSs:
            FCS_dict_i['state_TC'][FCS] = {}
            FCS_dict_i['state_TC'][FCS]['FCS'] = dFCM_i.FCSs[FCS]

        FCS_dict_j = {}
        FCS_dict_j['state_TC'] = {}
        for FCS in dFCM_j.FCSs:
            FCS_dict_j['state_TC'][FCS] = {}
            FCS_dict_j['state_TC'][FCS]['FCS'] = dFCM_j.FCSs[FCS]

        FCS_sim = self.FCS_sim_calc( \
            FCS_dict_i=FCS_dict_i, \
            FCS_dict_j=FCS_dict_j, \
            normalize=False \
        )
        # FCS_sim_dict[FCS_i] = FCS_similarity_vec
        state_match_dict = self.dFCM_state_match( \
            trans_sim_dict=trans_sim_dict, \
            FCS_dict_i=FCS_dict_i, \
            FCS_dict_j=FCS_dict_j, \
            FCS_sim=FCS_sim, \
            matching_method=matching_method \
        )

    # visualization
    D_A = state_match_dict['orig_FCSs']
    D_B = state_match_dict['matched_FCSs']

    if normalize:
        visualize_conn_mat(dFC_dict_normalize(D_A), \
            disp_diag=False, cmap='viridis', \
            save_image=self.params['save_image'], output_root=output_root+'original_FCSs' \
        )
        visualize_conn_mat(dFC_dict_normalize(D_B), \
            disp_diag=False, cmap='viridis', \
            save_image=self.params['save_image'], output_root=output_root+'matched_FCSs' \
        )
    else:
        visualize_conn_mat(D_A, \
            disp_diag=False, cmap='viridis', \
            save_image=self.params['save_image'], output_root=output_root+'original_FCSs' \
        )
        visualize_conn_mat(D_B, \
            disp_diag=False, cmap='viridis', \
            save_image=self.params['save_image'], output_root=output_root+'matched_FCSs' \
        )

    print('state TC corr of different pairs of states: ')
    print(np.array([state_match_dict['FCS_match'][FCS_i]['trans_corr'] for FCS_i in  state_match_dict['FCS_match']]))
    print('FCS corr of different pairs of states: ')
    print(np.array([state_match_dict['FCS_match'][FCS_i]['FCS_corr'] for FCS_i in  state_match_dict['FCS_match']]))
    print('score of different pairs of states: ')
    print(np.array([state_match_dict['FCS_match'][FCS_i]['score'] for FCS_i in  state_match_dict['FCS_match']]))

    ##### print matched scores #####

    print('average state TC corr: {:.2f}'.format(state_match_dict['avg_trans_corr']))
    print('average FCS corr: {:.2f}'.format(state_match_dict['avg_FCS_corr']))
    print('average score: {:.2f}'.format(state_match_dict['avg_score']))

    ############ state transition after matching ############

    state_TC_i = dFCM_i.state_TC(TRs=TRs_lst, \
                    state_match=True, state_match_dict=state_match_dict \
                )

    state_TC_j = dFCM_j.state_TC(TRs=TRs_lst, \
                    state_match=False, state_match_dict=None \
                )

    visualize_state_TC([state_TC_i, state_TC_j], \
        TRs=TRs, TC_name_lst=TC_name_lst, \
        state_lst=FCS_lst, \
        title='state transition after matching', \
        save_image=self.params['save_image'], output_root=output_root+'after_match' \
    )

    if verb:
        print("Whole state Time Course Equality: {:.2f}".format(np.sum(state_TC_i == state_TC_j)/len(state_TC_i)))

    ##### visualize matched states co transitions #####

    for key_a in state_match_dict['FCS_match']:

        key_b = state_match_dict['FCS_match'][key_a]['match']

        state_act_dict_i = dFCM_i.state_act_dict(TRs=TRs)
        state_act_dict_j = dFCM_j.state_act_dict(TRs=TRs)

        state_TC_i = state_act_dict_i['state_TC'][key_a]['act_TC']
        state_TC_j = state_act_dict_j['state_TC'][key_b]['act_TC']

        visualize_state_TC([state_TC_i, state_TC_j], \
            TRs=TRs, TC_name_lst=TC_name_lst, \
            state_lst=['off', 'on'], \
            title='state transition of ' + key_a + ' and ' + key_b, \
            save_image=self.params['save_image'], output_root=output_root+'trans_'+key_a+'_and_'+key_b \
        )

        if verb:
            print('state Time Course Equality of {s1} and {s2}: '.format(s1=key_a, s2=key_b) + '{:.2f}'.format(state_match_dict['FCS_match'][key_a]['trans_corr']))

    return state_match_dict

def FC_mat_corr(self, A, B):
    # it excludes diagonal values
    A = np.multiply(A, 1-np.eye(len(A)))
    B = np.multiply(B, 1-np.eye(len(B)))
    return np.corrcoef(A.flatten(), B.flatten())[0,1]

def similarity(self, D_A, D_B, normalize=False, return_matched=False):
    # searchs D_A FC_mats in D_B
    # this functions is recommended to be used only for inter session similarity 
    # assessment since it does not take into account the temporal transition
    similarity_dict = {}
    for key_a in D_A:
        similarity_vec = np.zeros((len(D_B)))
        b_keys = [key for key in D_B]
        for b, key_b in enumerate(D_B):
            if normalize:
                similarity_vec[b] = self.FC_mat_corr( \
                    dFC_mat_normalize(C_t=D_B[key_b], \
                        global_normalization=False), \
                    dFC_mat_normalize(C_t=D_A[key_a], \
                        global_normalization=False) \
                ) 
            else:
                similarity_vec[b] = self.FC_mat_corr( \
                    D_B[key_b], \
                    D_A[key_a]\
                ) 
        similarity_dict[key_a] = {}
        similarity_dict[key_a]['match'] = b_keys[np.argmax(similarity_vec)]
        similarity_dict[key_a]['score'] = np.max(similarity_vec)
    similarity_score = np.mean([similarity_dict[key_a]['score'] for key_a in similarity_dict])

    if return_matched:
        return similarity_score, similarity_dict
    else:
        return similarity_score

def FCS_session_similarity(self):
    # measures inter session similarity
    
    FCS_session_sim_dict = {}
    SESSIONs = [session for session in self.MEASURES_fit_lst]
    for m in range(len(self.MEASURES_fit_lst[SESSIONs[0]])):
        # if the measure is DD -> continue to the next loop
        if not self.MEASURES_fit_lst[SESSIONs[0]][m].is_state_based:
            continue
        # this measure is a name/string
        measure=self.MEASURES_fit_lst[SESSIONs[0]][m].measure_name
        FCS_session_sim_dict[measure] = {}
        FCS_session_sim_dict[measure]['session_lst'] = SESSIONs
        FCS_session_sim_dict[measure]['sim_mat'] = np.zeros((len(SESSIONs),len(SESSIONs)))
        for i, session_i in enumerate(SESSIONs):
            for j, session_j in enumerate(SESSIONs):

                assert self.MEASURES_fit_lst[session_i][m].measure_name==self.MEASURES_fit_lst[session_j][m].measure_name, \
                    'measures mismatch!'
                
                C_A = self.MEASURES_fit_lst[session_i][m].FCS
                C_B = self.MEASURES_fit_lst[session_j][m].FCS
                D_A = {}
                for k in range(C_A.shape[0]):
                    D_A['FCS'+str(i+1)] = C_A[k,:,:]
                D_B = {}
                for k in range(C_B.shape[0]):
                    D_B['FCS'+str(i+1)] = C_B[k,:,:]
                # self.similarity(dFC_dict_normalize(D_A), dFC_dict_normalize(D_B), return_matched=False, normalize=True)
                FCS_session_sim_dict[measure]['sim_mat'][i, j] = self.similarity(D_A, D_B, return_matched=False, normalize=False) # normalize ??
    return FCS_session_sim_dict

def FCS_measure_similarity(self):
    
    # this function returns similarity of different measures' FCSs regardless of state transitions
    FCS_measure_sim_dict = {}
    for session in self.MEASURES_fit_lst:
        # we only keep SB measures:
        sb_measures_lst = self.SB_MEASURES_lst(self.MEASURES_fit_lst[session])
        FCS_measure_sim_dict[session] = {}
        FCS_measure_sim_dict[session]['measure_lst'] = [measure.measure_name for measure in sb_measures_lst]
        FCS_measure_sim_dict[session]['sim_mat'] = np.zeros((len(sb_measures_lst),len(sb_measures_lst)))
        for i, measure_i in enumerate(sb_measures_lst):
            for j, measure_j in enumerate(sb_measures_lst):
                C_A = measure_i.FCS
                C_B = measure_j.FCS
                D_A = {}
                for k in range(C_A.shape[0]):
                    D_A['FCS'+str(k+1)] = C_A[k,:,:]
                D_B = {}
                for k in range(C_B.shape[0]):
                    D_B['FCS'+str(k+1)] = C_B[k,:,:]
                # self.similarity(dFC_dict_normalize(D_A), dFC_dict_normalize(D_B), return_matched=False, normalize=True)
                FCS_measure_sim_dict[session]['sim_mat'][i, j] = self.similarity(D_A, D_B, return_matched=False, normalize=False) # normalize ??

    return FCS_measure_sim_dict

def dFC_session_similarity(self, dFCM_dict):
    # measures inter session similarity

    dFC_session_sim_dict = {}
    SESSIONs = [session for session in dFCM_dict]
    for measure in dFCM_dict[SESSIONs[0]]:
        dFC_session_sim_dict[measure] = {}
        dFC_session_sim_dict[measure]['session_lst'] = SESSIONs
        dFC_session_sim_dict[measure]['sim_mat'] = np.zeros((len(SESSIONs),len(SESSIONs)))
        for i, session_i in enumerate(SESSIONs):
            for j, session_j in enumerate(SESSIONs):

                # dFCM_dict[session_i][measure] is a dFCM object
                # it only considers num_samples number of samples for measuring similarity (uniformly picked)

                D_A = dFCM_dict[session_i][measure].dFC2dict(num_samples=self.sim_assess_params['num_samples'])
                D_B = dFCM_dict[session_j][measure].dFC2dict(num_samples=self.sim_assess_params['num_samples'])

                # self.similarity(dFC_dict_normalize(D_A), dFC_dict_normalize(D_B), return_matched=False, normalize=True)
                dFC_session_sim_dict[measure]['sim_mat'][i, j] = self.similarity(D_A, D_B, return_matched=False, normalize=False) # normalize ??
    return dFC_session_sim_dict

def dFC_measure_similarity(self, dFCM_dict):
    # this is not useful, instead we are using dFC_corr
    pass


def dFCM_trans_sim(self, dFCM_i, dFCM_j, common_TRs=None):

    # returns, for each pair of FCS of the two dFCMs, the portion 
    # of time that they have been both active as a dict of 
    # trans_sim[FCS_i]['trans_sim_vec'] = transition_similarity_vec
    # it only considers TRs in common_TRs

    if common_TRs is None:
        common_TRs = TR_intersection([dFCM_i, dFCM_j])

    state_act_dict_i = dFCM_i.state_act_dict(TRs=common_TRs)
    state_act_dict_j = dFCM_j.state_act_dict(TRs=common_TRs)

    # state_act_dict['state_TC'][FCS_key]['act_TC']
    # state_act_dict['state_TC'][FCS_key]['FCS']
    # state_act_dict['TR_array']

    trans_sim_dict = {}
    for FCS_i in state_act_dict_i['state_TC']:
        trans_sim_dict[FCS_i] = {}
        # FCS_similarity_vec = np.zeros((len(state_act_dict_j['state_TC'])))
        # if coat==False, transition_similarity_vec won't effect the final multiplication 
        # by FCS_similarity_vec
        transition_similarity_vec = np.ones((len(state_act_dict_j['state_TC'])))
        for j, FCS_j in enumerate(state_act_dict_j['state_TC']):

            # transition_similarity_vec[j] = np.mean( \
            #     state_act_dict_i['state_TC'][FCS_i]['act_TC']== \
            #     state_act_dict_j['state_TC'][FCS_j]['act_TC'] \
            # )

            # how many times out of all the times i has been on, j has been on too
            if np.sum(state_act_dict_i['state_TC'][FCS_i]['act_TC'])==0:
                if np.sum(state_act_dict_j['state_TC'][FCS_j]['act_TC'])==0:
                    transition_similarity_vec[j] = 1
                else:
                    transition_similarity_vec[j] = 0
            else:
                transition_similarity_vec[j] = np.divide( \
                    np.sum(np.multiply( \
                        state_act_dict_i['state_TC'][FCS_i]['act_TC'], \
                        state_act_dict_j['state_TC'][FCS_j]['act_TC'] )), \
                    np.sum(state_act_dict_i['state_TC'][FCS_i]['act_TC']) \
                )

        trans_sim_dict[FCS_i]['trans_sim_vec'] = transition_similarity_vec

    return trans_sim_dict

def methods_avg_trans_freq(self):
    # it assumes all subjects have all sessions and the order of their measures is the same ?
    # FO_vec_dict[session][state] is the FO vector; each element is the FO for a subject
    
    avg_trans_freq = {}
    for s, methods_assess_subj_dict in enumerate(self.methods_assess_dict_lst):
        for session in methods_assess_subj_dict:
            if s==0:
                avg_trans_freq[session] = {}
            for measure in methods_assess_subj_dict[session]['trans_freq']:
                trans_freq = methods_assess_subj_dict[session]['trans_freq'][measure]
                if s==0:
                    avg_trans_freq[session][measure] = {}
                    avg_trans_freq[session][measure]['trans_freq'] = list()
                    avg_trans_freq[session][measure]['trans_norm'] = list()   
                avg_trans_freq[session][measure]['trans_freq'].append(trans_freq['trans_freq'])
                avg_trans_freq[session][measure]['trans_norm'].append(trans_freq['trans_norm'])

    for session in avg_trans_freq:
        for measure in avg_trans_freq[session]:
            avg_trans_freq[session][measure]['trans_freq'] = np.array(avg_trans_freq[session][measure]['trans_freq'])
            avg_trans_freq[session][measure]['trans_freq'] = np.mean(avg_trans_freq[session][measure]['trans_freq'])
            avg_trans_freq[session][measure]['trans_norm'] = np.array(avg_trans_freq[session][measure]['trans_norm'])
            avg_trans_freq[session][measure]['trans_norm'] = np.mean(avg_trans_freq[session][measure]['trans_norm'])

    return avg_trans_freq

 def visualize_dyn_conns(self, SUBJs_dyn_conn):
        
    for subject in SUBJs_dyn_conn:

        visualize_conn_mat(data=SUBJs_dyn_conn[subject], \
            title='Subject '+subject+' Dynamic Connections', \
            save_image=self.params['save_image'], \
            output_root=self.params['output_root']+'DYN_CONN/'+'subject'+subject+'_dyn_conn', \
            fix_lim=True \
        )

############################# Dynamic Connection Detector class ################################

"""

todo:
- 
"""

from statsmodels.tsa.api import VAR
from scipy.stats import norm

class DYN_CONN_DETECTOR:

    a = 0.95

    def __init__(self, **params):
        self.VAR_model = None
        self.lag_order = None
        self.TH_mask = None
        self.params = params

    # @property
    # def methods_corr(self):
    #     return np.mean(self.methods_corr_lst, axis=0)

    def train_VAR(self, time_series, p=None):
        self.VAR_model = VAR(time_series.data.T)

        if p is None:
            self.VAR_model = self.VAR_model.fit(maxlags=10, ic='aic')
        else:
            self.VAR_model = self.VAR_model.fit(p)

        self.lag_order = self.VAR_model.k_ar

    def subj_lvl_calc_TH_mask(self, time_series, MEASURES_lst, N, L):

        SURROGATE = self.gen_surrogate( \
            time_series=time_series, \
            N=N, L=L, verbose=self.params['verbose']  \
        )
        dFCM_var = self.calc_dFC_var(time_series=SURROGATE, MEASURES_lst=MEASURES_lst)
        TH_mask = self.calc_TH_mask(dFCM_var, a=self.a)
        return TH_mask

    def calc_subj_TH_mask(self, time_series, MEASURES_lst, \
        N, L=None):

        SUBJECTs = list(set(time_series.subj_id_array))
        
        if self.params['n_jobs'] is None:
            SUBJs_TH_mask_lst = list()
            for subject in SUBJECTs:
                SURROGATE = self.gen_surrogate( \
                    time_series=time_series.get_subj_ts(subj_id=subject), \
                    N=N, L=L, verbose=self.params['verbose']  \
                )
                dFCM_var = self.calc_dFC_var(time_series=SURROGATE, MEASURES_lst=MEASURES_lst)
                TH_mask = self.calc_TH_mask(dFCM_var, a=self.a)
                SUBJs_TH_mask_lst.append(TH_mask)
        else:
            SUBJs_TH_mask_lst = Parallel( \
                    n_jobs=self.params['n_jobs'], \
                    verbose=self.params['verbose'] , \
                    backend=self.params['backend'])( \
                delayed(self.subj_lvl_calc_TH_mask)( \
                    time_series=time_series.get_subj_ts(subj_id=subject), \
                    MEASURES_lst=MEASURES_lst, \
                    N=N, L=L \
                    ) \
                    for subject in SUBJECTs)

        SUBJs_TH_mask = {}
        for i, TH_mask in enumerate(SUBJs_TH_mask_lst):
            SUBJs_TH_mask[SUBJECTs[i]] = TH_mask

        return SUBJs_TH_mask

    def gen_surrogate(self, time_series, N, L=None, verbose=0):
        if L is None:
            L = time_series.n_time

        SURROGATE = None
        for n in range(N):

            t0 = np.random.choice(\
                range(time_series.n_time-self.lag_order), \
                size=1, replace=False)[0]

            simul = self.VAR_model.forecast(time_series.data.T[t0:t0+self.lag_order, :], L)

            if SURROGATE is None:
                SURROGATE = TIME_SERIES(data=simul.T, subj_id='surrogate'+str(n+1), Fs=1/0.72, TS_name='BOLD Surrogate')
            else:
                SURROGATE.append_ts(new_time_series=simul.T, subj_id='surrogate'+str(n+1))

        if verbose==1:
            print(SURROGATE.n_regions, SURROGATE.n_time)

        return SURROGATE

    def calc_dFC_var(self, time_series, MEASURES_lst):

        # OUTPUT shape = [sample, measure, node, node]

        dFC_analyzer = DFC_ANALYZER(MEASURES_lst=MEASURES_lst, \
            n_jobs=self.params['n_jobs'], verbose=self.params['verbose'] , backend=self.params['backend'] \
            )

        print("FCS estimation started...")
        dFC_analyzer.estimate_group_FCS(time_series=time_series)
        print("FCS estimation done.")

        ### estimate dFCM ###

        print("dFCM estimation started...")
        # SUBJs_dFC_var for SURROGATE is dFC_var of different bootstrap SAMPLEs
        SAMPLEs_dFC_var = dFC_analyzer.group_dFCM_assess( \
            time_series=time_series \
            )
        print("dFCM estimation done.")

        MEASURES_sample_dFC_var = {}
        for sample in SAMPLEs_dFC_var:
            for measure in SAMPLEs_dFC_var[sample]:
                # SAMPLEs_dFC_var[sample][measure] is a n_region x n_region dFC_var_mat
                if measure in MEASURES_sample_dFC_var:
                    MEASURES_sample_dFC_var[measure].append(SAMPLEs_dFC_var[sample][measure])
                else:
                    MEASURES_sample_dFC_var[measure] = []
                    MEASURES_sample_dFC_var[measure].append(SAMPLEs_dFC_var[sample][measure])

        for measure in MEASURES_sample_dFC_var:
            MEASURES_sample_dFC_var[measure] = np.array(MEASURES_sample_dFC_var[measure])

        return MEASURES_sample_dFC_var

    def calc_TH_mask(self, MEASURES_sample_dFC_var, a=0.95):

        # returns list of TH_masks for different measures

        TH_mask = {}
        for measure in MEASURES_sample_dFC_var:
            n_regions = MEASURES_sample_dFC_var[measure].shape[1]
            TH_mask_mat = np.zeros((n_regions, n_regions))
            for i in range(n_regions):
                for j in range(n_regions):
                    C = np.squeeze(MEASURES_sample_dFC_var[measure][:, i, j])
                    mu, sigma = norm.fit(C)
                    TH_mask_mat[i, j] = norm.ppf(a, mu, sigma)
            TH_mask[measure] = TH_mask_mat

        return TH_mask

    def mask_SUBJs_dFC(self, SUBJs_dFC_var, SUBJs_TH_mask):

        SUBJs_dyn_conn = {}
        for subject in SUBJs_dFC_var:
            dyn_conn = {}
            for measure in SUBJs_dFC_var[subject]:
                dyn_conn[measure] = (SUBJs_dFC_var[subject][measure]>=SUBJs_TH_mask[subject][measure])*[1] 
            SUBJs_dyn_conn[subject] = dyn_conn

        return SUBJs_dyn_conn

'''