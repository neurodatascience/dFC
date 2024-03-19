# -*- coding: utf-8 -*-
"""
functions for facilitating dynamic functional connectivity analysis

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import os
from copy import deepcopy

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from nilearn.plotting import plot_markers
from scipy import signal, stats

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
fig_bbox_inches = "tight"
fig_pad = 0.1
show_title = False
save_fig_format = "png"

################################# Other Functions ####################################


# test
def zip_name(name):
    # zip measure names
    if "Clustering" in name:
        new_name = "SWC"
    if "CAP" in name:
        new_name = "CAP"
    if "ContinuousHMM" in name:
        new_name = "CHMM"
    if "Windowless" in name:
        new_name = "WL"
    if "DiscreteHMM" in name:
        new_name = "DHMM"
    if "Time-Freq" in name:
        new_name = "TF"
    if "SlidingWindow" in name:
        new_name = "SW"
    return new_name


# test
# pear_corr problem
def unzip_name(name):
    # unzip measure names
    if "SWC" in name:
        new_name = "Clustering"
    elif "CAP" in name:
        new_name = "CAP"
    elif "CHMM" in name:
        new_name = "ContinuousHMM"
    elif "WL" in name:
        new_name = "Windowless"
    elif "DHMM" in name:
        new_name = "DiscreteHMM"
    elif "TF" in name:
        new_name = "Time-Freq"
    elif "SW" in name:
        new_name = "SlidingWindow"
    return new_name


def find_new_order(old_list, new_list):
    """
    new_order is a list of indices
    old_list = ['E', 'B', 'A', 'C', 'D']
    new_list = ['A', 'B', 'C', 'D', 'E']
    """
    new_order = [old_list.index(a) for a in new_list]
    return new_order


def mat_reorder(A, new_order):
    """
    new_order must be a list of indices:
    old_list = ['E', 'B', 'A', 'C', 'D']
    new_list = ['A', 'B', 'C', 'D', 'E']
    new_order = find_new_order(old_list, new_list)
    A_sorted is a copy of A
    """
    assert (
        len(new_order) == A.shape[0] and len(new_order) == A.shape[1]
    ), "dimension mismatch in reordering."
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
def filter_dFC_lst(dFC_lst, **param_dict):
    dFC_lst2check = list()
    for dFC in dFC_lst:
        if dFC.measure.param_match(**param_dict):
            dFC_lst2check.append(dFC)
    return dFC_lst2check


def SW_downsample(data, Fs, W, n_overlap, tapered_window=False):
    """
    data = (n_time, ...)
    the time samples will be picked after
    averaging over a window which slides
    W is in sec
    SWed_data = (n_time_new, ...)
    """

    SWed_data = list()
    L = data.shape[0]
    # change W to timepoints
    W = int(W * Fs)
    step = int((1 - n_overlap) * W)
    if step == 0:
        step = 1

    window_taper = signal.windows.gaussian(W, std=3 * W / 22)

    TR_array = list()
    for l in range(0, L - W + 1, step):

        ######### creating a rectangel window ############
        window = np.zeros((L))
        window[l : l + W] = 1

        ########### tapering the window ##############
        if tapered_window:
            window = signal.convolve(window, window_taper, mode="same") / sum(
                window_taper
            )

        # int(l-W/2):int(l+3*W/2) is the nonzero interval after tapering
        SWed_data.append(np.average(data, weights=window, axis=0))

        TR_array.append(int((l + (l + W)) / 2))

    SWed_data = np.array(SWed_data)
    return SWed_data


def mutual_information(X, Y, N_bins=100):
    """Mutual information for joint histogram
    https://matthew-brett.github.io/teaching/mutual_information.html#:~:text=Mutual%20information%20is%20a%20measure,signal%20intensity%20in%20the%20first.
    """

    # 2D histogram
    hist_2d, x_edges, y_edges = np.histogram2d(X, Y, bins=N_bins)

    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def corr2distance(corr_mat, metric):
    """
    metric can be: MI, euclidean_distance, spearman,
    corr (pearson)
    """
    if metric == "MI":
        if np.any(corr_mat > 1):
            print("MI values cannot be converted to distances.")
        # negative corr will be > 1.0
        dist_mat = 1 - corr_mat
    elif metric == "euclidean_distance":
        dist_mat = corr_mat
    else:
        # negative corr will be > 1.0
        dist_mat = 1 - corr_mat
    # dist_mat must be symmetric
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    # diagonal values of dist_mat must equal exactly zero
    np.fill_diagonal(dist_mat, 0)
    return dist_mat


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

    if (
        np.linalg.norm(x - np.mean(x)) ** 2 == 0
        and np.linalg.norm(y - np.mean(y)) ** 2 == 0
    ):
        return 0
    return 0.5 * (
        (np.linalg.norm((x - np.mean(x)) - (y - np.mean(y))) ** 2)
        / (np.linalg.norm(x - np.mean(x)) ** 2 + np.linalg.norm(y - np.mean(y)) ** 2)
    )


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
    N_edges = 200  # number of edges to keep
    if property == "shortest_path" or property == "clustering_coef":
        threshold = True

    G = nx.from_numpy_array(np.abs(A))
    G.remove_edges_from(nx.selfloop_edges(G))
    # G = G.to_undirected()

    # pruning edges
    if threshold:
        labels = [d["weight"] for (u, v, d) in G.edges(data=True)]
        labels.sort()
        threshold = labels[-1 * N_edges]
        ebunch = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < threshold]
        G.remove_edges_from(ebunch)

    if binarize:
        weight = "None"
    else:
        weight = "weight"

    if property == "ECM":
        graph_property = nx.eigenvector_centrality(G, weight=weight)
        graph_property = [graph_property[node] for node in graph_property]
        graph_property = np.array(graph_property)
    if property == "shortest_path":
        SHORTEST_PATHS = dict(nx.shortest_path_length(G, weight=weight))
        graph_property = np.zeros((A.shape[0], A.shape[0]))
        for node_i in SHORTEST_PATHS:
            for node_j in SHORTEST_PATHS[node_i]:
                graph_property[node_i, node_j] = SHORTEST_PATHS[node_i][node_j]
        graph_property = graph_property + graph_property.T
        graph_property = dFC_mat2vec(graph_property)
    if property == "degree":
        graph_property = [G.degree(weight=weight)[node] for node in G]
        graph_property = np.array(graph_property)
    if property == "clustering_coef":
        graph_property = nx.clustering(G, weight=weight)
        graph_property = [graph_property[node] for node in graph_property]
        graph_property = np.array(graph_property)

    return graph_property


def rank_norm(dFC_mat, global_norm=True):
    """
    dFC_mat = (n_time, n_region, n_region)
    dFC_mat_norm = rank_norm(dFC_mat)
    if global_norm=True, all time points ranked together, ow separately
    if dFC_mat = (n_region, n_region) -> dFC_mat_new = (n_region, n_region)
    """
    dFC_mat_copy = deepcopy(dFC_mat)
    flag_dim = False
    if len(dFC_mat_copy.shape) < 3:
        dFC_mat_copy = np.expand_dims(dFC_mat_copy, axis=0)
        flag_dim = True
    assert dFC_mat_copy.shape[1] == dFC_mat_copy.shape[2], "dimension mismatch."
    n_time = dFC_mat_copy.shape[0]
    n_region = dFC_mat_copy.shape[1]
    dFC_vecs = dFC_mat2vec(dFC_mat_copy)  # (n_time, (n_region*(n_region-1))/2)
    if global_norm:
        dFC_vecs_flatten = dFC_vecs.flatten()  # (n_time*(n_region*(n_region-1))/2,)
        dFC_vecs_flatten_ranked = stats.rankdata(dFC_vecs_flatten)
        dFC_vecs_ranked = dFC_vecs_flatten_ranked.reshape(
            (n_time, -1)
        )  # (n_time, (n_region*(n_region-1))/2)
        dFC_mat_ranked = dFC_vec2mat(
            dFC_vecs_ranked, N=n_region
        )  # (n_time, n_region, n_region)
        dFC_mat_new = dFC_mat_ranked
    else:
        # normalize time point-wise
        dFC_vecs_new = list()
        for i, vec in enumerate(dFC_vecs):
            vec_ranked = stats.rankdata(vec)  # (n_region*(n_region-1))/2,)
            dFC_vecs_new.append(vec_ranked)
        dFC_vecs_new = np.array(dFC_vecs_new)  # (n_time, (n_region*(n_region-1))/2)
        dFC_mat_new = dFC_vec2mat(
            dFC_vecs_new, N=n_region
        )  # (n_time, n_region, n_region)
    if flag_dim:
        dFC_mat_new = np.squeeze(dFC_mat_new)  # (n_region, n_region)

    return dFC_mat_new


def rank_norm_dFC_dict(dFC_dict, global_norm=True):
    """
    rank normalize dFC matrices in dFC_dict
    dFC_dict = {'TR1': FC_mat1, 'TR2': FC_mat2, ...}
    """
    dFC_mat = np.array([dFC_dict[TR] for TR in dFC_dict])
    dFC_mat_new = rank_norm(dFC_mat, global_norm=global_norm)
    dFC_dict_new = {}
    for i, TR in enumerate(dFC_dict):
        dFC_dict_new[TR] = dFC_mat_new[i, :, :]
    return dFC_dict_new


def cat_data(X_t, N):
    """
    X_t = (time, roi, roi)
    X_t is preferable to be ranked prior to cat_data
    """
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
    """
    dFC_mat and mask will be vectorized using dFC_mat2vec
    mask = (roi, roi)
    """
    dFC_vecs = dFC_mat2vec(dFC_mat)
    mask_vec = dFC_mat2vec(mask)

    dFC_vec_new = list()
    for dFC_vec in dFC_vecs:
        dFC_vec_new.append(dFC_vec[mask_vec])
    dFC_vec_new = np.array(dFC_vec_new)

    return dFC_vec_new


# test
# toDo: use ssd.squareform
def dFC_mat2vec(C_t):
    """
    C_t must be an array of matrices or a single matrix
    diagonal values will not be included. if you want to include
    them set k=0
    if C_t is a single matrix, F will be one dim
    changing F will not change C_t
    """
    if len(C_t.shape) == 2:
        assert C_t.shape[0] == C_t.shape[1], "C is not a square matrix"
        return C_t[np.triu_indices(C_t.shape[1], k=1)]

    F = list()
    for t in range(C_t.shape[0]):
        C = C_t[t, :, :]
        assert C.shape[0] == C.shape[1], "C is not a square matrix"
        F.append(C[np.triu_indices(C_t.shape[1], k=1)])

    F = np.array(F)
    return F


# test
# toDo: use ssd.squareform
def dFC_vec2mat(F, N):
    """
    diagonal values are set to 1.0
    F shape is (observations, features)
    """
    C = list()
    iu = np.triu_indices(N, k=1)
    for i in range(F.shape[0]):
        K = np.zeros((N, N))
        K[iu] = F[i, :]
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


def intersection(lst1, lst2):  # input is a list
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def TR_intersection(dFC_lst):  # input is a list of dFC objs
    TRs_lst_old = dFC_lst[0].TR_array
    common_Fs = dFC_lst[0].TS_info["Fs"]
    for dFC in dFC_lst:
        assert dFC.TS_info["Fs"] == common_Fs, "Fs mismatch. Cannot find the common TRs"

        TRs_lst_new = intersection(TRs_lst_old, dFC.TR_array)
        TRs_lst_old = TRs_lst_new
    TRs_lst_old.sort()
    if len(TRs_lst_old) == 0:
        print("No TR intersection.")
    return TRs_lst_old


def dFC_dict_slice(data, idx_lst):
    data_sliced = {}
    for i, k in enumerate(data):
        if i in idx_lst:
            data_sliced[k] = data[k]
    return data_sliced


def node_info2network(nodes_info):
    node_networks = []
    for info in nodes_info:
        if info[3] == "Network":
            continue
        node_networks.append(info[3])
    return node_networks


def label2network(label):
    """
    returns the network name of a label
    label format: Hemisphere_Network_ID
    """
    return label[label.find("_") + 1 : label.find("_", label.find("_") + 1)]


def node_labels2networks(node_labels):
    node_networks = []
    for label in node_labels:
        node_networks.append(label2network(label))
    return node_networks


def segment_FC(FC, node_networks):
    unique_node_networks = list(set(node_networks))
    segmented = np.zeros_like(FC)
    for network_i in unique_node_networks:
        node_id_i = [idx for idx, value in enumerate(node_networks) if value == network_i]
        for network_j in unique_node_networks:
            node_id_j = [
                idx for idx, value in enumerate(node_networks) if value == network_j
            ]
            segmented[
                node_id_i[0] : node_id_i[-1] + 1, node_id_j[0] : node_id_j[-1] + 1
            ] = np.mean(
                FC[node_id_i[0] : node_id_i[-1] + 1, node_id_j[0] : node_id_j[-1] + 1]
            )
    return segmented


def segment_FC_dict(FC_dict, node_networks):
    segmented_dict = {}
    for key in FC_dict:
        segmented_dict[key] = segment_FC(FC_dict[key], node_networks)
    return segmented_dict


def visualize_conn_mat(
    C,
    axis=None,
    title="",
    cmap="seismic",
    V_MIN=None,
    V_MAX=None,
    node_networks=None,
    title_fontsize=18,
    loc_x=None,
    loc_y=None,
):
    """
    C is (regions, regions)
    you can use loc_x and loc_y to set the location of the image
    loc_x and loc_y are lists of two elements, [start, end]
    """

    if axis is None:
        fig, axis = plt.subplots(1, 1, figsize=(5, 5))

    if node_networks is None:
        axis.set_axis_off()

    if V_MAX is None:
        V_MAX = np.max(np.abs(C))
    if V_MIN is None:
        V_MIN = -1 * V_MAX

    if loc_x is None or loc_y is None:
        im = axis.imshow(
            C,
            interpolation="nearest",
            aspect="equal",
            cmap=cmap,  # 'viridis' or 'jet'
            vmin=V_MIN,
            vmax=V_MAX,
        )
    else:
        im = axis.imshow(
            C,
            interpolation="nearest",
            aspect="equal",
            cmap=cmap,  # 'viridis' or 'jet'
            vmin=V_MIN,
            vmax=V_MAX,
            extent=[loc_x[0], loc_x[1], loc_y[0], loc_y[1]],
        )

    # cluster node networks
    if not node_networks is None:

        # finding unique network names wrt order
        network_names = []
        for node in node_networks:
            if not node in network_names:
                network_names.append(node)
        network_labels = [network_names.index(node) for node in node_networks]

        network_borders = np.argwhere(np.diff(network_labels) != 0)
        ticks_position = []
        last_line_position = 0
        for i in network_borders:
            # 0.5 is the visualization offset of imshow
            line_position = i[0] + 1 - 0.5
            axis.axvline(x=line_position, color="k", linewidth=1)
            axis.axhline(y=line_position, color="k", linewidth=1)
            ticks_position.append((line_position + last_line_position) / 2)
            last_line_position = line_position
        line_position = len(node_networks) + 1 - 0.5
        ticks_position.append((line_position + last_line_position) / 2)

        axis.set_xticks(ticks_position)
        axis.set_yticks(ticks_position)
        axis.set_xticklabels(network_names, rotation=90, fontsize=13)
        axis.set_yticklabels(network_names, fontsize=13)

    axis.set_title(title, fontdict={"fontsize": title_fontsize, "fontweight": "bold"})

    return im


def visualize_conn_mat_dict(
    data,
    title="",
    cmap="seismic",
    normalize=False,
    disp_diag=True,
    label_dict={},
    save_image=False,
    output_root=None,
    axes=None,
    fig=None,
    fix_lim=True,
    center_0=True,
    node_networks=None,
    segmented=False,
):
    """
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
    """

    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3.0})

    sns.set_style("white")

    if node_networks is None:
        fig_width = 25 * (len(data) / 10)
    else:
        fig_width = 60 * (len(data) / 10)
    fig_height = 5

    fig_flag = True
    if axes is None or fig is None:
        fig_flag = False

    if not fig_flag:
        fig, axes = plt.subplots(
            1, len(data), figsize=(fig_width, fig_height), facecolor="w", edgecolor="k"
        )

    if not type(axes) is np.ndarray:
        axes = np.array([axes])

    if show_title:
        fig.suptitle(title, fontsize=20, y=0.98)  # , fontsize=20, size=20

    axes = axes.ravel()

    # normalizing and scale
    conn_mats = list()
    V_MAX_all = None
    for i, key in enumerate(data):

        if segmented:
            C = data[key]
            if not disp_diag:
                C = np.multiply(C, 1 - np.eye(len(C)))
                C = C + np.mean(C.flatten()) * np.eye(len(C))
            C = segment_FC(C, node_networks)
        else:
            C = data[key]

        if normalize:
            C = dFC_mat_normalize(
                C[None, :, :], global_normalization=False, threshold=0.0
            )[0]

        if (not disp_diag) and (not segmented):
            C = np.multiply(C, 1 - np.eye(len(C)))
            C = C + np.mean(C.flatten()) * np.eye(len(C))

        if V_MAX_all is None:
            V_MAX_all = np.max(np.abs(C))
        else:
            V_MAX_all = max(V_MAX_all, np.max(np.abs(C)))

        conn_mats.append(C)
    conn_mats = np.array(conn_mats)

    if np.any(conn_mats < 0) or center_0:
        V_MIN = -1
        V_MAX = 1
    else:
        V_MIN = 0
        V_MAX = 1

    if not fix_lim:
        V_MAX = V_MAX_all
        if np.any(conn_mats < 0) or center_0:
            V_MIN = -1 * V_MAX_all
        else:
            V_MIN = 0

    # plot
    for i, key in enumerate(data):

        C = conn_mats[i, :, :]

        mat_title = key
        if key in label_dict:
            mat_title = label_dict[key]
        im = visualize_conn_mat(
            C,
            axis=axes[i],
            title=mat_title,
            cmap=cmap,
            V_MIN=V_MIN,
            V_MAX=V_MAX,
            node_networks=node_networks,
        )
    if not fig_flag:
        fig.subplots_adjust(
            bottom=0.1,
            top=0.85,
            left=0.1,
            right=0.9,
            # wspace=0.02, \
            # hspace=0.02\
        )

        if not node_networks is None:
            fig.subplots_adjust(wspace=0.55)

    l, b, w, h = axes[-1].get_position().bounds
    if fig_flag:
        cb_ax = fig.add_axes([0.91, b, 0.007, h])
    else:
        cb_ax = fig.add_axes([0.91, b, 0.01, h])
    fig.colorbar(im, cax=cb_ax, shrink=0.8)  # shrink=0.8??

    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title.replace(" ", "_") + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()
    # else:
    #     plt.show()


def visualize_conn_mat_2D_dict(
    data,
    title="",
    cmap="seismic",
    normalize=False,
    disp_diag=True,
    save_image=False,
    output_root=None,
    fix_lim=True,
    center_0=True,
    node_networks=None,
    segmented=False,
):
    """
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
    """
    zip_measure_name = True
    sns.set_context("paper", font_scale=3.5, rc={"lines.linewidth": 3.0})

    sns.set_style("white")

    if node_networks is None:
        fig_width = 30 * (len(data) / 10)
    else:
        fig_width = 55 * (len(data) / 10) + 4
    fig_height = fig_width * 1.0

    fig, axs = plt.subplots(
        len(data),
        len(data),
        figsize=(fig_width, fig_height),
        facecolor="w",
        edgecolor="k",
    )

    if not type(axs) is np.ndarray:
        axs = np.array([axs])

    if show_title:
        fig.suptitle(title, fontsize=25, y=0.98)  # , fontsize=20, size=20

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
                C = dFC_mat_normalize(
                    C[None, :, :], global_normalization=False, threshold=0.0
                )[0]

            if not disp_diag:
                C = np.multiply(C, 1 - np.eye(len(C)))
                C = C + np.mean(C.flatten()) * np.eye(len(C))

            if V_MAX_all is None:
                V_MAX_all = np.max(np.abs(C))
            else:
                V_MAX_all = max(V_MAX_all, np.max(np.abs(C)))

            conn_mats.append(C)
    conn_mats = np.array(conn_mats)

    if np.any(conn_mats < 0) or center_0:
        V_MIN = -1
        V_MAX = 1
    else:
        V_MIN = 0
        V_MAX = 1

    if not fix_lim:
        V_MAX = V_MAX_all
        if np.any(conn_mats < 0) or center_0:
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
                C = dFC_mat_normalize(
                    C[None, :, :], global_normalization=False, threshold=0.0
                )[0]

            if not disp_diag:
                C = np.multiply(C, 1 - np.eye(len(C)))
                C = C + np.mean(C.flatten()) * np.eye(len(C))

            if zip_measure_name:
                mat_title = zip_name(key_i) + "-" + zip_name(key_j)
            else:
                mat_title = key_i + " and " + key_j

            im = visualize_conn_mat(
                C,
                axis=axs[i][j],
                title=mat_title,
                cmap=cmap,
                V_MIN=V_MIN,
                V_MAX=V_MAX,
                node_networks=node_networks,
                title_fontsize=25,
            )

            axs_plotted.append(axs[i][j])

    # remove extra subplots
    for ax in axs.ravel():
        if not ax in axs_plotted:
            ax.set_axis_off()
            ax.xaxis.set_tick_params(which="both", labelbottom=True)

    fig.subplots_adjust(
        bottom=0.1, top=0.95, left=0.1, right=0.9, wspace=0.001, hspace=0.4
    )

    if not node_networks is None:
        fig.subplots_adjust(wspace=0.45, hspace=0.50)

    l, b, w, h = axs[-1][-1].get_position().bounds
    if node_networks is None:
        cb_ax = fig.add_axes([0.91, 0.5 - h / 2, 0.007, h])
    else:
        cb_ax = fig.add_axes([0.91, 0.5 - h / 2, 0.015, h])
    fig.colorbar(im, cax=cb_ax, shrink=0.8)  # shrink=0.8??

    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title.replace(" ", "_") + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()
    else:
        plt.show()


def visualize_FCS(
    measure, normalize=True, fix_lim=True, save_image=False, output_root=None
):

    if measure.FCS == []:
        return

    if normalize:
        D = dFC_dict_normalize(D=measure.FCS_dict, global_normalization=False)
    else:
        D = measure.FCS_dict

    fig_width = 45 * (len(D) / 10)
    fig_height = 8

    fig, axes = plt.subplots(
        2, len(D), figsize=(fig_width, fig_height), facecolor="w", edgecolor="k"
    )

    fig.subplots_adjust(bottom=0.1, top=0.85, left=0.1, right=0.9, wspace=0.1, hspace=0.6)

    # plot mean activity
    for i, mean_act in enumerate(measure.mean_act):
        # setting vmin=-vmax to make 0 correspond to white color
        max_activity = np.max(np.abs(mean_act))
        min_activity = -1 * max_activity
        plot_markers(
            node_values=mean_act,
            node_coords=measure.TS_info["nodes_locs"],
            node_cmap="seismic",
            node_vmin=min_activity,
            node_vmax=max_activity,
            display_mode="z",
            colorbar=False,
            axes=axes[1, i],
        )

    # plot FC pattern
    node_networks = node_info2network(measure.TS_info["nodes_info"])

    visualize_conn_mat_dict(
        data=D,
        node_networks=node_networks,
        title=measure.measure_name + " FCS",
        save_image=save_image,
        axes=axes[0, :],
        fig=fig,
        output_root=output_root,
        disp_diag=False,
        fix_lim=fix_lim,
    )

    fig.subplots_adjust(bottom=0.1, top=0.85, left=0.1, right=0.9, wspace=0.1, hspace=1.0)


"""
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
        G = nx.from_numpy_array(FCS, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(FCS)

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
"""


def dFC_dict_normalize(D, global_normalization=False, threshold=0.0):

    C = list()
    for key in D:
        C.append(D[key])
    C = np.array(C)

    C_z = dFC_mat_normalize(
        C, global_normalization=global_normalization, threshold=threshold
    )

    D_z = {}
    for i, key in enumerate(D):
        D_z[key] = C_z[i, :, :]

    return D_z


def dFC_mat_normalize(C_t, global_normalization=False, threshold=0.0):

    # threshold is ratio of connections wanted to be zero
    C_t_z = deepcopy(C_t)
    if len(C_t_z.shape) < 3:
        C_t_z = np.expand_dims(C_t_z, axis=0)

    if global_normalization:

        # transform the whole abs(dFC mat) to [0, 1]

        signs = np.sign(C_t_z)
        C_t_z = np.abs(C_t_z)

        miN = list()
        for i in range(C_t_z.shape[0]):
            slice = C_t_z[i, :, :]
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0], dtype=bool))]
            miN.append(np.min(slice_non_diag))

        C_t_z = C_t_z - np.min(miN)

        maX = list()
        for i in range(C_t_z.shape[0]):
            slice = C_t_z[i, :, :]
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0], dtype=bool))]
            maX.append(np.max(slice_non_diag))

        if np.max(maX) != 0:
            C_t_z = np.divide(C_t_z, np.max(maX))

        # thresholding
        d = deepcopy(np.ravel(C_t_z))
        d.sort()
        new_threshold = d[int(threshold * len(d))]
        C_t_z = np.multiply(C_t_z, (C_t_z >= new_threshold))
        C_t_z = np.multiply(C_t_z, signs)

    else:

        # transform abs of each time slice to [0, 1]

        signs = np.sign(C_t_z)
        C_t_z = np.abs(C_t_z)

        for i in range(C_t_z.shape[0]):
            slice = C_t_z[i, :, :]
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0], dtype=bool))]
            slice = slice - np.min(slice_non_diag)
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0], dtype=bool))]
            if np.max(slice_non_diag) != 0:
                slice = np.divide(slice, np.max(slice_non_diag))

            # thresholding
            d = deepcopy(np.ravel(slice))
            d.sort()
            new_threshold = d[int(threshold * len(d))]
            slice = np.multiply(slice, (slice >= new_threshold))

            C_t_z[i, :, :] = slice

        C_t_z = np.multiply(C_t_z, signs)

    # removing self connections
    for i in range(C_t_z.shape[1]):
        C_t_z[:, i, i] = np.mean(C_t_z)  # ?????????????????

    return C_t_z


def print_mat(mat, s=0):
    if len(mat.shape) == 1:
        mat = np.expand_dims(mat, axis=0)
    for i in mat:
        print("\t" * s, end=" ")
        for j in i:
            print(f"{j:.2f}", end=" ")
        print()


def print_dict(t, s=0):
    if not isinstance(t, dict) and not isinstance(t, list):
        if isinstance(t, np.ndarray):
            print_mat(t, s)
        else:
            if isinstance(t, float):
                print("\t" * s + f"{t:.2f}")
            else:
                print("\t" * s + str(t))
    else:
        for key in t:
            print("\t" * s + str(key))
            if not isinstance(t, list):
                print_dict(t[key], s + 1)
