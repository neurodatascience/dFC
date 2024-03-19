# -*- coding: utf-8 -*-
"""
Plotting function for visualizing comparison results.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import math
import os
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as shc
import scipy.spatial.distance as ssd
import seaborn as sns
from matplotlib.colors import ListedColormap
from nilearn.plotting import plot_markers
from sklearn.manifold import TSNE

from ..dfc_utils import visualize_conn_mat_dict

################################# Parameters ####################################

fig_dpi = 120
fig_bbox_inches = "tight"
fig_pad = 0.1
show_title = False
save_fig_format = "png"  # pdf, png,

################################# Plotting Functions ####################################


def title2file_name(title):
    """
    change all spaces in the title to _
    the original string remains unchanged
    """
    return title.replace(" ", "_")


def plot_TS(X, Fs, title=""):
    """
    X = (n_time, n_features)
    """
    if X.shape[1] > 10:
        print("Too many features to plot")
        return
    fig_width = 35
    fig_height = 2 * X.shape[1]
    fig, axes = plt.subplots(
        X.shape[1], 1, figsize=(fig_width, fig_height), facecolor="w", edgecolor="k"
    )
    time = np.arange(0, X.shape[0]) / Fs
    for i in range(0, X.shape[1]):
        axes[i].plot(time, X[:, i], linewidth=4)
        axes[i].set_title("TC" + str(i), fontdict={"fontsize": 15, "fontweight": "bold"})
    plt.suptitle(title)
    plt.xlabel("Time (s)")
    plt.show()
    # if save_image:
    #     folder = output_root[:output_root.rfind('/')]
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)
    #     fig.savefig(output_root+title.replace(" ", "_")+'.'+save_fig_format,
    #         dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
    #     )
    #     plt.close()
    # else:
    #     plt.show()


def plot_sample_dFC(
    D,
    x,
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
    D is a dictionary of dFC samples. each
    key is the name of a dFC matrix (e.g. method
    used for assessing it), and D[key][x] contains the
    the dFC matrix as a numpy ndarray
    """

    num_dFC = len(D)
    names_lst = [key for key in D]
    num_time = len(D[names_lst[0]][x])

    fig_width = 48 * (num_time / 10)
    fig_height = 55 * (num_dFC / 10)

    fig, axes = plt.subplots(
        num_dFC, num_time, figsize=(fig_width, fig_height), facecolor="w", edgecolor="k"
    )

    fig.subplots_adjust(bottom=0.1, top=0.85, left=0.1, right=0.9, wspace=0.5, hspace=0.6)

    for i, dFC_mat_name in enumerate(D):
        visualize_conn_mat_dict(
            data=D[dFC_mat_name][x],
            node_networks=node_networks,
            title=dFC_mat_name,
            cmap=cmap,
            center_0=center_0,
            normalize=normalize,
            fix_lim=fix_lim,
            disp_diag=disp_diag,
            segmented=segmented,
            save_image=False,
            output_root=output_root,
            axes=axes[i, :],
            fig=fig,
        )

    fig.subplots_adjust(bottom=0.1, top=0.85, left=0.1, right=0.9, wspace=0.5, hspace=0.6)

    # set row names
    for i, dFC_mat_name in enumerate(D):
        axes[i, 0].set_ylabel(
            dFC_mat_name, fontdict={"fontsize": 25, "fontweight": "bold"}, rotation=90
        )

    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(
            output_root + title.replace(" ", "_") + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()
    else:
        plt.show()


def pairwise_cat_plots(
    data=None,
    x=None,
    y=None,
    z=None,
    title="",
    label_dict={},
    save_image=False,
    output_root=None,
):
    """
    data is a dictionary with different vars as keys
    if z is specidied, it will be used as a out of distribution
    sample, e.g. actual similarity when plotting randomized
    distribution.
    """

    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3.0})

    row_keys = [key for key in data]
    n_rows = len(row_keys)
    column_keys = [key for key in data[row_keys[-1]]]
    n_columns = len(column_keys)

    sns.set_style("darkgrid")

    fig_width = n_columns * 5
    fig_height = n_rows * 5
    fig, axs = plt.subplots(
        n_rows,
        n_columns,
        figsize=(fig_width, fig_height),
        facecolor="w",
        edgecolor="k",
        sharex=True,
        sharey=True,
    )

    axs_plotted = list()
    for i, key_i in enumerate(data):
        for j, key_j in enumerate(data[key_i]):
            df = pd.DataFrame(data[key_i][key_j])

            if not z is None:
                sns.stripplot(
                    ax=axs[i, j], data=df, x=x, y=z, color="red", jitter=False, size=10
                )
            sns.violinplot(ax=axs[i, j], data=df, x=x, y=y)

            axs[i, j].set_title(
                key_i + "-" + key_j, fontdict={"fontsize": 25, "fontweight": "bold"}
            )

            ## set labels
            ylabel = axs[i, j].get_ylabel()
            if ylabel in label_dict:
                ylabel = label_dict[ylabel]
            axs[i, j].set_ylabel(ylabel, fontdict={"fontsize": 20, "fontweight": "bold"})
            xlabel = axs[i, j].get_xlabel()
            if xlabel in label_dict:
                xlabel = label_dict[xlabel]
            axs[i, j].set_xlabel(xlabel, fontdict={"fontsize": 20, "fontweight": "bold"})
            # set font size of the tick labels and make them bold
            tick_labels = axs[i, j].get_xticklabels() + axs[i, j].get_yticklabels()
            for label in tick_labels:
                label.set_fontweight("bold")

            axs_plotted.append(axs[i, j])

    # remove extra subplots
    for ax in axs.ravel():
        if not ax in axs_plotted:
            ax.set_axis_off()
            ax.xaxis.set_tick_params(which="both", labelbottom=True)

    if show_title:
        plt.suptitle(title, fontsize=15, y=0.90)

    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title2file_name(title) + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()
    else:
        plt.show()


def joint_dist_plot(data, title="", label_dict={}, save_image=False, output_root=None):
    """
    data is a dictionary including list of dFC values
    of each dFC method
    """
    df = pd.DataFrame(data)
    fig_width = 5 * len(data)
    fig_height = 5 * len(data)

    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3.0})

    sns.set_style("darkgrid")

    g = sns.PairGrid(df)

    g.map_diag(sns.histplot)
    g.map_offdiag(sns.histplot)

    g.fig.set_figwidth(fig_width)
    g.fig.set_figheight(fig_height)
    g.fig.subplots_adjust(top=0.95)

    for i in range(g.axes.shape[0]):
        for j in range(g.axes.shape[1]):

            ## set labels
            ylabel = g.axes[i, j].get_ylabel()
            if ylabel in label_dict:
                ylabel = label_dict[ylabel]
            g.axes[i, j].set_ylabel(
                ylabel, fontdict={"fontsize": 25, "fontweight": "bold"}
            )
            xlabel = g.axes[i, j].get_xlabel()
            if xlabel in label_dict:
                xlabel = label_dict[xlabel]
            g.axes[i, j].set_xlabel(
                xlabel, fontdict={"fontsize": 25, "fontweight": "bold"}
            )

            # set font size of the tick labels and make them bold
            tick_labels = g.axes[i, j].get_xticklabels() + g.axes[i, j].get_yticklabels()
            for label in tick_labels:
                label.set_fontweight("bold")

    if show_title:
        plt.suptitle(title, fontsize=50, y=0.98)

    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title2file_name(title) + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()
    else:
        plt.show()


def pairwise_scatter_plots(
    data,
    x,
    y,
    title="",
    hist=False,
    label_dict={},
    equal_axis_lim=False,
    show_x_equal_y=False,
    save_image=False,
    output_root=None,
):
    """
    data is a dictionary with different vars as keys
    """

    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3.0})

    row_keys = [key for key in data]
    n_rows = len(row_keys)
    column_keys = [key for key in data[row_keys[-1]]]
    n_columns = len(column_keys)

    sns.set_style("darkgrid")

    fig_width = n_columns * 5
    fig_height = n_rows * 5
    fig, axs = plt.subplots(
        n_rows,
        n_columns,
        figsize=(fig_width, fig_height),
        facecolor="w",
        edgecolor="k",
        sharex=True,
        sharey=True,
    )

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
        min_lim = min_lim - lim_L * 0.1
        max_lim = max_lim + lim_L * 0.1

    axs_plotted = list()
    for i, key_i in enumerate(data):
        for j, key_j in enumerate(data[key_i]):
            df = pd.DataFrame(data[key_i][key_j])
            if hist:
                g = sns.histplot(ax=axs[i, j], data=df, x=x, y=y, bins=50)
            else:
                g = sns.scatterplot(ax=axs[i, j], data=df, x=x, y=y, s=50)
            axs[i, j].set_title(
                key_i + "-" + key_j, fontdict={"fontsize": 25, "fontweight": "bold"}
            )

            ## set labels and font sizes
            ylabel = g.get_ylabel()
            if ylabel in label_dict:
                ylabel = label_dict[ylabel]
            g.set_ylabel(ylabel, fontdict={"fontsize": 18, "fontweight": "bold"})
            xlabel = g.get_xlabel()
            if xlabel in label_dict:
                xlabel = label_dict[xlabel]
            g.set_xlabel(xlabel, fontdict={"fontsize": 18, "fontweight": "bold"})
            g.tick_params(axis="x", which="major", labelsize=18)
            g.tick_params(axis="y", which="major", labelsize=18)
            tick_labels = g.get_xticklabels() + g.get_yticklabels()
            for label in tick_labels:
                label.set_fontweight("bold")

            # equal x_lim and y_lim
            if equal_axis_lim:
                axs[i, j].set_xlim(min_lim, max_lim)
                axs[i, j].set_ylim(min_lim, max_lim)

            # y=x line
            if show_x_equal_y:
                X_plot = np.linspace(min_lim, max_lim, 100)
                Y_plot = X_plot
                axs[i, j].plot(X_plot, Y_plot, color="r")

            axs_plotted.append(axs[i, j])
    # remove extra subplots
    for ax in axs.ravel():
        if not ax in axs_plotted:
            ax.set_axis_off()
            ax.xaxis.set_tick_params(which="both", labelbottom=True)

    if show_title:
        plt.suptitle(title, fontsize=15, y=0.90)

    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title2file_name(title) + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()
    else:
        plt.show()


def scatter_plot(
    data,
    x,
    y,
    labels=None,
    hue=None,
    title="",
    hist=False,
    label_dict={},
    equal_axis_lim=False,
    show_x_equal_y=False,
    c=0.25,
    save_image=False,
    output_root=None,
):
    """
    data is a dictionary with different vars as keys
    c determines how far the annotation will be from dots
    """
    df = pd.DataFrame(data)

    sns.set_context("paper", font_scale=2.5, rc={"lines.linewidth": 3.0})

    fig_width = 20
    fig_height = 20
    plt.figure(figsize=(fig_width, fig_height))
    sns.set_style("darkgrid")
    if hist:
        g = sns.histplot(data=df, x=x, y=y, hue=hue)
    else:
        g = sns.scatterplot(data=df, x=x, y=y, s=100, hue=hue)

    ## set labels and font sizes
    ylabel = g.get_ylabel()
    if ylabel in label_dict:
        ylabel = label_dict[ylabel]
    g.set_ylabel(ylabel, fontdict={"fontsize": 35, "fontweight": "bold"})
    xlabel = g.get_xlabel()
    if xlabel in label_dict:
        xlabel = label_dict[xlabel]
    g.set_xlabel(xlabel, fontdict={"fontsize": 35, "fontweight": "bold"})
    g.tick_params(axis="x", which="major", labelsize=30)
    g.tick_params(axis="y", which="major", labelsize=30)
    tick_labels = g.get_xticklabels() + g.get_yticklabels()
    for label in tick_labels:
        label.set_fontweight("bold")

    # equal x_lim and y_lim
    if equal_axis_lim:
        min_lim = np.minimum(df[x].min(), df[y].min())
        max_lim = np.maximum(df[x].max(), df[y].max())
        lim_L = max_lim - min_lim
        min_lim = min_lim - lim_L * 0.1
        max_lim = max_lim + lim_L * 0.1
        g.set_xlim(min_lim, max_lim)
        g.set_ylim(min_lim, max_lim)

    # y=x line
    if show_x_equal_y:
        min_lim = np.minimum(df[x].min(), df[y].min())
        max_lim = np.maximum(df[x].max(), df[y].max())
        lim_L = max_lim - min_lim
        min_lim = min_lim - lim_L * 0.1
        max_lim = max_lim + lim_L * 0.1
        X_plot = np.linspace(min_lim, max_lim, 100)
        Y_plot = X_plot
        plt.plot(X_plot, Y_plot, color="r")

    if (not labels is None) and (not hist):
        # the labels are located smartly
        # the direction will be away from mean
        # the distance will be inverse proportional to
        # distance from mean
        mid_x = (np.max(df[x]) + np.min(df[x])) / 2
        mid_y = (np.max(df[y]) + np.min(df[y])) / 2
        x_range = max(np.max(df[x]) - mid_x, mid_x - np.min(df[x]))
        y_range = max(np.max(df[y]) - mid_y, mid_y - np.min(df[y]))
        distance_from_mean_range = math.sqrt(x_range**2 + y_range**2)
        for i in range(len(df[x])):
            distance_from_mean = math.sqrt(
                (df[x][i] - mid_x) ** 2 + (df[y][i] - mid_y) ** 2
            )
            text_x = (
                df[x][i]
                + c
                * np.sign(df[x][i] - mid_x)
                * np.abs(df[x][i] - mid_x)
                * (distance_from_mean_range - distance_from_mean)
                / distance_from_mean
            )
            text_y = (
                df[y][i]
                + c
                * np.sign(df[y][i] - mid_y)
                * np.abs(df[y][i] - mid_y)
                * (distance_from_mean_range - distance_from_mean)
                / distance_from_mean
            )
            plt.text(
                x=text_x,
                y=text_y,
                s=df[labels][i],
                fontdict=dict(color="black", size=14, weight="bold"),
            )
            plt.plot([df[x][i], text_x], [df[y][i], text_y], "k", linewidth=0.5)

    if show_title:
        plt.title(title, fontsize=15)

    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title2file_name(title) + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()
    else:
        plt.show()


def cat_plot(
    data,
    x,
    y,
    kind="bar",
    scale_dist=False,
    log=False,
    title="",
    label_dict={},
    y_lim=None,
    save_image=False,
    output_root=None,
):
    """
    data is a dictionary with different vars as keys
    kind can be = box or violin or bar
    scale_dist is only for kind=='violin'
    """

    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.0})

    sns.set_style("darkgrid")

    df = pd.DataFrame(data)

    fig_width = 2 * len(np.unique(data[x]))
    fig_height = 5

    if kind == "violin" and scale_dist:
        g = sns.catplot(
            data=df,
            x=x,
            y=y,
            kind=kind,
            scale="width",
            # errorbar=("pi", 95)
        )
    elif kind == "bar":
        g = sns.catplot(
            data=df,
            x=x,
            y=y,
            kind=kind,
            width=0.25,
            # errorbar=("pi", 95)
        )
    elif kind == "box":
        g = sns.catplot(
            data=df,
            x=x,
            y=y,
            kind=kind,
            width=0.25,
            fliersize=1.0,
            # errorbar=("pi", 95)
        )
    else:
        g = sns.catplot(
            data=df,
            x=x,
            y=y,
            kind=kind,
            # errorbar=("pi", 95)
        )

    if log:
        plt.yscale("log")

    g.fig.set_figwidth(fig_width)
    g.fig.set_figheight(fig_height)

    ## set labels
    ylabel = g.ax.get_ylabel()
    if ylabel in label_dict:
        ylabel = label_dict[ylabel]
    g.ax.set_ylabel(ylabel, fontdict={"fontsize": 13, "fontweight": "bold"})
    xlabel = g.ax.get_xlabel()
    if xlabel in label_dict:
        xlabel = label_dict[xlabel]
    g.ax.set_xlabel(xlabel, fontdict={"fontsize": 13, "fontweight": "bold"})
    # set font size of the tick labels and make them bold
    tick_labels = g.ax.get_xticklabels() + g.ax.get_yticklabels()
    for label in tick_labels:
        label.set_fontweight("bold")
    if not y_lim is None:
        g.ax.set_ylim(y_lim)

    if show_title:
        plt.title(title, fontsize=15)
    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title2file_name(title) + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()
    else:
        plt.show()


def visualize_sim_mat(
    data,
    mat_key,
    title="",
    name_lst_key=None,
    cmap="viridis",
    annot=True,
    fmt=2,
    label_dict={},
    show_diag=False,
    show_sig=False,
    no_color=False,
    save_image=False,
    output_root=None,
    axes=None,
    fig=None,
):
    """
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
    """

    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.0})

    sns.set_style("white")

    if no_color:
        cmap = ListedColormap(["white"])

    if name_lst_key is None:
        fig_width = int(25 * (len(data) / 10))
    else:
        fig_width = int(60 * (len(data) / 10) + 1)
    fig_height = 5

    fig_flag = True
    if axes is None or fig is None:
        fig_flag = False

    if not fig_flag:
        fig, axes = plt.subplots(
            1,
            len(data),
            figsize=(fig_width, fig_height),
            facecolor="w",
            edgecolor="k",
            sharey=False,
        )

    if not type(axes) is np.ndarray:
        axes = np.array([axes])

    if show_title:
        fig.suptitle(title, fontsize=20, y=0.98)  # , fontsize=20, size=20

    axes = axes.ravel()

    # normalizing and scale
    sim_mats = list()
    for i, key in enumerate(data):
        sim_mats.append(data[key][mat_key])
    sim_mats = np.array(sim_mats)

    # plot
    for i, key in enumerate(data):

        C = sim_mats[i, :, :]

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
                annot_labels = df.applymap(
                    lambda v: (
                        ""
                        if np.isnan(v)
                        else str(round(v, fmt))
                        + "".join(["*" for t in [0.05, 0.01, 0.001] if v <= t])
                    )
                )
            else:
                annot_labels = df.applymap(
                    lambda v: "" if np.isnan(v) else str(round(v, fmt))
                )
        else:
            annot_labels = False

        # borderlines color
        if no_color:
            linecolor = "black"
            annot_kws = {"weight": "bold"}
        else:
            linecolor = "w"
            annot_kws = {"weight": "bold"}

        im = sns.heatmap(
            C,
            annot=annot_labels,
            annot_kws=annot_kws,
            fmt="",
            cmap=cmap,
            xticklabels=name_lst,
            yticklabels=name_lst,
            ax=axes[i],
            cbar=cbar_flag,
            square=True,
            linewidth=2,
            linecolor=linecolor,
        )
        axis_title = key
        if key in label_dict:
            axis_title = label_dict[key]
        axes[i].set_title(axis_title, fontdict={"fontsize": 18, "fontweight": "bold"})
        im.set_xticklabels(
            im.get_xticklabels(),
            fontdict={"fontsize": 14, "fontweight": "bold"},
            rotation=90,
        )
        im.set_yticklabels(
            im.get_yticklabels(),
            fontdict={"fontsize": 14, "fontweight": "bold"},
            rotation=0,
        )

    if not fig_flag:

        fig.subplots_adjust(
            bottom=0.1,
            top=0.85,
            left=0.1,
            right=0.9,
        )

        if not name_lst is None:
            fig.subplots_adjust(wspace=0.5)

    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title2file_name(title) + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()
    else:
        plt.show()


def distance2Z(dist_mat, method="ward"):
    # convert the redundant n*n square matrix form into a condensed nC2 array
    distArray = ssd.squareform(dist_mat)
    Z = shc.linkage(distArray, method=method)
    return Z


def dist_mat_dendo(
    Z,
    labels,
    distances_CI=None,
    title="",
    save_image=False,
    output_root=None,
):
    """
    if distances_CI is provided, confidence intervals (CI)
    of the distances will be shown. the order should be the same as
    Z
    """

    sns.set_context(
        "paper", font_scale=3.5, rc={"lines.linewidth": 3.0, "font.weight": "bold"}
    )

    sns.set_style("darkgrid")

    width = int(2.5 * len(Z))
    fig = plt.figure(figsize=(width, 5))
    ax = fig.add_subplot(1, 1, 1)
    with mpl.rc_context({"lines.linewidth": 3}):

        dend = shc.dendrogram(Z, distance_sort="ascending", no_plot=False, labels=labels)

        # show confidence interval of distances
        if not distances_CI is None:

            max_y_lim = None
            for i, d in zip(dend["icoord"], dend["dcoord"]):

                # we have to match the distances in dcoord
                # with those in Z, because the orders are not
                # the same
                count = 0
                for idx, clstr in enumerate(Z):
                    if np.isclose(Z[idx][2], d[1]):
                        count += 1
                        Z_CI = distances_CI[idx]

                if count > 1 or count == 0:
                    warnings.warn("Error in finding std of linkage.", UserWarning)

                x = 0.5 * sum(i[1:3])
                y = d[1]
                ci_line_y = np.linspace(y - Z_CI, y + Z_CI, 100)
                ci_line_x = x * np.ones(100)
                # cut start and the end for better
                # visualization
                ci_line_y = ci_line_y[5:-5]
                ci_line_x = ci_line_x[5:-5]

                plt.plot(ci_line_x, ci_line_y, "black")
                plt.plot(x, y - Z_CI, "k_", markersize=15, linewidth=15)
                plt.plot(x, y + Z_CI, "k_", markersize=15, linewidth=15)
                plt.plot(x, y, "wo", markersize=5, mec="k")
                # plt.annotate("%.2g" % y, (x, y), xytext=(15, 13),
                #             fontsize = 11,
                #             fontweight= 'bold',
                #             textcoords='offset points',
                #             va='top', ha='center')
                if max_y_lim is None:
                    max_y_lim = y + Z_CI
                else:
                    max_y_lim = max(y + Z_CI, max_y_lim)
            plt.ylim(0, max_y_lim * 1.1)

    if show_title:
        plt.title(title, fontsize=15)

    # set font size of the tick labels and make them bold
    ax.tick_params(axis="x", which="major", labelsize=15)
    ax.tick_params(axis="y", which="major", labelsize=15)
    tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
    for label in tick_labels:
        label.set_fontweight("bold")

    # save figure
    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title2file_name(title) + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()


def plot_TSNE(
    dist_mat,
    sample_measure_lst,
    color_dict,
    projection="2d",
    title="",
    save_image=False,
    output_root=None,
):

    sns.set_context(
        "paper", font_scale=2.5, rc={"lines.linewidth": 3.0, "lines.markersize": 10.0}
    )

    sns.set_style("darkgrid")

    fig_width = 20
    fig_height = 20

    if projection == "2d":
        X_embedded = TSNE(
            n_components=2,
            learning_rate="auto",
            init="random",
            perplexity=30,
            metric="precomputed",
        ).fit_transform(dist_mat)

        # 2D plot
        plt.figure(figsize=(fig_width, fig_height))
        sns.scatterplot(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            hue=sample_measure_lst,
            palette=color_dict,
            alpha=0.7,
        )
    elif projection == "3d":
        X_embedded = TSNE(
            n_components=3,
            learning_rate="auto",
            init="random",
            perplexity=30,
            metric="precomputed",
        ).fit_transform(dist_mat)

        measures_lst = list(set(sample_measure_lst))
        measures_lst.sort()

        # 3D plot
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(projection="3d")
        sample_measure_array = np.array(sample_measure_lst)
        for measure in measures_lst:
            scatter = ax.scatter(
                X_embedded[sample_measure_array == measure, 0],
                X_embedded[sample_measure_array == measure, 1],
                X_embedded[sample_measure_array == measure, 2],
                c=color_dict[measure],
                label=measure,
            )
        ax.legend()

    if show_title:
        plt.title(title, fontsize=15)

    # save figure
    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title2file_name(title) + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()


def plot_brain_act(act_vec, locs, axes, title="", save_image=False, output_root=""):

    plot_markers(
        node_values=act_vec,
        node_coords=locs,
        node_cmap="hot",
        display_mode="z",
        colorbar=False,
        axes=axes,
    )

    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title2file_name(title) + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()


def visualize_state_TC(
    TC_lst, TRs, state_lst, TC_name_lst, title="", save_image=None, output_root=None
):

    color_lst = ["k", "b", "g", "r"]

    if "on" in state_lst and "off" in state_lst:
        ticks = range(2)
    else:
        ticks = range(1, len(state_lst) + 1)

    plt.figure(figsize=(25, 5))
    for i, TC in enumerate(TC_lst):
        plt.plot(TRs, TC, color_lst[i], linewidth=2)
    plt.xlabel("TR")
    plt.yticks(ticks=ticks, labels=state_lst)
    plt.legend(TC_name_lst)
    if show_title:
        plt.title(title)
    if save_image:
        folder = output_root[: output_root.rfind("/")]
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(
            output_root + title2file_name(title) + "." + save_fig_format,
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()
    # else:
    #     plt.show()

    return
