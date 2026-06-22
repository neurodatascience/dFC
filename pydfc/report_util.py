# -*- coding: utf-8 -*-
"""
Functions to facilitate reporting.

Created on Feb 5 2025
@author: Mohammad Torabi
"""

import os

import matplotlib.pyplot as plt
import seaborn as sns

################################# Parameters ####################################

fig_dpi = 120
fig_bbox_inches = "tight"
fig_pad = 0.1
show_title = True
save_fig_format = "png"  # pdf, png,

########## Plotting Classification Results Functions ##########


def plot_classification_metrics(
    dataframe, ML_algorithm, pred_metric, title, suffix, output_dir
):
    """
    This function plots these metrics:
    - accuracy
    - balanced accuracy
    - precision
    - recall
    - f1 score (f1)
    - true positive (tp)
    - true negative (tn)
    - false positive (fp)
    - false negative (fn)
    - average precision
    """

    plt.figure(figsize=(10, 5))

    g = sns.pointplot(
        data=dataframe,
        x="dFC method",
        y=f"{ML_algorithm} {pred_metric}",
        hue="group",
        hue_order=["train", "test"],
        errorbar="sd",
        linestyle="none",
        dodge=True,
        capsize=0.1,
    )
    plt.xlabel(g.get_xlabel(), fontweight="bold")
    plt.ylabel(g.get_ylabel(), fontweight="bold")
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    if pred_metric == "balanced accuracy":
        # add a horizontal line at 0.5 corresponding to chance level
        g.axhline(0.5, color="r", linestyle="--")
    if not pred_metric in ["fp", "fn", "tp", "tn"]:
        # set the y-axis upper limit to 1, but not set the lower limit
        g.set(ylim=(None, 1))
    if show_title:
        g.set_title(title, fontdict={"fontsize": 10, "fontweight": "bold"})

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pred_metric_no_space = pred_metric.replace(" ", "_")
    plt.savefig(
        f"{output_dir}/classification_{pred_metric_no_space}_{suffix}.{save_fig_format}",
        dpi=fig_dpi,
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
        format=save_fig_format,
    )

    plt.close()


def plot_clustering_metrics(dataframe, metric, title, suffix, output_dir):
    """
    This function plots these metrics:
    - SI
    """

    plt.figure(figsize=(10, 5))

    g = sns.pointplot(
        data=dataframe,
        x="dFC method",
        y=f"{metric}",
        hue="group",
        hue_order=["train", "test"],
        errorbar="sd",
        linestyle="none",
        dodge=True,
        capsize=0.1,
    )
    plt.xlabel(g.get_xlabel(), fontweight="bold")
    plt.ylabel(g.get_ylabel(), fontweight="bold")
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")

    # set the y-axis upper limit to 1, but not set the lower limit
    g.set(ylim=(None, 1))

    if show_title:
        g.set_title(title, fontdict={"fontsize": 10, "fontweight": "bold"})

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metric_no_space = metric.replace(" ", "_")
    plt.savefig(
        f"{output_dir}/clustering_{metric_no_space}_{suffix}.{save_fig_format}",
        dpi=fig_dpi,
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
        format=save_fig_format,
    )

    plt.close()
