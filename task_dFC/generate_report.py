import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import image, masking, plotting
from nilearn.glm.first_level import FirstLevelModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pydfc import DFC, data_loader, task_utils
from pydfc.dfc_utils import (
    TR_intersection,
    dFC_mat2vec,
    dFC_vec2mat,
    rank_norm,
    visualize_conn_mat_dict,
)
from pydfc.report_util import plot_classification_metrics, plot_clustering_metrics

################################# Parameters ####################################

fig_dpi = 120
fig_bbox_inches = "tight"
fig_pad = 0.1
show_title = True
save_fig_format = "png"  # pdf, png,

#######################################################################################


def load_dFC(dFC_root, subj, task, dFC_id, run=None, session=None):
    """
    Load the dFC results for a given subject, task, dFC_id, run and session.
    """
    if session is None:
        if run is None:
            dFC = np.load(
                f"{dFC_root}/{subj}/dFC_{task}_{dFC_id}.npy", allow_pickle="TRUE"
            ).item()
        else:
            dFC = np.load(
                f"{dFC_root}/{subj}/dFC_{task}_{run}_{dFC_id}.npy", allow_pickle="TRUE"
            ).item()
    else:
        if run is None:
            dFC = np.load(
                f"{dFC_root}/{subj}/{session}/dFC_{session}_{task}_{dFC_id}.npy",
                allow_pickle="TRUE",
            ).item()
        else:
            dFC = np.load(
                f"{dFC_root}/{subj}/{session}/dFC_{session}_{task}_{run}_{dFC_id}.npy",
                allow_pickle="TRUE",
            ).item()

    return dFC


def load_task_data(roi_root, subj, task, run=None, session=None):
    """
    Load the task data for a given subject, task and run.
    """
    if session is None:
        if run is None:
            task_data = np.load(
                f"{roi_root}/{subj}/{subj}_{task}_task-data.npy", allow_pickle="TRUE"
            ).item()
        else:
            task_data = np.load(
                f"{roi_root}/{subj}/{subj}_{task}_{run}_task-data.npy",
                allow_pickle="TRUE",
            ).item()
    else:
        if run is None:
            task_data = np.load(
                f"{roi_root}/{subj}/{session}/{subj}_{session}_{task}_task-data.npy",
                allow_pickle="TRUE",
            ).item()
        else:
            task_data = np.load(
                f"{roi_root}/{subj}/{session}/{subj}_{session}_{task}_{run}_task-data.npy",
                allow_pickle="TRUE",
            ).item()

    return task_data


def get_func_data(fmriprep_root, subj, task, bold_suffix, run=None, session=None):
    if session is None:
        ALL_TASK_FILES = os.listdir(f"{fmriprep_root}/{subj}/func/")
    else:
        ALL_TASK_FILES = os.listdir(f"{fmriprep_root}/{subj}/{session}/func/")

    ALL_TASK_FILES = [
        file_i
        for file_i in ALL_TASK_FILES
        if (bold_suffix in file_i) and (f"_{task}_" in file_i)
    ]

    if not len(ALL_TASK_FILES) >= 1:
        return None

    if run is None:
        task_file = ALL_TASK_FILES[0]
    else:
        task_file = [file_i for file_i in ALL_TASK_FILES if f"_{run}_" in file_i][0]
    if session is None:
        func_file = f"{fmriprep_root}/{subj}/func/{task_file}"
    else:
        func_file = f"{fmriprep_root}/{subj}/{session}/func/{task_file}"

    return func_file


# def plot_anatomical(
#     fmriprep_root,
#     subj,
#     anat_suffix,
#     session=None,
# ):
#     anat_suffix = '_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
#     anat_file = f"{fmriprep_root}/{subj}/anat/{subj}{anat_suffix}"
#     display = plotting.plot_anat(anat_file, title="plot_anat")


# def plot_functional(
#     fmriprep_root,
#     subj,
#     bold_suffix,
#     task,
#     session=None,
#     run=None,
# ):
#     if session is None:
#         if run is None:
#             task_file = f"{subj}_{task}{bold_suffix}"
#         else:
#             task_file = f"{subj}_{task}_{run}{bold_suffix}"
#         func_file = f"{fmriprep_root}/{subj}/func/{task_file}"
#     else:
#         if run is None:
#             task_file = f"{subj}_{session}_{task}{bold_suffix}"
#         else:
#             task_file = f"{subj}_{session}_{task}_{run}{bold_suffix}"
#         func_file = f"{fmriprep_root}/{subj}/{session}/func/{task_file}"

#     # Compute voxel-wise mean functional image across time dimension. Now we have
#     # functional image in 3D assigned in mean_func_img
#     mean_func_img = image.mean_img(func_file)
#     display = plotting.plot_anat(mean_func_img, title="plot_func")


def get_events_df(events, trial_type_label="trial_type", rest_labels=["rest", "Rest"]):
    # find which column is the "onset" in the first row
    onset_idx = np.where(events[0, :] == "onset")[0][0]
    duration_idx = np.where(events[0, :] == "duration")[0][0]
    if trial_type_label is not None:
        trial_type_idx = np.where(events[0, :] == trial_type_label)[0][0]

    # assign the time between active onsets to 'rest'
    events_new = []
    prev_onset = 0.0
    for i in range(1, events.shape[0]):

        if trial_type_label is not None:
            if events[i, trial_type_idx] in rest_labels:
                continue

        current_onset = float(events[i, onset_idx])
        current_duration = float(events[i, duration_idx])
        rest_duration = current_onset - prev_onset
        if rest_duration > 0.0:
            events_new.append([prev_onset, rest_duration, "rest"])
        events_new.append([current_onset, current_duration, "active"])
        prev_onset = current_onset + current_duration

    events_new = np.array(events_new)

    # convert to pandas dataframe
    events_df = pd.DataFrame(events_new, columns=["onset", "duration", "trial_type"])

    return events_df


def plot_glm(
    fmriprep_root,
    roi_root,
    subj,
    task,
    bold_suffix,
    trial_type_label,
    rest_labels,
    output_root,
    run=None,
    session=None,
):

    func_file = get_func_data(
        fmriprep_root=fmriprep_root,
        subj=subj,
        task=task,
        bold_suffix=bold_suffix,
        run=run,
        session=session,
    )
    task_data = load_task_data(roi_root, subj, task, run, session)
    TR_mri = task_data["TR_mri"]

    events_df = get_events_df(
        events=task_data["events"],
        trial_type_label=trial_type_label,
        rest_labels=rest_labels,
    )

    # Make an average
    mean_img = image.mean_img(func_file)
    mask = masking.compute_epi_mask(mean_img)

    # Clean and smooth data
    fmri_img = image.clean_img(func_file, standardize=False)
    fmri_img = image.smooth_img(fmri_img, 5.0)

    fmri_glm = FirstLevelModel(
        t_r=TR_mri,
        drift_model="cosine",
        signal_scaling=False,
        mask_img=mask,
        minimize_memory=False,
    )

    fmri_glm = fmri_glm.fit(fmri_img, events_df)

    z_map = fmri_glm.compute_contrast("active - rest")

    plotting.plot_stat_map(z_map, bg_img=mean_img, threshold=3.1)

    # save the figure
    output_dir = f"{output_root}/subject_results/{subj}/GLM"
    if session is not None:
        output_dir = f"{output_dir}/{session}"
    output_dir = f"{output_dir}/{task}"
    if run is not None:
        output_dir = f"{output_dir}/{run}"
    output_dir = f"{output_dir}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(
        f"{output_dir}/glm.{save_fig_format}",
        dpi=fig_dpi,
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
        format=save_fig_format,
    )

    plt.close()


def plot_roi_signals(
    roi_root,
    subj,
    task,
    start_time,
    end_time,
    output_root,
    nodes_list=range(0, 10),
    session=None,
    run=None,
):
    if session is None:
        if run is None:
            file_name = "{subj_id}_{task}_time-series.npy"
        else:
            file_name = "{subj_id}_{task}_{run}_time-series.npy"
    else:
        if run is None:
            file_name = "{subj_id}_{session}_{task}_time-series.npy"
        else:
            file_name = "{subj_id}_{session}_{task}_{run}_time-series.npy"

    task_data = load_task_data(roi_root, subj, task, run, session)
    TR_mri = task_data["TR_mri"]

    BOLD = data_loader.load_TS(
        data_root=roi_root,
        file_name=file_name,
        subj_id2load=subj,
        task=task,
        run=run,
        session=session,
    )

    time = np.arange(0, BOLD.data.shape[1]) * TR_mri
    start_TR = int(start_time / TR_mri)
    end_TR = int(end_time / TR_mri)
    # keep the figure width proportional to the number of time points
    fig_width = int(2.5 * (end_time - start_time) / 2)
    fig_width = min(fig_width, 500)
    plt.figure(figsize=(fig_width, 5))
    for i in nodes_list:
        plt.plot(time[start_TR:end_TR], BOLD.data[i, start_TR:end_TR], linewidth=4)
    # put vertical lines at the start of each TR
    for TR in range(start_TR, end_TR):
        plt.axvline(x=TR * TR_mri, color="r", linestyle="--")
    # show TR labels on the red lines with a small font and at the top
    for TR in range(start_TR, end_TR):
        plt.text(TR * TR_mri, 1.2, f"TR {TR}", fontsize=8, color="black", ha="center")
    if show_title:
        plt.title("ROI signals")
    plt.xlabel("Time (s)")

    # save the figure
    output_dir = f"{output_root}/subject_results/{subj}/ROI_signals"
    if session is not None:
        output_dir = f"{output_dir}/{session}"
    output_dir = f"{output_dir}/{task}"
    if run is not None:
        output_dir = f"{output_dir}/{run}"
    output_dir = f"{output_dir}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(
        f"{output_dir}/ROI_signals.{save_fig_format}",
        dpi=fig_dpi,
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
        format=save_fig_format,
    )

    plt.close()


def plot_event_labels(
    roi_root,
    subj,
    task,
    start_time,
    end_time,
    output_root,
    run=None,
    session=None,
):
    task_data = load_task_data(roi_root, subj, task, run, session)
    Fs_task = task_data["Fs_task"]
    TR_task = 1 / Fs_task
    # TR_mri = task_data["TR_mri"]

    time = np.arange(0, task_data["event_labels"].shape[0]) / Fs_task
    start_timepoint = int(start_time / TR_task)
    end_timepoint = int(end_time / TR_task)
    # keep the figure width proportional to the number of time points
    fig_width = int(2.5 * (end_time - start_time) / 2)
    fig_width = min(fig_width, 500)
    plt.figure(figsize=(fig_width, 5))
    plt.plot(
        time[start_timepoint:end_timepoint],
        task_data["event_labels"][start_timepoint:end_timepoint],
        linewidth=4,
    )
    plt.title("Event labels")
    plt.xlabel("Time (s)")

    # save the figure
    output_dir = f"{output_root}/subject_results/{subj}/event_labels"
    if session is not None:
        output_dir = f"{output_dir}/{session}"
    output_dir = f"{output_dir}/{task}"
    if run is not None:
        output_dir = f"{output_dir}/{run}"
    output_dir = f"{output_dir}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(
        f"{output_dir}/event_labels.{save_fig_format}",
        dpi=fig_dpi,
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
        format=save_fig_format,
    )

    plt.close()


def plot_task_presence(
    roi_root,
    subj,
    task,
    start_time,
    end_time,
    output_root,
    run=None,
    session=None,
):
    task_data = load_task_data(roi_root, subj, task, run, session)
    Fs_task = task_data["Fs_task"]
    TR_task = 1 / Fs_task
    TR_mri = task_data["TR_mri"]
    Fs_mri = 1 / TR_mri

    task_presence_non_binarized, _ = task_utils.extract_task_presence(
        event_labels=task_data["event_labels"],
        TR_task=TR_task,
        TR_mri=task_data["TR_mri"],
        binary=False,
    )

    task_presence, indices = task_utils.extract_task_presence(
        event_labels=task_data["event_labels"],
        TR_task=TR_task,
        TR_mri=task_data["TR_mri"],
        binary=True,
        binarizing_method="GMM",
    )

    time = np.arange(0, task_presence.shape[0]) / Fs_mri
    start_TR = int(start_time / TR_mri)
    end_TR = int(end_time / TR_mri)
    # keep the figure width proportional to the number of time points in data
    fig_width = int(2.5 * (end_time - start_time) / 2)
    fig_width = min(fig_width, 500)
    plt.figure(figsize=(fig_width, 5))
    plt.plot(
        time[start_TR:end_TR], task_presence_non_binarized[start_TR:end_TR], linewidth=4
    )
    plt.plot(time[start_TR:end_TR], task_presence[start_TR:end_TR], linewidth=4)

    # put vertical lines at the start of each TR
    for TR in range(start_TR, end_TR):
        if TR in indices:
            plt.axvline(x=TR * TR_mri, color="g", linestyle="--")
        else:
            plt.axvline(x=TR * TR_mri, color="r", linestyle="--")
    # show TR labels on the red lines with a small font and at the top
    for TR in range(start_TR, end_TR):
        plt.text(TR * TR_mri, 1.2, f"TR {TR}", fontsize=8, color="black", ha="center")
    plt.title("Task presence")
    plt.xlabel("Time (s)")

    # save the figure
    output_dir = f"{output_root}/subject_results/{subj}/task_presence"
    if session is not None:
        output_dir = f"{output_dir}/{session}"
    output_dir = f"{output_dir}/{task}"
    if run is not None:
        output_dir = f"{output_dir}/{run}"
    output_dir = f"{output_dir}/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(
        f"{output_dir}/task_presence.{save_fig_format}",
        dpi=fig_dpi,
        bbox_inches=fig_bbox_inches,
        pad_inches=fig_pad,
        format=save_fig_format,
    )

    plt.close()


# def plot_FCS():
#     visualize_FCS(
#         measure,
#         normalize=True,
#         fix_lim=False,
#         save_image=save_image,
#         output_root=output_root + "FCS/",
#     )


def plot_dFC_matrices(
    dFC_root,
    subj,
    task,
    start_time,
    end_time,
    output_root,
    run=None,
    session=None,
):
    """
    plot dFC matrices for a given subject, task, run, session, start_time and end_time
    parameters:
    ----------
        dFC_root: str, path to dFC results
        subj: str, subject id
        task: str, task name
        start_time: float, start time in seconds
        end_time: float, end time in seconds
    """
    task_data = load_task_data(roi_root, subj, task, run, session)
    TR_mri = task_data["TR_mri"]

    dFC_lst = list()
    for dFC_id in range(0, 20):  # change this to the number of dFCs you have
        try:
            dFC = load_dFC(dFC_root, subj, task, dFC_id, run, session)
            dFC_lst.append(dFC)
        except Exception:
            pass

    TRs = TR_intersection(dFC_lst)
    start_TR = int(start_time / TR_mri)
    end_TR = int(end_time / TR_mri)
    start_TR_idx = np.where(np.array(TRs) >= start_TR)[0][0]
    end_TR_idx = np.where(np.array(TRs) <= end_TR)[0][-1]
    # if the TR_mri is low which will cause the figure to be too wide,
    # we will only plot a resampled version of the dFC matrices, e.g. to make it the same as TR_mri=2s
    if TR_mri < 2:
        TR_step = int(2 / TR_mri)
        chosen_TRs = TRs[start_TR_idx:end_TR_idx:TR_step]
        # raise warning if the TR_mri is low
        print(
            f"TR_mri is low ({TR_mri}s), the dFC matrices will be resampled to make the figure width reasonable"
        )
    else:
        chosen_TRs = TRs[start_TR_idx:end_TR_idx]

    output_dir = f"{output_root}/subject_results/{subj}/dFC_matrices"
    if session is not None:
        output_dir = f"{output_dir}/{session}"
    output_dir = f"{output_dir}/{task}"
    if run is not None:
        output_dir = f"{output_dir}/{run}"
    output_dir = f"{output_dir}/"

    for dFC in dFC_lst:
        dFC.visualize_dFC(
            TRs=chosen_TRs,
            normalize=False,
            rank_norm=True,
            fix_lim=False,
            save_image=True,
            output_root=output_dir,
        )


def plot_ML_results(
    ML_root,
    output_root,
    task,
    run=None,
    session=None,
    ML_algorithms=["KNN"],
    embedding="PCA",
):
    """
    Plot the ML classification results plus SI score for a given task, run and session.
    parameters:
    ----------
        ML_root: str, path to ML results
        output_root: str, path to save the figures
        task: str, task name
        run: int, run number
        session: str, session name
        ML_algorithms: list of str, list of ML algorithm name (default: KNN, other options: Logistic regression, SVM, Gradient Boosting, RF)
        embedding: str, embedding method (default: PCA, other options: LE)
    """
    # the ML_scores files are saved as ML_scores_classify_{dFC_id}.npy
    # find all the ML_scores files in the directory
    if session is None:
        input_dir = f"{ML_root}/classification"
    else:
        input_dir = f"{ML_root}/classification/{session}"
    ALL_ML_SCORES = os.listdir(input_dir)
    ALL_ML_SCORES = [
        score_file for score_file in ALL_ML_SCORES if "ML_scores_classify" in score_file
    ]
    ALL_ML_SCORES.sort()
    ML_scores = None
    for score_file in ALL_ML_SCORES:
        ML_scores_new = np.load(f"{input_dir}/{score_file}", allow_pickle="TRUE").item()
        ML_scores_new = ML_scores_new["subj_lvl"]
        if ML_scores is None:
            ML_scores = ML_scores_new
        else:
            for key in ML_scores_new.keys():
                ML_scores[key].extend(ML_scores_new[key])

    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.0})

    sns.set_style("darkgrid")

    dataframe = pd.DataFrame(ML_scores)
    if run is not None:
        dataframe = dataframe[dataframe["run"] == run]

    dataframe = dataframe[dataframe["task"] == task]
    dataframe = dataframe[dataframe["embedding"] == embedding]

    # save the figure
    if session is None:
        output_dir = f"{output_root}/group_results/classification"
    else:
        output_dir = f"{output_root}/group_results/classification/{session}"

    metrics = [
        # "accuracy",
        "balanced accuracy",
        "precision",
        "recall",
        # "f1",
        # "tp",
        # "tn",
        # "fp",
        # "fn",
        # "average precision",
    ]

    for ML_algorithm in ML_algorithms:
        if ML_algorithm == "Logistic regression":
            ML_algorithm_name = "LogReg"
        elif ML_algorithm == "SVM":
            ML_algorithm_name = "SVM"
        elif ML_algorithm == "KNN":
            ML_algorithm_name = "KNN"
        elif ML_algorithm == "Random Forest":
            ML_algorithm_name = "RF"
        elif ML_algorithm == "Gradient Boosting":
            ML_algorithm_name = "GBT"

        if run is None:
            suffix = f"{ML_algorithm_name}_{task}_{embedding}"
        else:
            suffix = f"{ML_algorithm_name}_{task}_{run}_{embedding}"

        for metric in metrics:
            plot_classification_metrics(
                dataframe=dataframe,
                ML_algorithm=ML_algorithm,
                pred_metric=metric,
                title=task,
                suffix=suffix,
                output_dir=output_dir,
            )

    # Clustering SI score

    # save the figure
    if run is None:
        suffix = f"{task}_{embedding}"
    else:
        suffix = f"{task}_{run}_{embedding}"

    if session is None:
        output_dir = f"{output_root}/group_results/clustering"
    else:
        output_dir = f"{output_root}/group_results/clustering/{session}"

    plot_clustering_metrics(
        dataframe=dataframe,
        metric="SI",
        title=task,
        suffix=suffix,
        output_dir=output_dir,
    )


def plot_visual_clstr_centroids(
    ML_root,
    output_root,
    session=None,
):
    """ """
    # the centroids files are saved as centroids_{session}_{task}_{run}_{measure_name}.npy
    # find all the centroids files in the directory
    if session is None:
        input_dir = f"{ML_root}/centroids"
    else:
        input_dir = f"{ML_root}/centroids/{session}"

    output_dir = f"{output_root}/group_results/visual_clustering_centroids"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ALL_CENTROID_RESULTS = os.listdir(input_dir)
    ALL_CENTROID_RESULTS = [
        result_file for result_file in ALL_CENTROID_RESULTS if "centroids_" in result_file
    ]
    ALL_CENTROID_RESULTS.sort()

    for result_file in ALL_CENTROID_RESULTS:
        centroids_results = np.load(
            f"{input_dir}/{result_file}", allow_pickle="TRUE"
        ).item()
        centroids_mat = centroids_results["centroids_mat"]
        co_occurrence_matrix = centroids_results["co_occurrence_matrix"]
        cluster_label_percentage = centroids_results["cluster_label_percentage"]
        task_label_percentage = centroids_results["task_label_percentage"]

        # result_file is centroids_{session}_{task}_{run}_{measure_name}.npy
        # suffix is whatever comes after the centroids and before .npy
        suffix = result_file.split("centroids_")[1].split(".npy")[0]

        centroids_dict = {}
        for i, centroid_mat in enumerate(centroids_mat):
            centroids_dict[f"Cluster {i + 1}"] = centroid_mat

        visualize_conn_mat_dict(
            data=centroids_dict,
            title=f"visual-centroids_{suffix}",
            cmap="seismic",
            normalize=True,
            disp_diag=False,
            save_image=True,
            output_root=f"{output_dir}/",
            center_0=True,
            # node_networks=None,
        )

        # plot co-occurrence matrix and cluster label percentage and task label percentage
        # as a seaborn heatmap with numbers in the cells
        # as separate figures

        # plot co-occurrence matrix
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            co_occurrence_matrix,
            annot=True,
            fmt=".0f",
            cmap="Reds",
            cbar_kws={"label": "Co-occurrence"},
            yticklabels=["rest", "task"],
            xticklabels=[str(i + 1) for i in range(co_occurrence_matrix.shape[1])],
        )
        plt.title("Co-occurrence matrix")
        plt.xlabel("Cluster")
        plt.ylabel("Task")
        plt.savefig(
            f"{output_dir}/co-occurrence-matrix_{suffix}.{save_fig_format}",
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()

        # plot cluster label percentage
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            cluster_label_percentage,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            cbar_kws={"label": "Percentage"},
            yticklabels=["rest", "task"],
            xticklabels=[str(i + 1) for i in range(co_occurrence_matrix.shape[1])],
        )
        plt.title("Cluster label percentage")
        plt.xlabel("Cluster")
        plt.ylabel("Task")
        plt.savefig(
            f"{output_dir}/cluster-label-percentage_{suffix}.{save_fig_format}",
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()

        # plot task label percentage
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            task_label_percentage,
            annot=True,
            fmt=".2f",
            cmap="Reds",
            cbar_kws={"label": "Percentage"},
            yticklabels=["rest", "task"],
            xticklabels=[str(i + 1) for i in range(co_occurrence_matrix.shape[1])],
        )
        plt.title("Task label percentage")
        plt.xlabel("Cluster")
        plt.ylabel("Task")
        plt.savefig(
            f"{output_dir}/task-label-percentage_{suffix}.{save_fig_format}",
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()


def plot_task_presence_features(
    ML_root,
    output_root,
    session=None,
    run=None,
):
    """
    Plot the task presence features for a given session and run.
    Features for both with and without HRF are plotted.
    for comparability of tasks, pass the same run number for all tasks
    parameters:
    ----------
        ML_root: str, path to ML results
        output_root: str, path to save the figures
        session: str, session name
        run: int, run number
    """
    if session is None:
        task_features = np.load(
            f"{ML_root}/task_features/task_features.npy", allow_pickle="TRUE"
        ).item()
        task_features_hrf = np.load(
            f"{ML_root}/task_features/task_features_hrf.npy", allow_pickle="TRUE"
        ).item()
    else:
        task_features = np.load(
            f"{ML_root}/task_features/{session}/task_features.npy", allow_pickle="TRUE"
        ).item()
        task_features_hrf = np.load(
            f"{ML_root}/task_features/{session}/task_features_hrf.npy",
            allow_pickle="TRUE",
        ).item()

    sns.set_context("paper", font_scale=1.0, rc={"lines.linewidth": 1.0})

    sns.set_style("darkgrid")

    task_features_df = pd.DataFrame(task_features)
    task_features_hrf_df = pd.DataFrame(task_features_hrf)
    if run is not None:
        task_features_df = task_features_df[task_features_df["run"] == run]
        task_features_hrf_df = task_features_hrf_df[task_features_hrf_df["run"] == run]

    # FEATURES are columns in the dataframe except for 'task' and 'run'
    FEATURES = list(task_features_df.columns)
    FEATURES.remove("task")
    FEATURES.remove("run")

    if session is None:
        output_dir = f"{output_root}/group_results/task_presence_features"
    else:
        output_dir = f"{output_root}/group_results/task_presence_features/{session}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, feature in enumerate(FEATURES):
        plt.figure(figsize=(10, 5))
        g = sns.pointplot(
            data=task_features_df,
            x="task",
            y=feature,
            errorbar="sd",
            linestyle="none",
            dodge=True,
            capsize=0.1,
        )
        plt.xlabel(g.get_xlabel(), fontweight="bold")
        plt.ylabel(g.get_ylabel(), fontweight="bold")
        plt.xticks(fontweight="bold")
        plt.yticks(fontweight="bold")

        # save the figure
        plt.savefig(
            f"{output_dir}/task_presence_features_{feature}.{save_fig_format}",
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()

        plt.figure(figsize=(10, 5))
        g = sns.pointplot(
            data=task_features_hrf_df,
            x="task",
            y=feature,
            errorbar="sd",
            linestyle="none",
            dodge=True,
            capsize=0.1,
        )
        plt.xlabel(g.get_xlabel(), fontweight="bold")
        plt.ylabel(g.get_ylabel(), fontweight="bold")
        plt.xticks(fontweight="bold")
        plt.yticks(fontweight="bold")

        # save the figure
        plt.savefig(
            f"{output_dir}/task_presence_features_hrf_{feature}.{save_fig_format}",
            dpi=fig_dpi,
            bbox_inches=fig_bbox_inches,
            pad_inches=fig_pad,
            format=save_fig_format,
        )
        plt.close()


def create_html_report_subj_results(
    subj,
    SESSIONS,
    TASKS,
    RUNS,
    reports_root,
):
    """
    This function creates an html report for the subject results
    using the generated figures.
    """

    # create html report
    subj_dir = f"{reports_root}/subject_results/{subj}"
    file = open(f"{subj_dir}/report.html", "w")
    file.write("<html>\n")
    file.write("<head>\n")
    file.write(f"<title>Subject {subj} Results</title>\n")
    file.write("</head>\n")
    file.write("<body>\n")
    file.write(f"<h1>Subject {subj} Results</h1>\n")
    for session in SESSIONS:
        if session is not None:
            file.write(f"<h1> {session} </h1>\n")
        for task in TASKS:
            file.write(f"<h1> {task} </h1>\n")
            for run in RUNS[task]:
                if run is not None:
                    file.write(f"<h2> {run} </h2>\n")
                if session is not None:
                    session_task_run_dir = f"{session}/{task}"
                else:
                    session_task_run_dir = f"{task}"
                if run is not None:
                    session_task_run_dir = f"{session_task_run_dir}/{run}"

                img_height = 100

                # display GLM
                glm_img = f"{subj_dir}/GLM/{session_task_run_dir}/glm.png"
                if os.path.exists(glm_img):
                    img = plt.imread(glm_img)
                    height, width, _ = img.shape
                    # change the width so that height equals img_height
                    width = int(width * img_height / height)
                    # replace the path to the image with a relative path
                    glm_img = glm_img.replace(subj_dir, ".")
                    file.write(
                        f"<img src='{glm_img}' alt='GLM' width='{width}' height='{img_height}'>\n"
                    )
                    file.write("<br>\n")

                # display ROI signals
                ROI_signals_img = (
                    f"{subj_dir}/ROI_signals/{session_task_run_dir}/ROI_signals.png"
                )
                if os.path.exists(ROI_signals_img):
                    img = plt.imread(ROI_signals_img)
                    height, width, _ = img.shape
                    # change the width so that height equals img_height
                    width = int(width * img_height / height)
                    # replace the path to the image with a relative path
                    ROI_signals_img = ROI_signals_img.replace(subj_dir, ".")
                    file.write(
                        f"<img src='{ROI_signals_img}' alt='ROI signals' width='{width}' height='{img_height}'>\n"
                    )
                    file.write("<br>\n")

                # display event labels
                event_labels_img = (
                    f"{subj_dir}/event_labels/{session_task_run_dir}/event_labels.png"
                )
                if os.path.exists(event_labels_img):
                    img = plt.imread(event_labels_img)
                    height, width, _ = img.shape
                    # change the width so that height equals img_height
                    width = int(width * img_height / height)
                    # replace the path to the image with a relative path
                    event_labels_img = event_labels_img.replace(subj_dir, ".")
                    file.write(
                        f"<img src='{event_labels_img}' alt='Event labels' width='{width}' height='{img_height}'>\n"
                    )
                    file.write("<br>\n")

                # display task presence
                task_presence_img = (
                    f"{subj_dir}/task_presence/{session_task_run_dir}/task_presence.png"
                )
                if os.path.exists(task_presence_img):
                    img = plt.imread(task_presence_img)
                    height, width, _ = img.shape
                    # change the width so that height equals img_height
                    width = int(width * img_height / height)
                    # replace the path to the image with a relative path
                    task_presence_img = task_presence_img.replace(subj_dir, ".")
                    file.write(
                        f"<img src='{task_presence_img}' alt='Task presence' width='{width}' height='{img_height}'>\n"
                    )
                    file.write("<br>\n")

                # display dFC matrices
                img_height = 45
                # for dFC matrices find all png files in the directory
                dFC_matrices_dir = f"{subj_dir}/dFC_matrices/{session_task_run_dir}"
                if os.path.exists(dFC_matrices_dir):
                    for file_name in os.listdir(dFC_matrices_dir):
                        if file_name.endswith(".png"):
                            file.write(f"<h3>{file_name[:file_name.find('_dFC')]}</h3>\n")
                            dFC_matrices_img = f"{dFC_matrices_dir}/{file_name}"
                            # get the original size of the image
                            img = plt.imread(dFC_matrices_img)
                            height, width, _ = img.shape
                            # change the width so that height equals img_height
                            width = int(width * img_height / height)
                            # replace the path to the image with a relative path
                            dFC_matrices_img = dFC_matrices_img.replace(subj_dir, ".")
                            file.write(
                                f"<img src='{dFC_matrices_img}' alt='{file_name}' width='{width}' height='{img_height}'>\n"
                            )
                            file.write("<br>\n")

    file.write("</body>\n")
    file.write("</html>\n")
    file.close()


def create_html_report_group_results(
    SESSIONS,
    TASKS,
    RUNS,
    reports_root,
):
    """
    This function creates an html report for the group results
    using the generated figures.
    """
    # create html report
    group_dir = f"{reports_root}/group_results"
    file = open(f"{group_dir}/report.html", "w")
    file.write("<html>\n")
    file.write("<head>\n")
    file.write("<title>Group Results</title>\n")
    file.write("</head>\n")
    file.write("<body>\n")
    file.write("<h1>Group Results</h1>\n")

    # task presence features
    img_height = 300
    file.write("<h1>Task Presence Features</h1>\n")
    for session in SESSIONS:
        if session is not None:
            file.write(f"<h1> {session} </h1>\n")
        # display task presence features
        if session is not None:
            task_presence_features_dir = f"{group_dir}/task_presence_features/{session}"
        else:
            task_presence_features_dir = f"{group_dir}/task_presence_features"

        for condition in ["with_HRF", "without_HRF"]:
            file.write(f"<h2>{condition}</h2>\n")
            # find all png files in the directory
            for file_name in os.listdir(task_presence_features_dir):
                if file_name.endswith(".png"):
                    if (condition == "with_HRF" and "hrf" not in file_name) or (
                        condition == "without_HRF" and "hrf" in file_name
                    ):
                        continue
                    task_presence_features_img = (
                        f"{task_presence_features_dir}/{file_name}"
                    )
                    # get the original size of the image
                    img = plt.imread(task_presence_features_img)
                    height, width, _ = img.shape
                    # change the width so that height equals img_height
                    width = int(width * img_height / height)
                    # replace the path to the image with a relative path
                    task_presence_features_img = task_presence_features_img.replace(
                        group_dir, "."
                    )
                    file.write(
                        f"<img src='{task_presence_features_img}' alt='Task presence features' width='{width}' height='{img_height}'>\n"
                    )

            file.write("<br>\n")

    file.write("<br>\n")

    # classification results
    metrics = [
        # "accuracy",
        "balanced accuracy",
        "precision",
        "recall",
        # "f1",
        # "tp",
        # "tn",
        # "fp",
        # "fn",
        # "average precision",
    ]
    classification_models = {"LogReg": "Logistic Regression", "SVM": "SVM"}
    img_height = 300
    file.write("<h1>Classification Results</h1>\n")
    for session in SESSIONS:
        if session is not None:
            file.write(f"<h1> {session} </h1>\n")
        for task in TASKS:
            file.write(f"<h1> {task} </h1>\n")
            for run in RUNS[task]:
                if run is not None:
                    file.write(f"<h2> {run} </h2>\n")
                if session is not None:
                    classification_dir = f"{group_dir}/classification/{session}"
                else:
                    classification_dir = f"{group_dir}/classification"

                for model in classification_models:
                    file.write(f"<h3>{classification_models[model]}</h3>\n")
                    for embedding in ["PCA", "LE"]:
                        file.write(f"<h3>{embedding}</h3>\n")
                        for metric in metrics:
                            metric_no_space = metric.replace(" ", "_")
                            if run is None:
                                classification_img = f"{classification_dir}/classification_{metric_no_space}_{model}_{task}_{embedding}.png"
                            else:
                                classification_img = f"{classification_dir}/classification_{metric_no_space}_{model}_{task}_{run}_{embedding}.png"
                            if os.path.exists(classification_img):
                                img = plt.imread(classification_img)
                                height, width, _ = img.shape
                                # change the width so that height equals img_height
                                width = int(width * img_height / height)
                                # replace the path to the image with a relative path
                                classification_img = classification_img.replace(
                                    group_dir, "."
                                )
                                file.write(
                                    f"<img src='{classification_img}' alt='Classification results' width='{width}' height='{img_height}'>\n"
                                )

                        file.write("<br>\n")

    # clustering results
    img_height = 300
    file.write("<h1>Clustering Results</h1>\n")
    for session in SESSIONS:
        if session is not None:
            file.write(f"<h1> {session} </h1>\n")
        for task in TASKS:
            file.write(f"<h1> {task} </h1>\n")
            for run in RUNS[task]:
                if run is not None:
                    file.write(f"<h2> {run} </h2>\n")
                if session is not None:
                    clustering_dir = f"{group_dir}/clustering/{session}"
                else:
                    clustering_dir = f"{group_dir}/clustering"

                for embedding in ["PCA", "LE"]:
                    file.write(f"<h3>{embedding}</h3>\n")
                    # display clustering ARI results
                    if run is None:
                        clustering_img = (
                            f"{clustering_dir}/clustering_SI_{task}_{embedding}.png"
                        )
                    else:
                        clustering_img = (
                            f"{clustering_dir}/clustering_SI_{task}_{run}_{embedding}.png"
                        )
                    if os.path.exists(clustering_img):
                        img = plt.imread(clustering_img)
                        height, width, _ = img.shape
                        # change the width so that height equals img_height
                        width = int(width * img_height / height)
                        # replace the path to the image with a relative path
                        clustering_img = clustering_img.replace(group_dir, ".")
                        file.write(
                            f"<img src='{clustering_img}' alt='Clustering results' width='{width}' height='{img_height}'>\n"
                        )

                        file.write("<br>\n")

    # display visual clustering centroids
    img_height = 300
    file.write("<h1>Visual Clustering Centroids</h2>\n")
    # find all png files in the directory
    visual_clustering_centroids_dir = f"{group_dir}/visual_clustering_centroids"
    for session in SESSIONS:
        if session is not None:
            file.write(f"<h3> {session} </h3>\n")
        for task in TASKS:
            file.write(f"<h3> {task} </h3>\n")
            for run in RUNS[task]:
                if run is not None:
                    file.write(f"<h3> {run} </h3>\n")

                # visual-centroids_{session}_{task}_{run}_{measure_name}.png
                all_centroids_img_files = os.listdir(visual_clustering_centroids_dir)
                all_centroids_img_files = [
                    centroids_img_file
                    for centroids_img_file in all_centroids_img_files
                    if "visual-centroids" in centroids_img_file
                    and f"_{task}" in centroids_img_file
                ]
                if session is not None:
                    all_centroids_img_files = [
                        centroids_img_file
                        for centroids_img_file in all_centroids_img_files
                        if f"_{session}" in centroids_img_file
                    ]
                if run is not None:
                    all_centroids_img_files = [
                        centroids_img_file
                        for centroids_img_file in all_centroids_img_files
                        if f"_{run}" in centroids_img_file
                    ]
                all_centroids_img_files.sort()

                for centroids_img_file in all_centroids_img_files:
                    # iterate over centroids images of different measures
                    centroid_img = (
                        f"{visual_clustering_centroids_dir}/{centroids_img_file}"
                    )
                    measure_name = centroids_img_file.split("_")[-1].split(".")[0]
                    file.write(f"<h3>{measure_name}</h3>\n")
                    # get the original size of the image
                    if os.path.exists(centroid_img):
                        img = plt.imread(centroid_img)
                        height, width, _ = img.shape
                        # change the width so that height equals img_height
                        width = int(width * img_height / height)
                        # replace the path to the image with a relative path
                        centroid_img = centroid_img.replace(group_dir, ".")
                        file.write(
                            f"<img src='{centroid_img}' alt='Visual clustering centroids' width='{width}' height='{img_height}'>\n"
                        )

                    # visual-centroids_{suffix}.png
                    suffix = centroids_img_file[
                        centroids_img_file.find("visual-centroids_") + 17 : -4
                    ]

                    # display co-occurrence matrix
                    co_occurrence_matrix_img = f"{visual_clustering_centroids_dir}/co-occurrence-matrix_{suffix}.png"
                    if os.path.exists(co_occurrence_matrix_img):
                        img = plt.imread(co_occurrence_matrix_img)
                        height, width, _ = img.shape
                        # change the width so that height equals img_height
                        width = int(width * img_height / height)
                        # replace the path to the image with a relative path
                        co_occurrence_matrix_img = co_occurrence_matrix_img.replace(
                            group_dir, "."
                        )
                        file.write(
                            f"<img src='{co_occurrence_matrix_img}' alt='Co-occurrence matrix' width='{width}' height='{img_height}'>\n"
                        )

                    # display cluster label percentage
                    cluster_label_percentage_img = f"{visual_clustering_centroids_dir}/cluster-label-percentage_{suffix}.png"
                    if os.path.exists(cluster_label_percentage_img):
                        img = plt.imread(cluster_label_percentage_img)
                        height, width, _ = img.shape
                        # change the width so that height equals img_height
                        width = int(width * img_height / height)
                        # replace the path to the image with a relative path
                        cluster_label_percentage_img = (
                            cluster_label_percentage_img.replace(group_dir, ".")
                        )
                        file.write(
                            f"<img src='{cluster_label_percentage_img}' alt='Cluster label percentage' width='{width}' height='{img_height}'>\n"
                        )

                    # display task label percentage
                    task_label_percentage_img = f"{visual_clustering_centroids_dir}/task-label-percentage_{suffix}.png"
                    if os.path.exists(task_label_percentage_img):
                        img = plt.imread(task_label_percentage_img)
                        height, width, _ = img.shape
                        # change the width so that height equals img_height
                        width = int(width * img_height / height)
                        # replace the path to the image with a relative path
                        task_label_percentage_img = task_label_percentage_img.replace(
                            group_dir, "."
                        )
                        file.write(
                            f"<img src='{task_label_percentage_img}' alt='Task label percentage' width='{width}' height='{img_height}'>\n"
                        )

                    file.write("<br>\n")

    file.write("</body>\n")
    file.write("</html>\n")
    file.close()


#######################################################################################
if __name__ == "__main__":
    # argparse
    HELPTEXT = """
    Script to generate a report of subject results.
    """

    parser = argparse.ArgumentParser(description=HELPTEXT)

    parser.add_argument("--dataset_info", type=str, help="path to dataset info file")
    parser.add_argument("--subj_list", type=str, help="path to subject list file")

    args = parser.parse_args()

    dataset_info_file = args.dataset_info
    subj_list_file = args.subj_list

    # Read dataset info
    with open(dataset_info_file, "r") as f:
        dataset_info = json.load(f)

    # Read subject list file, a txt file with one subject id per line
    with open(subj_list_file, "r") as f:
        SUBJECTS = f.read().splitlines()

    TASKS = dataset_info["TASKS"]
    if "RUNS" in dataset_info:
        RUNS = dataset_info["RUNS"]
    else:
        RUNS = None
    if RUNS is None:
        RUNS = {task: [None] for task in TASKS}

    if "SESSIONS" in dataset_info:
        SESSIONS = dataset_info["SESSIONS"]
    else:
        SESSIONS = None
    if SESSIONS is None:
        SESSIONS = [None]

    if "{dataset}" in dataset_info["main_root"]:
        main_root = dataset_info["main_root"].replace(
            "{dataset}", dataset_info["dataset"]
        )
    else:
        main_root = dataset_info["main_root"]

    if "{main_root}" in dataset_info["fmriprep_root"]:
        fmriprep_root = dataset_info["fmriprep_root"].replace("{main_root}", main_root)
    elif "{dataset}" in dataset_info["fmriprep_root"]:
        fmriprep_root = dataset_info["fmriprep_root"].replace(
            "{dataset}", dataset_info["dataset"]
        )
    else:
        fmriprep_root = dataset_info["fmriprep_root"]

    if "{main_root}" in dataset_info["roi_root"]:
        roi_root = dataset_info["roi_root"].replace("{main_root}", main_root)
    else:
        roi_root = dataset_info["roi_root"]

    if "{main_root}" in dataset_info["dFC_root"]:
        dFC_root = dataset_info["dFC_root"].replace("{main_root}", main_root)
    else:
        dFC_root = dataset_info["dFC_root"]

    if "{main_root}" in dataset_info["ML_root"]:
        ML_root = dataset_info["ML_root"].replace("{main_root}", main_root)
    else:
        ML_root = dataset_info["ML_root"]

    if "{main_root}" in dataset_info["reports_root"]:
        reports_root = dataset_info["reports_root"].replace("{main_root}", main_root)
    else:
        reports_root = dataset_info["reports_root"]

    print("Generating report...")

    # Generate report only 3 subjects
    SUBJECTS = SUBJECTS[:3]

    start_time = 0
    end_time = 200

    for subj in SUBJECTS:
        for session in SESSIONS:
            for task in TASKS:
                for run in RUNS[task]:

                    try:
                        plot_dFC_matrices(
                            dFC_root=dFC_root,
                            subj=subj,
                            task=task,
                            start_time=start_time,
                            end_time=end_time,
                            output_root=reports_root,
                            run=run,
                            session=session,
                        )
                    except Exception as e:
                        print(f"Error in plotting dFC matrices: {e}")

                    # try:
                    #     plot_glm(
                    #         fmriprep_root=fmriprep_root,
                    #         roi_root=roi_root,
                    #         subj=subj,
                    #         task=task,
                    #         bold_suffix=dataset_info["bold_suffix"],
                    #         trial_type_label=dataset_info["trial_type_label"],
                    #         rest_labels=dataset_info["rest_labels"],
                    #         output_root=reports_root,
                    #         run=run,
                    #         session=session,
                    #     )
                    # except Exception as e:
                    #     print(f"Error in plotting GLM: {e}")

                    try:
                        plot_roi_signals(
                            roi_root=roi_root,
                            subj=subj,
                            task=task,
                            start_time=start_time,
                            end_time=end_time,
                            nodes_list=range(0, 10),
                            output_root=reports_root,
                            run=run,
                            session=session,
                        )
                    except Exception as e:
                        print(f"Error in plotting ROI signals: {e}")

                    try:
                        plot_event_labels(
                            roi_root=roi_root,
                            subj=subj,
                            task=task,
                            start_time=start_time,
                            end_time=end_time,
                            output_root=reports_root,
                            run=run,
                            session=session,
                        )
                    except Exception as e:
                        print(f"Error in plotting event labels: {e}")

                    try:
                        plot_task_presence(
                            roi_root=roi_root,
                            subj=subj,
                            task=task,
                            start_time=start_time,
                            end_time=end_time,
                            output_root=reports_root,
                            run=run,
                            session=session,
                        )
                    except Exception as e:
                        print(f"Error in plotting task presence: {e}")

                    # try:
                    #     plot_dFC_clustering(
                    #         dFC_root=dFC_root,
                    #         subj=subj,
                    #         task=task,
                    #         start_time=start_time,
                    #         end_time=end_time,
                    #         output_root=reports_root,
                    #         run=run,
                    #         session=session,
                    #         normalize_dFC=True,
                    #     )
                    # except Exception as e:
                    #     print(f"Error in plotting dFC clustering: {e}")
        # create html report
        try:
            create_html_report_subj_results(
                subj=subj,
                SESSIONS=SESSIONS,
                TASKS=TASKS,
                RUNS=RUNS,
                reports_root=reports_root,
            )
        except Exception as e:
            print(f"Error in creating html report for subject results: {e}")

    # plot group results
    # find the common run number for all tasks for task presence features
    common_run = None
    for task in TASKS:
        if common_run is None:
            common_run = RUNS[task][0]
        else:
            if RUNS[task][0] != common_run:
                common_run = None
                # raise warning
                print(
                    "Warning: Tasks have different run numbers for task presence features!"
                )
                break

    for session in SESSIONS:
        try:
            plot_task_presence_features(
                ML_root=ML_root,
                output_root=reports_root,
                session=session,
                run=common_run,
            )
        except Exception as e:
            print(f"Error in plotting task presence features: {e}")

        try:
            plot_visual_clstr_centroids(
                ML_root=ML_root,
                output_root=reports_root,
                session=session,
            )
        except Exception as e:
            print(f"Error in plotting visual clustering centroids: {e}")

        for task in TASKS:
            for run in RUNS[task]:
                for embedding in ["PCA", "LE"]:
                    try:
                        plot_ML_results(
                            ML_root=ML_root,
                            output_root=reports_root,
                            task=task,
                            run=run,
                            session=session,
                            ML_algorithms=["SVM", "Logistic regression"],
                            embedding=embedding,
                        )
                    except Exception as e:
                        print(f"Error in plotting ML results for {embedding}: {e}")

    # create html report
    try:
        create_html_report_group_results(
            SESSIONS=SESSIONS,
            TASKS=TASKS,
            RUNS=RUNS,
            reports_root=reports_root,
        )
    except Exception as e:
        print(f"Error in creating html report for group results: {e}")

    print("Report generated successfully!")

#######################################################################################
