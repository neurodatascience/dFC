# -*- coding: utf-8 -*-
"""
Functions to facilitate task-based validation.

Created on Oct 25 2023
@author: Mohammad Torabi
"""

import matplotlib.pyplot as plt
import numpy as np
from nilearn import glm
from scipy import signal

from .dfc_utils import TR_intersection, rank_norm, visualize_conn_mat

################################# Preprocessing Functions ####################################


def events_time_to_labels(
    events, TR_mri, num_time_mri, event_types=[], oversampling=50, return_0_1=False
):
    """
    event_types is a list of event types to be considered. If None, 0 and 1s will be returned.
    Assigns the longest event in each TR to that TR (in the interval from last TR to current TR).
    It assumes that the first time point is TR0 which corresponds to [0 sec, TR sec] interval.
    oversampling: number of samples per TR_mri to improve the time resolution of tasks
    """
    assert (
        events[0, 0] == "onset"
    ), "The first column of the events file should be the onset!"
    assert (
        events[0, 1] == "duration"
    ), "The second column of the events file should be the duration!"
    assert (
        events[0, 2] == "trial_type"
    ), "The third column of the events file should be the trial type!"

    Fs = float(1 / TR_mri) * oversampling
    num_time_task = int(num_time_mri * oversampling)
    event_labels = np.zeros((num_time_task, 1))
    for i in range(events.shape[0]):
        # skip the first row which is the header
        if i == 0:
            continue

        if events[i, 2] in event_types:
            start_time = float(events[i, 0])
            end_time = float(events[i, 0]) + float(events[i, 1])
            start_TR = int(np.rint(start_time * Fs))
            end_TR = int(np.rint(end_time * Fs))
            event_labels[start_TR:end_TR] = event_types.index(events[i, 2])

    if return_0_1:
        event_labels = np.multiply(event_labels != 0, 1)

    return event_labels, Fs


################################# Visualization Functions ####################################


def plot_task_dFC(task_labels, dFC_lst, event_types, Fs_mri, TR_step=12):
    """
    task_labels: numpy array of shape (num_time_task, num_event_types) containing the event or task labels
    this function assumes that the task data has the same Fs as the dFC data, i.e. MRI data
    and that the time points of the task data are aligned with the time points of the dFC data
    """
    conn_mat_size = 20
    scale_task_plot = 20

    # plot task_data['event_labels']
    fig = plt.figure(figsize=(50, 200))

    ax = plt.gca()

    time = np.arange(0, task_labels.shape[0]) / Fs_mri
    for i in range(0, task_labels.shape[1]):
        ax.plot(
            time, task_labels[:, i] * scale_task_plot, label=event_types[i], linewidth=4
        )
    plt.legend()
    plt.xlabel("Time (s)")

    comman_TRs = TR_intersection(dFC_lst)
    TRs_dFC = comman_TRs[::TR_step]

    for dFC_id, dFC in enumerate(dFC_lst):
        dFC_mat = rank_norm(dFC.get_dFC_mat(), global_norm=True)
        TR_array = dFC.TR_array
        for i in range(0, len(TR_array), 1):

            C = dFC_mat[i, :, :]
            TR = TR_array[i]
            if not TR in TRs_dFC:
                continue
            visualize_conn_mat(
                C=C,
                axis=ax,
                title="",
                cmap="plasma",
                V_MIN=0,
                V_MAX=None,
                node_networks=None,
                title_fontsize=18,
                loc_x=[TR / Fs_mri - conn_mat_size / 2, TR / Fs_mri + conn_mat_size / 2],
                loc_y=[(1 + dFC_id) * conn_mat_size, (2 + dFC_id) * conn_mat_size],
            )

            x1, y1 = [TR / Fs_mri, TR / Fs_mri], [conn_mat_size, 0]
            ax.plot(x1, y1, color="k", linestyle="-", linewidth=2)

    plt.show()


################################# PCA Functions ####################################

# def BOLD


################################# Prediction Functions ####################################

from sklearn.linear_model import LinearRegression


def linear_reg(X, y):
    """
    X = (n_samples, n_features)
    y = (n_samples, n_targets)
    """
    reg = LinearRegression().fit(X, y)
    print(reg.score(X, y))
    return reg.predict(X)


################################# Validation Functions ####################################


def event_conv_hrf(event_signal, TR_mri, TR_task):
    time_length_HRF = 32.0  # in sec
    hrf_model = "spm"  # 'spm' or 'glover'

    TR_HRF = TR_task
    oversampling = (
        TR_mri / TR_HRF
    )  # more samples per TR than the func data to have a better HRF resolution,  same as for event_labels
    if hrf_model == "glover":
        HRF = glm.first_level.glover_hrf(
            tr=TR_mri, oversampling=oversampling, time_length=time_length_HRF, onset=0.0
        )
    elif hrf_model == "spm":
        HRF = glm.first_level.spm_hrf(
            tr=TR_mri, oversampling=oversampling, time_length=time_length_HRF, onset=0.0
        )

    events_hrf = signal.convolve(HRF, event_signal, mode="full")[: len(event_signal)]

    return events_hrf


def event_labels_conv_hrf(event_labels, TR_mri, TR_task):
    """
    event_labels: event labels including 0 and event ids at the time each event happens
    TR_mri: TR of MRI
    TR_task: TR of task
    assumes that 0 is the resting state

    return: event labels convolved with HRF for each event type
    the convolved event labels have the same length as the event_labels
    event type i convolved with HRF is in events_hrf[:, i-1]
    """

    event_labels = np.array(event_labels)
    L = event_labels.shape[0]
    event_ids = np.unique(event_labels)
    event_ids = event_ids.astype(int)
    events_hrf = np.zeros((L, len(event_ids)))  # 0 is the resting state
    for i, event_id in enumerate(event_ids):
        # 0 is not an event, is the resting state
        if event_id == 0:
            continue
        event_signal = np.zeros(L)
        event_signal[event_labels[:, 0] == event_id] = 1.0

        events_hrf[:, i] = event_conv_hrf(event_signal, TR_mri, TR_task)

    # the time points that are not in any event are considered as resting state
    events_hrf[np.sum(events_hrf[:, 1:], axis=1) == 0.0, 0] = 1.0

    # time_length_task = len(event_labels)*TR_task

    return events_hrf


def downsample_events_hrf(events_hrf, TR_mri, TR_task, method="uniform"):
    """
    method:
        uniform
        resample
        decimate
    no major difference was observed between these methods
    the shape of events_hrf is (num_time_task, num_event_types) or (num_time_task,)
    the shape of the downsampled events_hrf is (num_time_mri, num_event_types)
    """
    if len(events_hrf.shape) == 1:
        flag = 1
        events_hrf = np.expand_dims(events_hrf, axis=1)
    events_hrf_ds = []
    for i in range(events_hrf.shape[1]):
        if method == "uniform":
            events_hrf_ds.append(events_hrf[:: int(TR_mri / TR_task), i])
        elif method == "resample":
            events_hrf_ds.append(
                signal.resample(
                    events_hrf[:, i], int(events_hrf.shape[0] * TR_task / TR_mri)
                )
            )
        elif method == "decimate":
            events_hrf_ds.append(signal.decimate(events_hrf[:, i], int(TR_mri / TR_task)))
    events_hrf_ds = np.array(events_hrf_ds).T

    if flag:
        events_hrf_ds = events_hrf_ds[:, 0]

    return events_hrf_ds


def extract_task_presence(event_labels, TR_task, TR_array, TR_mri, binary=True):
    """
    event_labels: event labels including 0 and event ids at the time each event happens
    TR_task: TR of task
    TR_array: the time points of the dFC data
    TR_mri: TR of MRI

    This function extracts the task presence from the event labels and returns it in the same time points as the dFC data
    It also downsamples the task presence to the time points of the dFC data
    if binary is True, the task presence is binarized using the mean of the task presence
    """

    # event_labels_all_task is all conditions together, rest vs. task times
    event_labels_all_task = np.multiply(event_labels != 0, 1)

    event_labels_all_task_hrf = event_labels_conv_hrf(
        event_labels=event_labels_all_task, TR_mri=TR_mri, TR_task=TR_task
    )

    # keep the task signal of events_hrf_0_1_ds
    if event_labels_all_task_hrf.shape[1] == 1:
        # rest
        # raise error if no task
        raise ValueError("No task signal in the event data")
    else:
        # other tasks
        event_labels_all_task_hrf = event_labels_all_task_hrf[:, 1]

    if binary:
        task_presence = np.where(
            event_labels_all_task_hrf > np.mean(event_labels_all_task_hrf), 1, 0
        )
    else:
        task_presence = event_labels_all_task_hrf

    task_presence = downsample_events_hrf(task_presence, TR_mri, TR_task)

    # some dFC measures (window-based) have a different TR than the task data
    task_presence = task_presence[TR_array]

    return task_presence
