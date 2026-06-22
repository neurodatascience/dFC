# -*- coding: utf-8 -*-
"""
Functions to facilitate task-based validation.

Created on Oct 25 2023
@author: Mohammad Torabi
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from nilearn import glm
from scipy import signal
from sklearn.mixture import GaussianMixture
from statsmodels.tsa.stattools import acf

from .dfc_utils import TR_intersection, rank_norm, visualize_conn_mat

################################# Preprocessing Functions ####################################


def events_time_to_labels(
    events,
    TR_mri,
    num_time_mri,
    event_types=None,
    oversampling=50,
    trial_type_label="trial_type",
    rest_labels=["rest", "Rest"],
    return_0_1=False,
):
    """
    event_types is a list of event types to be considered. If None, it will found based on events.
    Assigns the longest event in each TR to that TR (in the interval from last TR to current TR).
    It assumes that the first time point is TR0 which corresponds to [0 sec, TR sec] interval.
    oversampling: number of samples per TR_mri to improve the time resolution of tasks

    if trial_type_label is None, we use event type "unknown" as the trial type
    """

    # find which column is the "onset" in the first row
    onset_idx = np.where(events[0, :] == "onset")[0][0]
    duration_idx = np.where(events[0, :] == "duration")[0][0]
    if trial_type_label is not None:
        trial_type_idx = np.where(events[0, :] == trial_type_label)[0][0]

    assert (
        events[0, onset_idx] == "onset"
    ), "Something went wrong with the events file! The onset column was not found!"
    assert (
        events[0, duration_idx] == "duration"
    ), "Something went wrong with the events file! The duration column was not found!"
    if trial_type_label is not None:
        assert (
            events[0, trial_type_idx] == trial_type_label
        ), "Something went wrong with the events file! The trial_type column was not found!"

    if event_types is None:
        if trial_type_label is None:
            event_types = ["unknown"]
        else:
            event_types = list(np.unique(events[1:, trial_type_idx]))
            # remove all the rest labels
            for rest_label in rest_labels:
                if rest_label in event_types:
                    event_types.remove(rest_label)
        # add the rest label to the beginning for consistency
        event_types = ["rest"] + event_types

    Fs = float(1 / TR_mri) * oversampling
    num_time_task = int(num_time_mri * oversampling)
    event_labels = np.zeros((num_time_task, 1))
    for i in range(events.shape[0]):
        # skip the first row which is the header
        if i == 0:
            continue

        if trial_type_label is None:
            trial_type = "unknown"
        else:
            trial_type = events[i, trial_type_idx]

        if trial_type in event_types:
            # the only rest label that is left in event types is "rest" but we don't want to consider it
            if trial_type == "rest":
                continue
            start_time = float(events[i, onset_idx])
            end_time = float(events[i, onset_idx]) + float(events[i, duration_idx])
            start_timepoint = int(np.rint(start_time * Fs))
            end_timepoint = int(np.rint(end_time * Fs))
            event_labels[start_timepoint:end_timepoint] = event_types.index(trial_type)

    if return_0_1:
        event_labels = np.multiply(event_labels != 0, 1)

    return event_labels, Fs, event_types


################################# Visualization Functions ####################################


def plot_task_dFC(task_presence, dFC_lst, Fs_mri, TR_step=12):
    """
    task_presence: numpy array containing the task presence in the time points of the dFC data
    this function assumes that the task presence has the same Fs as the dFC data, i.e. MRI data
    and that the time points of the task presence are aligned with the time points of the dFC data
    """
    conn_mat_size = 20
    scale_task_plot = 20

    # plot task_data['event_labels']
    plt.figure(figsize=(50, 200))

    ax = plt.gca()

    time = np.arange(0, task_presence.shape[0]) / Fs_mri
    ax.plot(time, task_presence * scale_task_plot, linewidth=4)
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


################################# Stat Functions ####################################


def cohen_d_bold(X, y):
    """
    Compute Cohen's d per ROI between task and rest.

    Parameters
    ----------
    X : ndarray, shape (n_timepoints, n_ROIs)
        BOLD signals.
    y : ndarray, shape (n_timepoints,)
        Task labels: 0 = rest, 1 = task.

    Returns
    -------
    d_values : ndarray, shape (n_ROIs,)
        Cohen's d per ROI.
    """
    task_idx = y == 1
    rest_idx = y == 0

    X_task = X[task_idx, :]
    X_rest = X[rest_idx, :]

    mean_task = X_task.mean(axis=0)
    mean_rest = X_rest.mean(axis=0)

    std_task = X_task.std(axis=0, ddof=1)
    std_rest = X_rest.std(axis=0, ddof=1)

    n_task = X_task.shape[0]
    n_rest = X_rest.shape[0]

    pooled_std = np.sqrt(
        ((n_task - 1) * std_task**2 + (n_rest - 1) * std_rest**2) / (n_task + n_rest - 2)
    )

    d_values = (mean_task - mean_rest) / pooled_std

    return d_values


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


def event_labels_conv_hrf(event_labels, TR_mri, TR_task, no_hrf=False):
    """
    event_labels: event labels including 0 and event ids at the time each event happens
    TR_mri: TR of MRI
    TR_task: TR of task
    assumes that 0 is the resting state

    return: event labels convolved with HRF for each event type
    the convolved event labels have the same length as the event_labels
    event type i convolved with HRF is in events_hrf[:, i-1]

    events_hrf[:, 0] is the resting state

    if no_hrf is True, the event labels are not convolved with HRF
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

        if no_hrf:
            events_hrf[:, i] = event_signal
        else:
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
    flag = False
    if len(events_hrf.shape) == 1:
        flag = True
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


def shifted_binarizing(
    event_labels_all_task_hrf,
    task_presence_ratio=0.5,
    step=0.001,
):
    # find threshold such that the after binarization of event_labels_all_task_hrf,
    # the ratio of 1 to 0 is equal to task_presence_ratio
    for threshold in np.arange(0, np.max(event_labels_all_task_hrf), step):
        # binarize the event_labels_all_task_hrf
        event_labels_all_task_hrf_binarized = np.where(
            event_labels_all_task_hrf > threshold, 1, 0
        )
        # find the ratio of 1 to 0 in event_labels_all_task_hrf_binarized
        new_ratio = np.mean(event_labels_all_task_hrf_binarized)
        if new_ratio <= task_presence_ratio:
            break
    return threshold


def GMM_binarizing(
    event_labels_all_task_hrf,
    threshold=None,
    downsample=True,
    TR_mri=None,
    TR_task=None,
    TR_array=None,
):
    """_summary_

    Parameters
    ----------
    event_labels_all_task_hrf : _type_
        _description_
    threshold : float, optional
        _description_, by default 0.01
    downsample : bool, optional
        _description_, by default True
    TR_mri : _type_, optional
        _description_, by default None
    TR_task : _type_, optional
        _description_, by default None
    TR_array : _type_, optional
        _description_, by default None

    Returns
    -------
    task_presence : array
        _description_
    indices : array
        _description_
    -----------
    in order to get the task presence, use task_presence[indices]
    """
    if threshold is None:
        thresholds_list = [0.01, 0.1, 0.2, 0.3, 0.4]
    else:
        thresholds_list = [threshold]
    event_labels_all_task_hrf = event_labels_all_task_hrf.copy()
    event_labels_all_task_hrf_reshaped = event_labels_all_task_hrf.reshape(-1, 1)
    # normal the signal to [0, 1]
    event_labels_all_task_hrf_reshaped = (
        event_labels_all_task_hrf_reshaped - np.min(event_labels_all_task_hrf_reshaped)
    ) / (
        np.max(event_labels_all_task_hrf_reshaped)
        - np.min(event_labels_all_task_hrf_reshaped)
    )
    # Fit GMM
    gmm = GaussianMixture(
        n_components=2, means_init=np.array([[0.0], [1.0]]), n_init=5
    ).fit(event_labels_all_task_hrf_reshaped)
    means = gmm.means_.flatten()
    # if the lower mean is larger than 0.25 or the higher mean is smaller than 0.75, we need to use 3 components
    # first find the lower and higher mean
    lower_mean = np.min(means)
    higher_mean = np.max(means)
    if lower_mean > 0.25 or higher_mean < 0.75:
        # Fit GMM with 3 components
        gmm = GaussianMixture(
            n_components=3, means_init=np.array([[0.0], [0.5], [1.0]]), n_init=5
        ).fit(event_labels_all_task_hrf_reshaped)
    # downsample to MRI TR
    if downsample:
        event_labels_all_task_hrf_reshaped = downsample_events_hrf(
            event_labels_all_task_hrf_reshaped, TR_mri, TR_task
        )
    # some dFC measures (window-based) have a different TR than the task data
    if TR_array is not None:
        event_labels_all_task_hrf_reshaped = event_labels_all_task_hrf_reshaped[TR_array]
    # now predict on vs. off for the downsampled time points
    probs = gmm.predict_proba(event_labels_all_task_hrf_reshaped)
    # Identify which component corresponds to "on" (higher mean)
    # Each component has a mean, and in this case:
    # The "off" state should have a lower mean (closer to baseline).
    # The "on" state should have a higher mean (HRF-convolved signal is elevated during task).
    means = gmm.means_.flatten()
    on_component = np.argmax(means)
    off_component = np.argmin(means)
    # if len(means) == 3:
    #     # set the mid component to the one that is in between the on and off components
    #     mid_component = np.argsort(means)[1]
    # Get probability of being in the "on" and "off" state
    p_on = probs[:, on_component]
    p_off = probs[:, off_component]
    # if len(means) == 3:
    #     p_mid = probs[:, mid_component]
    # Create a binarized signal with transition points discarded
    for threshold_ in thresholds_list:
        # try different thresholds
        # lower thresholds may result in only one class being present
        indices = np.where((p_off >= (1 - threshold_)) | (p_on >= (1 - threshold_)))[0]
        task_presence = np.where(p_on >= (1 - threshold_), 1, 0)

        # check that both classes are non-empty
        unique_labels = np.unique(task_presence[indices])
        if len(unique_labels) == 2:
            break

        if threshold_ == 0.4:
            warnings.warn(
                f"Even with threshold={threshold_}, only one class present in confident samples."
            )

    return task_presence, indices


def extract_task_presence(
    event_labels,
    TR_task,
    TR_mri,
    TR_array=None,
    binary=True,
    binarizing_method="GMM",
    no_hrf=False,
):
    """
    event_labels: event labels including 0 and event ids at the time each event happens
    TR_task: TR of task
    TR_array: the time points of the dFC data, optional
    TR_mri: TR of MRI

    This function extracts the task presence from the event labels and returns it in the same time points as the dFC data
    It also downsamples the task presence to the time points of the dFC data
    if binary is True, the task presence is binarized using the mean of the task presence
    binarizing_method: 'median' or 'mean' or 'shift' or 'GMM'
    if binarizing_method is 'shift', the task presence is binarized such that the ratio of 1 to 0 is equal to the task presence ratio

    if no_hrf is True, the task presence is not convolved with HRF
    """

    # event_labels_all_task is all conditions together, rest vs. task times
    event_labels_all_task = np.multiply(event_labels != 0, 1)

    event_labels_all_task_hrf = event_labels_conv_hrf(
        event_labels=event_labels_all_task, TR_mri=TR_mri, TR_task=TR_task, no_hrf=no_hrf
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
        if binarizing_method == "median":
            threshold = np.median(event_labels_all_task_hrf)
            task_presence = np.where(event_labels_all_task_hrf > threshold, 1, 0)
            task_presence = downsample_events_hrf(task_presence, TR_mri, TR_task)
            # some dFC measures (window-based) have a different TR than the task data
            if TR_array is not None:
                task_presence = task_presence[TR_array]
            indices = np.arange(task_presence.shape[0])
        elif binarizing_method == "mean":
            threshold = np.mean(event_labels_all_task_hrf)
            task_presence = np.where(event_labels_all_task_hrf > threshold, 1, 0)
            task_presence = downsample_events_hrf(task_presence, TR_mri, TR_task)
            # some dFC measures (window-based) have a different TR than the task data
            if TR_array is not None:
                task_presence = task_presence[TR_array]
            indices = np.arange(task_presence.shape[0])
        elif binarizing_method == "shift":
            task_presence_ratio = np.mean(event_labels_all_task)
            threshold = shifted_binarizing(
                event_labels_all_task_hrf=event_labels_all_task_hrf,
                task_presence_ratio=task_presence_ratio,
            )
            task_presence = np.where(event_labels_all_task_hrf > threshold, 1, 0)
            task_presence = downsample_events_hrf(task_presence, TR_mri, TR_task)
            # some dFC measures (window-based) have a different TR than the task data
            if TR_array is not None:
                task_presence = task_presence[TR_array]
            indices = np.arange(task_presence.shape[0])
        elif binarizing_method == "GMM":
            task_presence, indices = GMM_binarizing(
                event_labels_all_task_hrf=event_labels_all_task_hrf,
                threshold=None,
                downsample=True,
                TR_mri=TR_mri,
                TR_task=TR_task,
                TR_array=TR_array,
            )
        else:
            raise ValueError(
                "binarizing_method should be 'median', 'mean', 'shift', or 'GMM'"
            )
    else:
        task_presence = event_labels_all_task_hrf
        task_presence = downsample_events_hrf(task_presence, TR_mri, TR_task)
        # some dFC measures (window-based) have a different TR than the task data
        if TR_array is not None:
            task_presence = task_presence[TR_array]
        indices = np.arange(task_presence.shape[0])

    return task_presence, indices


################################# Task Design Features ####################################


def calc_relative_task_on(task_presence):
    """
    task_presence: 0, 1 array
    return: relative_task_on
    """
    return np.sum(task_presence) / len(task_presence)


def calc_task_duration(task_presence, TR_mri):
    """
    task_presence: 0, 1 array
    return: list of task_durations
    """
    task_durations = list()
    start = None
    for i in range(1, len(task_presence)):
        if task_presence[i] == 1 and task_presence[i - 1] == 0:
            start = i
        if (
            (task_presence[i] == 0)
            and (task_presence[i - 1] == 1)
            and (start is not None)
        ):
            end = i
            task_durations.append((end - start) * TR_mri)
            start = None
    task_durations = np.array(task_durations)
    return task_durations


def calc_rest_duration(task_presence, TR_mri):
    """
    task_presence: 0, 1 array
    return: list of rest_durations
    """
    rest_durations = list()
    if task_presence[0] == 0:
        start = 0
    for i in range(1, len(task_presence)):
        if task_presence[i] == 0 and task_presence[i - 1] == 1:
            start = i
        if task_presence[i] == 1 and task_presence[i - 1] == 0:
            end = i
            rest_durations.append((end - start) * TR_mri)
            start = None
    if task_presence[-1] == 0:
        end = len(task_presence)
        if not start is None:
            rest_durations.append((end - start) * TR_mri)
    rest_durations = np.array(rest_durations)
    return rest_durations


def calc_transition_freq(task_presence):
    """
    task_presence: 0, 1 array
    return: num_of_transitions, relative_transition_freq
    """
    transitions = np.abs(np.diff(task_presence))
    num_of_transitions = np.sum(transitions)
    relative_transition_freq = num_of_transitions / len(task_presence)
    return num_of_transitions, relative_transition_freq


def noise_model(f, alpha=1.0):
    # 1/f^alpha normalized to unit median (cheap default)
    spec = 1.0 / np.maximum(f, 1e-6) ** alpha
    med = np.median(spec[f > 0])
    return spec / med


def compute_periodicity_index(
    event_labels,
    TR_task,
    fmin=0.0,
    fmax=None,
    no_hrf=False,
):
    """
    Compute a noise-free periodicity index for a task timing time course.

    Parameters
    ----------
    event_labels : array, shape (T,)
        Event labels time course.
    TR_task : float
        Repetition time (seconds).
    fmin, fmax : float
        Frequency band (Hz) to consider. If fmax is None, Nyquist is used.
    no_hrf : bool
        If True, do not convolve with HRF.

    Returns
    -------
    results : dict
        {
          'periodicity_index': float in [0, 1], higher = more periodic,
          'spectral_entropy': float in [0, 1], lower = more periodic,
          'peak_freq': float, frequency of dominant peak (Hz),
          'peak_dominance': float in [0, 1], peak power / total power
        }
    """
    if no_hrf:
        task_tc = np.multiply(event_labels != 0, 1)
    else:
        task_tc, _ = extract_task_presence(
            event_labels=event_labels,
            TR_task=TR_task,
            TR_mri=TR_task,
            TR_array=None,
            binary=False,
            binarizing_method="GMM",
            no_hrf=False,
        )
    task_tc = np.asarray(task_tc)
    T = len(task_tc)

    # Detrend and mean-center
    x = task_tc - np.mean(task_tc)

    # FFT
    freqs = np.fft.rfftfreq(T, d=TR_task)
    fft_vals = np.fft.rfft(x)
    power = np.abs(fft_vals) ** 2

    # Restrict frequency range
    if fmax is None:
        fmax = 0.5 / TR_task  # Nyquist
    mask = (freqs >= fmin) & (freqs <= fmax)
    freqs = freqs[mask]
    power = power[mask]

    # Avoid division by zero
    if np.all(power == 0):
        return {
            "periodicity_index": 0.0,
            "spectral_entropy": 1.0,
            "peak_freq": 0.0,
            "peak_dominance": 0.0,
        }

    # Normalize spectrum to probability distribution
    p = power / power.sum()

    # Spectral entropy (normalized to [0,1])
    eps = 1e-12
    H = -(p * np.log(p + eps)).sum() / np.log(len(p))  # in [0,1], higher = more "flat"

    # Dominant peak and its dominance
    peak_idx = np.argmax(power)
    peak_freq = freqs[peak_idx]
    peak_power = power[peak_idx]
    peak_dominance = peak_power / power.sum()  # 0–1

    # Define periodicity index: high when entropy is low and peak is dominant
    periodicity_index = (1.0 - H) * peak_dominance

    return {
        "periodicity_index": float(periodicity_index),
        "spectral_entropy": float(H),
        "peak_freq": float(peak_freq),
        "peak_dominance": float(peak_dominance),
    }


def compute_optimality_index(
    event_labels, TR_task, TR_mri, fmin=0.0, fmax=None, alpha=1.0
):
    """
    Compute a Worsley-style optimality index (OI) and normalized OI.

    Uses HRF spectrum as the weighting term (no explicit noise model).
    OI_norm compares the observed design to an ideal sinusoid with
    matched in-band power placed at the optimal frequency.

    Returns:
    --------
    {
      "OI": float,
      "OI_ideal": float,
      "OI_norm": float,
      "peak_freq": float
    }
    """

    # -------------------------
    # 1. Preprocess task timing
    # -------------------------
    task_tc = np.multiply(event_labels != 0, 1).astype(float).flatten()
    T = len(task_tc)

    # -------------------------
    # 2. HRF Model
    # -------------------------
    # same length as our task
    time_length_HRF = T * TR_task
    oversampling = TR_mri / TR_task

    hrf_tc = glm.first_level.spm_hrf(
        tr=TR_mri, oversampling=oversampling, time_length=time_length_HRF, onset=0.0
    )
    hrf_tc = np.asarray(hrf_tc)

    # Pad or truncate HRF to length T
    if len(hrf_tc) < T:
        hrf_tc = np.pad(hrf_tc, (0, T - len(hrf_tc)), mode="constant")
    else:
        hrf_tc = hrf_tc[:T]

    # -------------------------
    # 3. Frequency grid + noise PSD
    # -------------------------
    freqs = np.fft.rfftfreq(T, d=TR_task)

    # -------------------------
    # 4. FFT-based spectra
    # -------------------------
    design_spectrum = np.abs(np.fft.rfft(task_tc)) ** 2
    hrf_spectrum = np.abs(np.fft.rfft(hrf_tc)) ** 2

    # -------------------------
    # 5. Frequency mask
    # -------------------------
    if fmax is None:
        fmax = 0.5 / TR_task

    mask = (freqs >= float(fmin)) & (freqs <= fmax)
    freqs_m = freqs[mask]
    design_spectrum_m = design_spectrum[mask]
    hrf_spectrum_m = hrf_spectrum[mask]

    eps = 1e-12
    snr_weight = hrf_spectrum_m

    # -------------------------
    # 6. ORIGINAL (TASK) OI
    # -------------------------
    OI = np.sum(design_spectrum_m * snr_weight)

    # -------------------------
    # 7. IDEAL OI UPPER BOUND
    # -------------------------

    if freqs_m.size == 0:
        # no nonzero frequencies in the band
        return {
            "OI": float(OI),
            "OI_ideal": 0.0,
            "OI_norm": 0.0,
            "peak_freq": 0.0,
        }

    # Report dominant task frequency for interpretability.
    peak_idx = np.argmax(design_spectrum_m)
    peak_freq = freqs_m[peak_idx]

    # Theoretical in-band upper bound for weighted spectral sum:
    #   sum(d_i * w_i) <= max(w_i) * sum(d_i), with d_i >= 0.
    design_band_power = float(np.sum(design_spectrum_m))
    max_weight = float(np.max(snr_weight)) if snr_weight.size > 0 else 0.0
    OI_ideal = design_band_power * max_weight

    # -------------------------
    # 8. Normalized OI
    # -------------------------
    if OI_ideal < eps:
        OI_norm = 0.0
    else:
        OI_norm = OI / OI_ideal

    return {
        "OI": float(OI),
        "OI_ideal": float(OI_ideal),
        "OI_norm": float(OI_norm),
        "peak_freq": float(peak_freq),
    }


from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


def periodicity_autocorr(event_labels, TR_task, max_lag=None):
    """
    Measure how periodic a 0/1 event label time course is using autocorrelation.

    Parameters
    ----------
    event_labels : array-like
        array of 0/1 labels (e.g., rest=0, task=1).
    TR_task : float
        Repetition time (seconds).
    max_lag : int or None
        Maximum lag to compute autocorrelation. If None, uses len(x)//2.

    Returns
    -------
    periodicity : float
        Strength of the strongest non-zero autocorrelation peak (in [−1, 1]).
    best_lag : int
        Lag (in samples) at which this peak occurs.
    r : np.ndarray
        Autocorrelation values from lag 0..max_lag.
    """
    x, _ = extract_task_presence(
        event_labels=event_labels,
        TR_task=TR_task,
        TR_mri=TR_task,
        TR_array=None,
        binary=False,
        binarizing_method="GMM",
        no_hrf=False,
    )

    # Optional: center to remove bias from unbalanced 0/1 ratio
    x = x - x.mean()

    if max_lag is None:
        max_lag = len(x) // 2

    # r[0] = 1 by definition
    r = acf(x, nlags=max_lag, fft=False)

    # Find true peaks (periodic peaks) ---
    peaks, _ = find_peaks(r)

    if len(peaks) == 0:
        return {"periodicity": 0.0, "best_lag": None, "r": r}

    # skip lag 0
    peaks = peaks[peaks > 0]

    if len(peaks) == 0:
        return {"periodicity": 0.0, "best_lag": None, "r": r}

    best_lag = peaks[np.argmax(r[peaks])]

    # # Ignore lag 0, find strongest positive correlation
    # r_nonzero = r[1:]
    # best_lag = np.argmax(r_nonzero) + 1
    periodicity = r[best_lag]

    return {
        "periodicity": periodicity,
        "best_lag": best_lag,
        "r": r,
    }
