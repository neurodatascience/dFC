# -*- coding: utf-8 -*-
"""
Functions to facilitate dFC simulation.

Created on April 25 2024
@author: Mohammad Torabi
"""

import numpy as np
from scipy import signal
from tvb.simulator.lab import *

from pydfc import TIME_SERIES, task_utils

################################# Simulation Functions ####################################


class CustomStimuli(patterns.StimuliRegion):
    def __init__(
        self, stimulus_timing, region_weighting, connectivity, amplitude=1.0, **kwargs
    ):
        """
        Parameters:
        - stimulus_timing: array of 0s and 1s (or amplitudes) over time
        - target_nodes: list or array of node indices where to apply the stimulus
        - amplitude: default amplitude (multiplied by stimulus_timing value)
        """
        super().__init__(**kwargs)
        self.stimulus_timing = np.array(stimulus_timing)
        self.amplitude = amplitude
        self.current_idx = 0
        self.weight = region_weighting
        self.connectivity = connectivity  # Required by TVB, even if not used
        # Required by TVB, even if not used
        self.temporal = equations.PulseTrain()
        self.spatial = equations.DiscreteEquation()

    def __call__(self, temporal_indices, spatial_indices=None):

        # if temporal_indices is not a single integer, raise an error
        if not isinstance(temporal_indices, (int, np.integer)):
            raise ValueError(
                "CustomStimuli expects a single integer for temporal_indices."
            )
        # time is milliseconds
        n_nodes = self.weight.shape[0]
        stim = np.zeros(n_nodes)

        # Determine which index in the stimulus array corresponds to current time
        self.current_idx = temporal_indices

        if self.current_idx < len(self.stimulus_timing):
            stim_value = self.stimulus_timing[self.current_idx] * self.amplitude
        else:
            stim_value = 0  # stimulus ends when array is exhausted

        stim = np.multiply(self.weight, stim_value)
        self.stimulus = stim
        return self.stimulus

    def set_state(self, state):
        self.state = state

    def configure_time(self, t):
        pass


def create_random_stimulus_weights(stimulated_regions_list, n_regions=76):
    """
    Create random stimulus weights for the stimulated regions.
    """
    rand_weighting = [
        np.random.normal(loc=2.0 ** (-1 * (2 + i)), scale=0.1 * (2.0**-2))
        for i in range(len(stimulated_regions_list))
    ]

    # configure stimulus spatial pattern
    weighting = np.zeros((n_regions,))
    weighting[stimulated_regions_list] = rand_weighting

    return weighting


def simulate_task_BOLD(
    stimulus_timing,
    sim_length,
    BOLD_period,
    TAVG_period,
    num_stimulated_regions=5,
    global_conn_coupling_coef=0.0126,
    D=0.001,
    conn_speed=1.0,
    dt=0.5,
    drop_initial_time=False,
):
    """
    Simulate BOLD signal for a task.

    Parameters
    ----------
    stimulus_timing : array-like, optional
        The stimulus timing array, which should contain 0s and 1s (or amplitudes) over time.
    sim_length : float
        The length of the simulation in seconds.
    BOLD_period : float
        The BOLD period in milliseconds.
    TAVG_period : float
        The TAVG period in milliseconds.
    num_stimulated_regions : int, optional
        The number of stimulated regions. The default is 5.
        if num_stimulated_regions is 5, the stimulated regions are:
        [0, 7, 13, 33, 42]
        if num_stimulated_regions is 16, the stimulated regions are:
        regions = list(range(0, 76, 5))
        if num_stimulated_regions is 26, the stimulated regions are:
        regions = list(range(0, 76, 3))
        else, the stimulated regions are randomly selected.
    """
    # randomize some parameters for each subjects
    global_conn_coupling = np.random.normal(loc=global_conn_coupling_coef, scale=0.0075)
    conn_speed_rand = np.random.normal(loc=conn_speed, scale=0.1 * conn_speed)
    ################################# Initialize Simulation ####################################
    conn = connectivity.Connectivity.from_file()
    conn.speed = np.array([conn_speed_rand])
    conn.configure()
    # randomize the structural connectivity
    # Additive Gaussian noise (e.g. 10% of weight magnitude)
    noise_level = 0.1  # 10%
    conn.weights += np.random.normal(
        loc=0,
        scale=noise_level * np.std(conn.weights[conn.weights > 0]),
        size=conn.weights.shape,
    )
    # Remove negative weights if any
    conn.weights = np.clip(conn.weights, 0, None)
    # reconfigure the connectivity
    conn.configure()

    # configure stimulus spatial pattern
    if num_stimulated_regions == 5:
        stimulated_regions_list = [0, 7, 13, 33, 42]
    elif num_stimulated_regions == 16:
        stimulated_regions_list = list(range(0, 76, 5))
    elif num_stimulated_regions == 26:
        stimulated_regions_list = list(range(0, 76, 3))
    else:
        stimulated_regions_list = np.random.choice(
            np.arange(76), num_stimulated_regions, replace=False
        )
        stimulated_regions_list = list(stimulated_regions_list)
    weighting = create_random_stimulus_weights(
        stimulated_regions_list=stimulated_regions_list, n_regions=76
    )

    # check if stimulus_timing is only containing 0s and 1s
    if not np.all(np.isin(stimulus_timing, [0, 1])):
        raise ValueError("stimulus_timing should only contain 0s and 1s.")

    stimulus = CustomStimuli(
        stimulus_timing=stimulus_timing,
        region_weighting=weighting,
        connectivity=conn,
        amplitude=1.0,
    )

    ################################# Run Simulation ####################################

    # set the global coupling strength
    # you can switch between deterministic (without noise) and stochastic integration (with noise)
    sim = simulator.Simulator(
        model=models.Generic2dOscillator(a=np.array([0.5])),
        connectivity=conn,
        coupling=coupling.Linear(a=np.array([global_conn_coupling])),
        # integrator=integrators.HeunDeterministic(dt=dt),
        integrator=integrators.HeunStochastic(
            dt=dt, noise=noise.Additive(nsig=np.array([D]))
        ),
        monitors=(
            monitors.TemporalAverage(period=TAVG_period),
            monitors.Bold(period=BOLD_period, hrf_kernel=equations.MixtureOfGammas()),
            monitors.ProgressLogger(period=10e3),
        ),
        stimulus=stimulus,
        simulation_length=sim_length,
    ).configure()

    (tavg_time, tavg_data), (bold_time, bold_data), _ = sim.run()

    if drop_initial_time:
        # truncate the first 10 seconds of the simulation
        # to avoid transient effects
        truncate_time = 10e3  # in m sec
        bold_truncate_idx = int(truncate_time / BOLD_period)
        bold_time = bold_time[bold_truncate_idx:]
        bold_data = bold_data[bold_truncate_idx:]
        tavg_truncate_idx = int(truncate_time / TAVG_period)
        tavg_time = tavg_time[tavg_truncate_idx:]
        tavg_data = tavg_data[tavg_truncate_idx:]

    centres_locs = conn.centres
    region_labels = list(conn.region_labels)
    TR_mri = BOLD_period * 1e-3  # in seconds

    bold_data = bold_data[:, 0, :, 0]
    # change time_series.shape to (roi, time)
    bold_data = bold_data.T

    TAVG_data = tavg_data[:, 0, :, 0]
    # change time_series.shape to (roi, time)
    TAVG_data = TAVG_data.T

    return (
        bold_data,
        bold_time,
        region_labels,
        centres_locs,
        TR_mri,
        TAVG_data,
        tavg_time,
        TAVG_period,
    )


def create_simul_task_info(
    TR_mri,
    task,
    onset,
    task_duration,
    task_block_duration,
    sim_length,
    oversampling=50,
):
    """
    Create a dictionary containing the task data for simulation.

    Parameters
    ----------
    TR_mri : float
        The repetition time of the MRI in seconds.
    task : str
        The task name.
    onset : float
        The onset time of the task.
    task_duration : float
        The duration of the task.
    task_block_duration : float
        The duration of the task block.
    sim_length : float
        The length of the simulation.
        in milliseconds
    oversampling : int, optional
        The oversampling factor. The default is 50.
        generate more samples per TR than the func data to have a
        better event_labels time resolution
    """
    ####################### EXTRACT TASK LABELS #######################
    events = []

    # using onset, task_duration, task_block_duration to create the events
    events.append(["onset", "duration", "trial_type"])
    t = onset
    while t < (sim_length * 1e-3):
        events.append([t, task_duration, "task"])
        t += task_block_duration
    events = np.array(events)

    # find the number of time points in the MRI data
    # sim_length is in milliseconds
    num_time_mri = int((sim_length * 1e-3) / TR_mri)

    event_labels, Fs_task, event_types = task_utils.events_time_to_labels(
        events=events,
        TR_mri=TR_mri,
        num_time_mri=num_time_mri,
        oversampling=oversampling,
        return_0_1=False,
    )
    # fill task labels with 0 (rest) and 1 (task's index, here only 1 task is used)
    task_labels = np.multiply(event_labels != 0, 1)
    ################################# SAVE #################################
    # save the ROI time series and task data
    task_data = {
        "task": task,
        "task_labels": task_labels,
        "event_labels": event_labels,
        "event_types": event_types,
        "events": events,
        "Fs_task": Fs_task,
        "TR_mri": TR_mri,
        "num_time_mri": num_time_mri,
    }

    return task_data


def event_labels_to_stimulus_timing(
    event_labels,
    Fs_task,
    dt,
):
    """
    Convert event labels to stimulus timing.
    Parameters
    ----------
    event_labels : array-like
        The event labels, which should contain 0s (rest) and event ids over time.
    Fs_task : float
        The sampling frequency of the task data in Hz.
    dt : float
        The simulation time step in milliseconds.
    """
    # make sure the timings are only 0s and 1s
    stimulus_timing = np.multiply(event_labels != 0, 1)

    # make sure task_data sampling frequency is equal to simulation time step
    L_old = len(stimulus_timing)
    L_new = int((L_old * 1e3) / (Fs_task * dt))
    stimulus_timing = signal.resample(stimulus_timing, L_new)
    # binarize the stimulus timing
    # because of the resampling, the values might not be exactly 0 or 1
    stimulus_timing = np.where(stimulus_timing > 0.5, 1, 0)

    return stimulus_timing


def simulate_task_BOLD_TS(
    subj_id,
    task_data,
    TAVG_period,
    num_stimulated_regions=5,
    global_conn_coupling_coef=0.0126,
    D=0.001,
    conn_speed=1.0,
    dt=0.5,
    drop_initial_time=False,
):
    """
    Simulate BOLD signal for a task and return a TIME_SERIES object.
    """
    task = task_data["task"]
    BOLD_period = task_data["TR_mri"] * 1e3  # convert to milliseconds
    sim_length = task_data["num_time_mri"] * BOLD_period  # in milliseconds
    stimulus_timing = event_labels_to_stimulus_timing(
        event_labels=task_data["event_labels"],
        Fs_task=task_data["Fs_task"],
        dt=dt,
    )

    bold_data, bold_time, region_labels, centres_locs, TR_mri, _, _, _ = (
        simulate_task_BOLD(
            stimulus_timing=stimulus_timing,
            sim_length=sim_length,
            BOLD_period=BOLD_period,
            TAVG_period=TAVG_period,
            num_stimulated_regions=num_stimulated_regions,
            global_conn_coupling_coef=global_conn_coupling_coef,
            D=D,
            conn_speed=conn_speed,
            dt=dt,
            drop_initial_time=drop_initial_time,
        )
    )
    time_series = TIME_SERIES(
        data=bold_data,
        subj_id=subj_id,
        Fs=1 / TR_mri,
        locs=centres_locs,
        node_labels=region_labels,
        TS_name=f"BOLD_{subj_id}_{task}",
        session_name=task,
    )

    return time_series


def simulate_task_data(subj_id, task_info):
    """
    Simulate task-based BOLD signal for a subject.

    Parameters
    ----------
    subj_id : str
        The subject ID.
    task_info : dict
        A dictionary containing the task information below:
            - task_name: str
                The name of the task.
            - task_data: str
                Path to a dictionary containing the task parameters
                if task_data is not provided, onset_time, task_duration, task_block_duration,
                sim_length, will be used to create the task data.
            - onset_time: float
                The onset time of the task in seconds.
            - task_duration: float
                The duration of the task in seconds.
            - task_block_duration: float
                The duration of the task block in seconds.
            - sim_length: float
                The length of the simulation in milliseconds.
            - BOLD_period: float
                The BOLD period in milliseconds.
            - TAVG_period: float
                The TAVG period in milliseconds.
            - num_stimulated_regions: int
                The number of stimulated regions.
            - global_conn_coupling_coef: float
                The global connectivity coupling coefficient.
            - D: float
                The noise parameter.
            - conn_speed: float
                The connectivity speed.
            - dt: float
                The simulation time step in milliseconds.
    """
    if "task_data" in task_info:
        # task_info["task_data"] is a path to a dictionary with {subj_id} as a placeholder
        if "{subj_id}" in task_info["task_data"]:
            task_data_path = task_info["task_data"].replace("{subj_id}", subj_id)
        else:
            task_data_path = task_info["task_data"]
        task_data = np.load(task_data_path, allow_pickle="TRUE").item()
    else:
        task_data = create_simul_task_info(
            TR_mri=task_info["BOLD_period"] * 1e-3,  # convert to seconds
            task=task_info["task_name"],
            onset=task_info["onset_time"],
            task_duration=task_info["task_duration"],
            task_block_duration=task_info["task_block_duration"],
            sim_length=task_info["sim_length"],
        )

    time_series = simulate_task_BOLD_TS(
        subj_id=subj_id,
        task_data=task_data,
        TAVG_period=task_info["TAVG_period"],
        num_stimulated_regions=task_info["num_stimulated_regions"],
        global_conn_coupling_coef=task_info["global_conn_coupling_coef"],
        D=task_info["D"],
        conn_speed=task_info["conn_speed"],
        dt=task_info["dt"],
    )

    # make sure task_data["num_time_mri"] is equal to the number of time points in the time series
    if task_data["num_time_mri"] != time_series.n_time:
        task_data["num_time_mri"] = time_series.n_time

    return time_series, task_data
