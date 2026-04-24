"""
Implementation of functions for loading fmri data.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import os
from copy import deepcopy

import h5py
import numpy as np
from nilearn import datasets
from nilearn.interfaces.fmriprep import load_confounds, load_confounds_strategy
from nilearn.maskers import NiftiLabelsMasker, NiftiSpheresMasker
from nilearn.plotting import find_parcellation_cut_coords

from .dfc_utils import intersection, label2network
from .time_series import TIME_SERIES

################################# DATA_LOADER functions ######################################


def find_subj_list(data_root):
    """
    find the list of subjects in data_root
    the files must follow the format: sub-subjectID
    only these files should be in the data_root
    """
    if data_root[-1] != "/":
        data_root += "/"

    ALL_FILES = os.listdir(data_root)
    FOLDERS = [item for item in ALL_FILES if os.path.isdir(data_root + item)]

    FOLDERS.sort()
    SUBJECTS = list()
    for s in FOLDERS:
        if "sub-" in s:
            SUBJECTS.append(s)
    SUBJECTS.sort()

    print(f"{len(SUBJECTS)} subjects were found. ")

    return SUBJECTS


def load_from_array(subj_id2load=None, **params):
    """
    load fMRI data from numpy or mat files
    input time_series.shape must be (time, roi)
    returns a dictionary of TIME_SERIES objects
    each corresponding to a session

    - if the file_name is a .mat file, it will be loaded using h5py
      if the file_name is a .npy file, it will be loaded using np.load

    - the roi locations should be in the same folder and a .npy file
      with the name: params['roi_locs_file']
      it must

    - and the roi labels should be in the same folder and a .npy file
      with the name: params['roi_labels_file']
      it must be a list of strings

    - labels should be in the format: Hemisphere_Network_ID
        ow, the network2include will not work properly
    """

    SESSIONs = params["SESSIONs"]  # list of sessions
    if subj_id2load is None:
        SUBJECTS = find_subj_list(params["data_root"])
    else:
        SUBJECTS = [subj_id2load]

    # LOAD Region Location DATA
    locs = np.load(
        params["data_root"] + params["roi_locs_file"], allow_pickle="True"
    ).item()
    locs = locs["locs"]

    # LOAD Region Labels DATA
    labels = np.load(
        params["data_root"] + params["roi_labels_file"], allow_pickle="True"
    ).item()
    labels = labels["labels"]

    assert type(locs) is np.ndarray, "locs must be a numpy array"
    assert type(labels) is list, "labels must be a list"
    assert locs.shape[0] == len(labels), "locs and labels must have the same length"
    assert locs.shape[1] == 3, "locs must have 3 columns"

    # apply networks2include
    # if params['networks2include'] is None, all the regions will be included
    if not params["networks2include"] is None:
        nodes2include = [
            i
            for i, x in enumerate(labels)
            if label2network(x) in params["networks2include"]
        ]
    else:
        nodes2include = [i for i, x in enumerate(labels)]
    locs = locs[nodes2include, :]
    labels = [x for node, x in enumerate(labels) if node in nodes2include]

    BOLD = {}
    for session in SESSIONs:
        BOLD[session] = None
        for subject in SUBJECTS:

            subj_fldr = subject + "_" + session

            # LOAD BOLD Data

            if params["file_name"][params["file_name"].find(".") :] == ".mat":
                with h5py.File(
                    params["data_root"] + subj_fldr + "/" + params["file_name"], "r"
                ) as f:
                    DATA = {k: np.array(f[k]) for k in f.keys()}
            elif params["file_name"][params["file_name"].find(".") :] == ".npy":
                DATA = np.load(
                    params["data_root"] + subj_fldr + "/" + params["file_name"],
                    allow_pickle="True",
                ).item()
            time_series = DATA["ROI_data"]  # time_series.shape = (time, roi)

            # change time_series.shape to (roi, time)
            time_series = time_series.T

            # apply networks2include
            time_series = time_series[nodes2include, :]

            if BOLD[session] is None:
                BOLD[session] = TIME_SERIES(
                    data=time_series,
                    subj_id=subject,
                    Fs=params["Fs"],
                    locs=locs,
                    node_labels=labels,
                    TS_name="BOLD Real",
                    session_name=session,
                )
            else:
                BOLD[session].append_ts(new_time_series=time_series, subj_id=subject)

        print("*** Session " + session + ": ")
        print(
            "number of regions= "
            + str(BOLD[session].n_regions)
            + ", number of time points= "
            + str(BOLD[session].n_time)
        )

    return BOLD


def extract_region_signals(
    nifti_file,
    masker_type="NiftiLabelsMasker",
    confound_strategy="none",
    standardize=False,
    labels_img=None,
    seeds=None,
    radius=None,
):
    """
    this function uses nilearn maskers to extract
    BOLD signals from nifti files

    returns a numpy array of shape (time, roi)
    and labels and locs of rois

    confound_strategy:
        'none': no confounds are used
        'no_motion': motion parameters are used
        'no_motion_no_gsr': motion parameters are used
                            and global signal regression
                            is applied.
        'simple': nilearn's simple preprocessing with
                            full motion and basic wm_csf
                            and high_pass
        'simple+gsr': nilearn's simple preprocessing with
                            full motion and basic wm_csf
                            and high_pass and global signal regression

    For now it only works with NiftiLabelsMasker and NiftiSpheresMasker and not with NiftiMapsMasker
    masker_type: "NiftiLabelsMasker" or "NiftiSpheresMasker"
    """
    if masker_type == "NiftiSpheresMasker":
        # check if seeds and radius are provided
        if seeds is None or radius is None:
            raise ValueError("For NiftiSpheresMasker, seeds and radius must be provided.")
        # create the masker for extracting time series
        masker = NiftiSpheresMasker(
            seeds=seeds,
            radius=radius,  # radius in mm
            standardize=standardize,
        )
    elif masker_type == "NiftiLabelsMasker":
        # check if labels_img is provided
        if labels_img is None:
            raise ValueError("For NiftiLabelsMasker, labels_img must be provided.")
        # create the masker for extracting time series
        masker = NiftiLabelsMasker(
            labels_img=labels_img,
            resampling_target="data",
            standardize=standardize,
        )
    else:
        raise ValueError(
            "masker_type must be 'NiftiLabelsMasker' or 'NiftiSpheresMasker', "
            f"but got {masker_type}"
        )

    ### extract the timeseries
    if confound_strategy == "none":
        time_series = masker.fit_transform(nifti_file)
    elif confound_strategy == "no_motion":
        confounds_simple, sample_mask = load_confounds(
            nifti_file,
            strategy=["high_pass", "motion", "wm_csf"],
            motion="basic",
            wm_csf="basic",
        )
        time_series = masker.fit_transform(
            nifti_file, confounds=confounds_simple, sample_mask=sample_mask
        )
    elif confound_strategy == "no_motion_no_gsr":
        confounds_simple, sample_mask = load_confounds(
            nifti_file,
            strategy=["high_pass", "motion", "wm_csf", "global_signal"],
            motion="basic",
            wm_csf="basic",
            global_signal="basic",
        )
        time_series = masker.fit_transform(
            nifti_file, confounds=confounds_simple, sample_mask=sample_mask
        )
    elif confound_strategy == "simple":
        confounds_simple, sample_mask = load_confounds_strategy(
            nifti_file, denoise_strategy="simple"
        )
        time_series = masker.fit_transform(
            nifti_file, confounds=confounds_simple, sample_mask=sample_mask
        )
    elif confound_strategy == "simple+gsr":
        confounds, sample_mask = load_confounds_strategy(
            nifti_file,
            denoise_strategy="simple",
            global_signal="basic",
        )
        time_series = masker.fit_transform(
            nifti_file, confounds=confounds, sample_mask=sample_mask
        )
    else:
        raise ValueError(
            "confound_strategy must be one of 'none', 'no_motion', 'no_motion_no_gsr', 'simple', or 'simple+gsr', "
            f"but got {confound_strategy}"
        )

    return time_series


def nifti2array(
    nifti_file,
    masker_type="NiftiLabelsMasker",
    confound_strategy="none",
    standardize=False,
    n_rois=100,
    labels_img=None,
    seeds=None,
    radius=None,
    region_names=None,
):
    """
    this function uses nilearn maskers to extract
    BOLD signals from nifti files

    returns a numpy array of shape (time, roi)
    and labels and locs of rois

    confound_strategy:
        'none': no confounds are used
        'no_motion': motion parameters are used
        'no_motion_no_gsr': motion parameters are used
                            and global signal regression
                            is applied.
        'simple': nilearn's simple preprocessing with
                            full motion and basic wm_csf
                            and high_pass
        'simple+gsr': nilearn's simple preprocessing with
                            full motion and basic wm_csf
                            and high_pass and global signal regression

    For now it only works with NiftiLabelsMasker and NiftiSpheresMasker and not with NiftiMapsMasker
    masker_type: "NiftiLabelsMasker" or "NiftiSpheresMasker"
    if masker_type is "NiftiLabelsMasker",
    labels_img must be provided or n_rois must be provided
    if masker_type is "NiftiSpheresMasker",
    seeds and radius must be provided

    Note:
    when not using Schaefer atlas, make sure
    that the labels_img/seeds and region_names are in the same order.
    """
    if masker_type == "NiftiLabelsMasker":
        if labels_img is None:
            # in this case, we will use the schaefer atlas
            # we use n_rois to determine the number of rois
            assert n_rois in [
                100,
                200,
                300,
                400,
                500,
                600,
                700,
                800,
                900,
                1000,
            ], "n_rois must be one of {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000}"
            # fetch the schaefer atlas
            parc = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois)
            labels_img = parc.maps
            labels = parc.labels
            labels = [label.decode() for label in labels]
        else:
            assert (
                region_names is not None
            ), "region_names must be provided if labels_img is provided"
            assert type(region_names) is list, "region_names must be a list of strings"

            labels = region_names

        # extract locs from labels_img
        # check if order is the same as labels
        locs, labels_ = find_parcellation_cut_coords(
            labels_img, background_label=0, return_label_names=True
        )  # numpy.ndarray of shape (n_labels, 3)

    elif masker_type == "NiftiSpheresMasker":

        # make sure seeds is a list of tuples (x, y, z)
        assert seeds is not None, "seeds must be provided for NiftiSpheresMasker"
        assert radius is not None, "radius must be provided for NiftiSpheresMasker"
        assert type(seeds) is list, "seeds must be a list of tuples (x, y, z)"
        assert all(
            isinstance(seed, tuple) and len(seed) == 3 for seed in seeds
        ), "seeds must be a list of tuples (x, y, z) with 3 elements each"

        locs = np.array(seeds)  # seeds should be a list of tuples (x, y, z)

        assert (
            region_names is not None
        ), "region_names must be provided if seeds are provided"
        assert type(region_names) is list, "region_names must be a list of strings"

        labels = region_names

    else:
        raise ValueError(
            "masker_type must be 'NiftiLabelsMasker' or 'NiftiSpheresMasker', "
            f"but got {masker_type}"
        )

    # extract the timeseries
    time_series = extract_region_signals(
        nifti_file=nifti_file,
        masker_type=masker_type,
        confound_strategy=confound_strategy,
        standardize=standardize,
        labels_img=labels_img,
        seeds=seeds,
        radius=radius,
    )

    return time_series, labels, locs


def nifti2timeseries(
    nifti_file,
    Fs,
    subj_id,
    confound_strategy="none",
    masker_type="NiftiLabelsMasker",
    n_rois=100,
    labels_img=None,
    seeds=None,
    radius=None,
    region_names=None,
    standardize=False,
    TS_name=None,
    session=None,
):
    """
    this function is only for single subject and single session data loading
    it uses nilearn maskers to extract ROI signals from nifti files
    and returns a TIME_SERIES object

    Parameters
    ----------
    nifti_file : str
        path to the nifti file
    Fs : float
        sampling frequency of the data
    subj_id : str
        subject ID, must start with 'sub-'
    confound_strategy : str, optional
        strategy for confound regression, by default "none"
    masker_type : str, optional
        type of masker to use, by default "NiftiLabelsMasker"
    n_rois : int, optional
        number of regions of interest to extract, by default 100
    labels_img : str, optional
        path to the labels image, by default None
    seeds : list, optional
        list of tuples (x, y, z) for NiftiSpheresMasker
        by default None
    radius : float, optional
        radius in mm for NiftiSpheresMasker, by default None
    region_names : list, optional
        list of region names for NiftiLabelsMasker or NiftiSpheresMasker,
        by default None
    standardize : bool, optional
        whether to standardize the time series, by default False
    TS_name : str, optional
        name of the time series, by default None
    session : str, optional
        session name, by default None

    For more information on confound_strategy, masker_type, and other parameters,
    see the documentation of the nifti2array function.
    """
    time_series, labels, locs = nifti2array(
        nifti_file=nifti_file,
        confound_strategy=confound_strategy,
        standardize=standardize,
        masker_type=masker_type,
        n_rois=n_rois,
        labels_img=labels_img,
        seeds=seeds,
        radius=radius,
        region_names=region_names,
    )

    assert type(locs) is np.ndarray, "locs must be a numpy array"
    assert type(labels) is list, "labels must be a list"
    assert locs.shape[0] == len(labels), "locs and labels must have the same length"
    assert locs.shape[1] == 3, "locs must have 3 columns"

    # change time_series.shape to (roi, time)
    time_series = time_series.T

    if TS_name is None:
        TS_name = subj_id + " time series"

    BOLD = TIME_SERIES(
        data=time_series,
        subj_id=subj_id,
        Fs=Fs,
        locs=locs,
        node_labels=labels,
        TS_name=TS_name,
        session_name=session,
    )

    return BOLD


def multi_nifti2timeseries(
    nifti_files_list,
    subj_id_list,
    Fs,
    masker_type="NiftiLabelsMasker",
    n_rois=100,
    labels_img=None,
    seeds=None,
    radius=None,
    region_names=None,
    confound_strategy="none",
    standardize=False,
    TS_name=None,
    session=None,
):
    """
    loading data of multiple subjects, but single session, from their nifti files
    """
    BOLD_multi = None
    for subj_id, nifti_file in zip(subj_id_list, nifti_files_list):
        if BOLD_multi is None:
            BOLD_multi = nifti2timeseries(
                nifti_file=nifti_file,
                subj_id=subj_id,
                Fs=Fs,
                confound_strategy=confound_strategy,
                masker_type=masker_type,
                n_rois=n_rois,
                labels_img=labels_img,
                seeds=seeds,
                radius=radius,
                region_names=region_names,
                standardize=standardize,
                TS_name=TS_name,
                session=session,
            )
        else:
            BOLD_multi.concat_ts(
                nifti2timeseries(
                    nifti_file=nifti_file,
                    subj_id=subj_id,
                    Fs=Fs,
                    confound_strategy=confound_strategy,
                    masker_type=masker_type,
                    n_rois=n_rois,
                    labels_img=labels_img,
                    seeds=seeds,
                    radius=radius,
                    region_names=region_names,
                    standardize=standardize,
                    TS_name=TS_name,
                    session=session,
                )
            )
    return BOLD_multi


def load_TS(
    data_root,
    file_name,
    subj_id2load=None,
    task=None,
    session=None,
    run=None,
):
    """
    load a TIME_SERIES object from a .npy file
    if subj_id2load is None, it will load all the subjects
    file_name: name of the file to load
        format example: {subj_id}_{session}_{task}_{run}_time-series.npy
        (keep the {} for the variables)
    """

    if subj_id2load is None:
        SUBJECTS = find_subj_list(data_root)
    else:
        assert "sub-" in subj_id2load, "subj_id2load must start with 'sub-'"
        SUBJECTS = [subj_id2load]

    TS = None
    for subj in SUBJECTS:
        subj_fldr = subj
        # make the file_name
        TS_file = deepcopy(file_name)
        if "{subj_id}" in file_name:
            TS_file = TS_file.replace("{subj_id}", subj)
        if "{task}" in file_name:
            assert task is not None, "task must be provided"
            TS_file = TS_file.replace("{task}", task)
        if "{session}" in file_name:
            assert session is not None, "session must be provided"
            TS_file = TS_file.replace("{session}", session)
        if "{run}" in file_name:
            assert run is not None, "run must be provided"
            TS_file = TS_file.replace("{run}", run)

        try:
            if session is None:
                time_series = np.load(
                    f"{data_root}/{subj_fldr}/{TS_file}", allow_pickle="True"
                ).item()
            else:
                time_series = np.load(
                    f"{data_root}/{subj_fldr}/{session}/{TS_file}",
                    allow_pickle="True",
                ).item()
        except FileNotFoundError:
            print(f"File {TS_file} not found for {subj}")
            continue

        if TS is None:
            TS = time_series
        else:
            try:
                TS.concat_ts(time_series)
            except AssertionError as e:
                # print the error message
                print(f"Error in concatenating time series for {subj}: {e}")
                # raise error with a message and stop the program
                raise Exception(
                    f"Fs of subj {subj} TS is {time_series.Fs} while the group Fs is {TS.Fs}"
                )

    return TS


####################################################################################################################################
