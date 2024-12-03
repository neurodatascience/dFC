"""
Implementation of functions for Multi Analysis of dFC.

Created on Dec 3 2024
@author: Mohammad Torabi
"""

from copy import deepcopy

from joblib import Parallel, delayed

from .dfc_methods import *

################################# DATA_LOADER functions ######################################


def create_measure_obj(MEASURES_name_lst, **params):

    MEASURES_lst = list()
    for MEASURES_name in MEASURES_name_lst:

        ###### CAP ######
        if MEASURES_name == "CAP":
            measure = CAP(**params)

        ###### CONTINUOUS HMM ######
        if MEASURES_name == "ContinuousHMM":
            measure = HMM_CONT(**params)

        ###### WINDOW_LESS ######
        if MEASURES_name == "Windowless":
            measure = WINDOWLESS(**params)

        ###### SLIDING WINDOW ######
        if MEASURES_name == "SlidingWindow":
            measure = SLIDING_WINDOW(**params)

        ###### TIME FREQUENCY ######
        if MEASURES_name == "Time-Freq":
            measure = TIME_FREQ(**params)

        ###### SLIDING WINDOW + CLUSTERING ######
        if MEASURES_name == "Clustering":
            measure = SLIDING_WINDOW_CLUSTR(**params)

        ###### DISCRETE HMM ######
        if MEASURES_name == "DiscreteHMM":
            measure = HMM_DISC(**params)

        MEASURES_lst.append(measure)

    return MEASURES_lst


def measures_initializer(MEASURES_name_lst, params_methods, alter_hparams):
    """
    - this will test values in alter_hparams other than
        values already in params_methods. values in params_methods
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
    """

    # a list of MEASURES with default parameter values
    MEASURES_lst = create_measure_obj(
        MEASURES_name_lst=MEASURES_name_lst, **params_methods
    )

    # adding MEASURES with alternative parameter values
    hyper_param_info = {}
    hyper_param_info["default_values"] = params_methods
    for hyper_param in alter_hparams:
        for value in alter_hparams[hyper_param]:
            params = deepcopy(params_methods)
            params[hyper_param] = value
            hyper_param_info[hyper_param + "_" + str(value)] = deepcopy(params)
            new_MEASURES = create_measure_obj(
                MEASURES_name_lst=MEASURES_name_lst, **params
            )
            for new_measure in new_MEASURES:
                flag = 0
                for MEASURE in MEASURES_lst:
                    if new_measure.issame(MEASURE):
                        flag = 1
                if flag == 0:
                    MEASURES_lst.append(new_measure)

    return MEASURES_lst, hyper_param_info


def get_SB_MEASURES_lst(MEASURES_lst):
    """returns state_based measures"""
    SB_MEASURES = list()
    for measure in MEASURES_lst:
        if measure.is_state_based:
            SB_MEASURES.append(measure)
    return SB_MEASURES


def get_DD_MEASURES_lst(MEASURES_lst):
    """returns data_driven measures"""
    DD_MEASURES = list()
    for measure in MEASURES_lst:
        if not measure.is_state_based:
            DD_MEASURES.append(measure)
    return DD_MEASURES


def estimate_group_FCS(time_series, MEASURES_lst, n_jobs=None, verbose=0, backend="loky"):

    SB_MEASURES_lst = get_SB_MEASURES_lst(MEASURES_lst)
    if n_jobs is None:
        SB_MEASURES_lst_NEW = list()
        for measure in SB_MEASURES_lst:
            SB_MEASURES_lst_NEW.append(measure.estimate_FCS(time_series=time_series))
    else:
        SB_MEASURES_lst_NEW = Parallel(
            n_jobs=n_jobs,
            verbose=verbose,
            backend=backend,
        )(
            delayed(measure.estimate_FCS)(time_series=time_series)
            for measure in SB_MEASURES_lst
        )
    MEASURES_fit_lst = get_DD_MEASURES_lst(MEASURES_lst) + SB_MEASURES_lst_NEW

    return MEASURES_fit_lst


##################### dFC ASSESSMENT ######################


def group_dFC_assess(
    time_series, MEASURES_fit_lst, n_jobs=None, verbose=0, backend="loky"
):
    """
    assess dFC for all subjects using all measures
    and a single time_series
    """
    SUBJECTs = time_series.subj_id_lst

    OUT = list()
    for subject in SUBJECTs:
        OUT.append(
            subj_lvl_dFC_assess(
                time_series=time_series.get_subj_ts(subjs_id=subject),
                MEASURES_fit_lst=MEASURES_fit_lst,
                n_jobs=n_jobs,
                verbose=verbose,
                backend=backend,
            )
        )

    return OUT


def subj_lvl_dFC_assess(
    time_series, MEASURES_fit_lst, n_jobs=None, verbose=0, backend="loky"
):
    """
    assess dFC for a single subject using all measures and a
    single time_series
    """

    dFC_dict = {}

    if n_jobs is None:
        dFC_lst = list()
        for measure in MEASURES_fit_lst:
            dFC_lst.append(measure.estimate_dFC(time_series=time_series))
    else:
        dFC_lst = Parallel(
            n_jobs=n_jobs,
            verbose=verbose,
            backend=backend,
        )(
            delayed(measure.estimate_dFC)(time_series=time_series)
            for measure in MEASURES_fit_lst
        )

    dFC_dict["dFC_lst"] = dFC_lst

    return dFC_dict


##############################################################################################################
