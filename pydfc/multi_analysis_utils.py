"""
Implementation of functions for Multi Analysis of dFC.

Created on Dec 3 2024
@author: Mohammad Torabi
"""

import importlib
import inspect
import pkgutil
import warnings
from copy import deepcopy

from joblib import Parallel, delayed

from . import dfc_methods as _dfc_pkg
from .dfc_methods import *

# cache for discovered measures: map measure_name -> class
_MEASURE_REGISTRY = None


def _build_measure_registry():
    global _MEASURE_REGISTRY
    if _MEASURE_REGISTRY is not None:
        return _MEASURE_REGISTRY

    registry = {}
    try:
        for finder in pkgutil.iter_modules(_dfc_pkg.__path__):
            mod_name = f"{_dfc_pkg.__name__}.{finder.name}"
            try:
                module = importlib.import_module(mod_name)
            except Exception:
                warnings.warn(f"Could not import module {mod_name}; skipping.")
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                try:
                    # ensure class originates from dfc_methods package
                    if not obj.__module__.startswith(_dfc_pkg.__name__):
                        continue
                    from .dfc_methods.base_dfc_method import BaseDFCMethod

                    if not issubclass(obj, BaseDFCMethod) or obj is BaseDFCMethod:
                        continue
                except Exception:
                    continue

                # class-level method name is required for stable discovery
                name = getattr(obj, "MEASURE_NAME", None)
                if name:
                    registry[name] = obj
                else:
                    warnings.warn(
                        f"{obj.__module__}.{obj.__name__} has no MEASURE_NAME; skipping."
                    )
    except Exception:
        warnings.warn("Failed to iterate dfc_methods package for discovery.")

    _MEASURE_REGISTRY = registry
    return _MEASURE_REGISTRY


################################# DATA_LOADER functions ######################################


def create_measure_obj(MEASURES_name_lst, **params):
    """
    Auto-discover dFC method classes under `pydfc.dfc_methods` and
    instantiate them with `**params` based on their `measure_name`.
    """

    registry = _build_measure_registry()
    MEASURES_lst = []
    for MEASURES_name in MEASURES_name_lst:
        cls = registry.get(MEASURES_name)
        if cls is None:
            raise ValueError(f"Unknown dFC measure name: {MEASURES_name}")
        try:
            measure = cls(**params)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate measure {MEASURES_name}: {e}")
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

    if n_jobs is None:
        MEASURES_fit_lst = list()
        for measure in MEASURES_lst:
            MEASURES_fit_lst.append(measure.estimate_FCS(time_series=time_series))
    else:
        MEASURES_fit_lst = Parallel(
            n_jobs=n_jobs,
            verbose=verbose,
            backend=backend,
        )(
            delayed(measure.estimate_FCS)(time_series=time_series)
            for measure in MEASURES_lst
        )

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
