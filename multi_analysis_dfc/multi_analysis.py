# -*- coding: utf-8 -*-
"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import numpy as np
from joblib import Parallel, delayed
from copy import deepcopy

from .dfc_methods import *
from .dfc_utils import get_subj_ts_dict, common_subj_lst

############################# MultiAnalysis class ################################

"""

todo:
- 
"""

class MultiAnalysis:
    # if self.n_jobs is None => no parallelization

    def __init__(self, analysis_name='', **params):

        self.analysis_name = analysis_name
        
        self.params = params
        if not 'n_jobs' in self.params:
            self.params['n_jobs'] = -1
        if not 'verbose' in self.params:
            self.params['verbose'] = 1
        if not 'backend' in self.params:
            self.params['backend'] = 'loky'

        self.MEASURES_lst_ = None
        self.MEASURES_fit_lst_ = []
        self.MEASURES_name_lst = []
        self.params_methods = {}
        self.alter_hparams = {}
        self.hyper_param_info = {}

    @property
    def MEASURES_lst(self):
        assert not self.MEASURES_lst_ is None, \
            'first set the MEASURES_lst!'
        return self.MEASURES_lst_

    @property
    def MEASURES_fit_lst(self):
        return self.MEASURES_fit_lst_

    def set_MEASURES_lst(self, MEASURES_lst):
        self.MEASURES_lst_ = MEASURES_lst

    def set_MEASURES_fit_lst(self, MEASURES_fit_lst):
        self.MEASURES_fit_lst_ = MEASURES_fit_lst

    def measures_initializer(self, MEASURES_name_lst, params_methods, alter_hparams):

        '''
        - this will test values in hyper_params other than
            values already in self.params. values in self.params 
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
        '''

        self.MEASURES_name_lst = MEASURES_name_lst
        self.params_methods = params_methods
        self.alter_hparams = alter_hparams

        # a list of MEASURES with default parameter values
        MEASURES_lst = self.create_measure_obj(MEASURES_name_lst=MEASURES_name_lst, **params_methods)

        # adding MEASURES with alternative parameter values
        hyper_param_info = {}
        hyper_param_info['default_values'] = params_methods
        for hyper_param in alter_hparams:
            for value in alter_hparams[hyper_param]:
                params = deepcopy(params_methods)
                params[hyper_param] = value
                hyper_param_info[hyper_param+'_'+str(value)] = deepcopy(params)
                new_MEASURES = self.create_measure_obj(MEASURES_name_lst=MEASURES_name_lst, **params)
                for new_measure in new_MEASURES:
                    flag=0
                    for MEASURE in MEASURES_lst:
                        if new_measure.issame(MEASURE):
                            flag=1
                    if flag==0:
                        MEASURES_lst.append(new_measure)

        self.hyper_param_info = hyper_param_info

        return MEASURES_lst

    def create_measure_obj(self, MEASURES_name_lst, **params):

        MEASURES_lst = list()
        for MEASURES_name in MEASURES_name_lst:

            ###### CAP ######
            if MEASURES_name=='CAP':
                measure = CAP(**params)

            ###### CONTINUOUS HMM ######
            if MEASURES_name=='ContinuousHMM':
                measure = HMM_CONT(**params)

            ###### WINDOW_LESS ######
            if MEASURES_name=='Windowless':
                measure = WINDOWLESS(**params)

            ###### SLIDING WINDOW ######
            if MEASURES_name=='SlidingWindow':
                measure = SLIDING_WINDOW(**params)

            ###### TIME FREQUENCY ######
            if MEASURES_name=='Time-Freq':
                measure = TIME_FREQ(**params)

            ###### SLIDING WINDOW + CLUSTERING ######
            if MEASURES_name=='Clustering':
                measure = SLIDING_WINDOW_CLUSTR(**params)

            ###### DISCRETE HMM ######
            if MEASURES_name=='DiscreteHMM':
                measure = HMM_DISC(**params)

            MEASURES_lst.append(measure)

        return MEASURES_lst

    def SB_MEASURES_lst(self, MEASURES_lst): # returns state_based measures
        SB_MEASURES = list()
        for measure in MEASURES_lst:
            if measure.is_state_based:
                SB_MEASURES.append(measure)
        return SB_MEASURES

    def DD_MEASURES_lst(self, MEASURES_lst): # returns data_driven measures
        DD_MEASURES = list()
        for measure in MEASURES_lst:
            if not measure.is_state_based:
                DD_MEASURES.append(measure)
        return DD_MEASURES

    ##################### FCS ESTIMATION ######################

    def estimate_group_FCS(self, time_series_dict):

        # time_series_dict is a dict of time_series

        for session in time_series_dict:

            time_series = time_series_dict[session]
            SB_MEASURES_lst = self.SB_MEASURES_lst(self.MEASURES_lst)
            if self.params['n_jobs'] is None:
                SB_MEASURES_lst_NEW = list()
                for measure in SB_MEASURES_lst:
                    SB_MEASURES_lst_NEW.append( \
                        measure.estimate_FCS(time_series=time_series) \
                        )
            else:
                SB_MEASURES_lst_NEW = Parallel( \
                    n_jobs=self.params['n_jobs'], verbose=self.params['verbose'], backend=self.params['backend'])( \
                    delayed(measure.estimate_FCS)(time_series=time_series) \
                        for measure in SB_MEASURES_lst)
            self.MEASURES_fit_lst_[session] = self.DD_MEASURES_lst(self.MEASURES_lst) + SB_MEASURES_lst_NEW

    ##################### dFC ASSESSMENT ######################

    def group_dFC_assess(self, time_series_dict):

        # time_series_dict is a dict of time_series

        SUBJ_s_dFC_dict = {}
        
        SUBJECTs = common_subj_lst(time_series_dict) 

        if self.params['n_jobs'] is None:
            OUT = list()
            for subject in SUBJECTs:
                OUT.append( \
                    self.subj_lvl_dFC_assess( \
                    time_series_dict=get_subj_ts_dict(time_series_dict, subjs_id=subject), \
                    ))
        else:
            OUT = Parallel( \
                        n_jobs=self.params['n_jobs'], \
                        verbose=self.params['verbose'], \
                        backend=self.params['backend'])( \
                    delayed(self.subj_lvl_dFC_assess)( \
                        time_series_dict=get_subj_ts_dict(time_series_dict, subjs_id=subject), \
                        ) \
                        for subject in SUBJECTs)
        
        return OUT

    def subj_lvl_dFC_assess(self, time_series_dict):

        # time_series_dict is a dict of time_series

        dFC_dict = {}
        # dFC_corr_assess_dict = {}

        if self.params['n_jobs'] is None:
            dFC_lst = list()
            for measure in self.MEASURES_fit_lst_:
                dFC_lst.append( \
                    measure.estimate_dFC(time_series=time_series_dict[measure.params['session']]) \
                )
        else:
            dFC_lst = Parallel( \
                n_jobs=self.params['n_jobs'], verbose=self.params['verbose'], backend=self.params['backend'])( \
                delayed(measure.estimate_dFC)(time_series=time_series_dict[measure.params['session']]) \
                    for measure in self.MEASURES_fit_lst_)

        dFC_dict['dFC_lst'] = dFC_lst

        return dFC_dict

##############################################################################################################