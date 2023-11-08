"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import numpy as np
import time
from hmmlearn import hmm

from .base_dfc_method import BaseDFCMethod
from ..time_series import TIME_SERIES
from ..dfc import DFC


################################# HMM Continuous ###############################

"""
by hmmlearn

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
- number of iter?
- ValueError: 'covars' must be symmetric, positive-definite
"""

class HMM_CONT(BaseDFCMethod):

    def __init__(self, **params):
        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'n_states', 'hmm_iter',
            'normalization', 'num_subj', 'num_select_nodes', 'num_time_point',
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None

        self.params['measure_name'] = 'ContinuousHMM'
        self.params['is_state_based'] = True

    @property
    def measure_name(self):
        return self.params['measure_name'] 

    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        # start timing
        tic = time.time()

        time_series = self.manipulate_time_series4FCS(time_series)

        Models, Scores = [], []
        for i in range(self.params['hmm_iter']):
            model = hmm.GaussianHMM(n_components=self.params['n_states'], covariance_type="full")
            model.fit(time_series.data.T) 
            score = model.score(time_series.data.T)
            Models.append(model)
            Scores.append(score)
            
        self.hmm_model = Models[np.argmax(Scores)]
        self.Z = self.hmm_model.predict(time_series.data.T)
        self.means_ = self.hmm_model.means_
        self.FCS_ = self.hmm_model.covars_ 
        self.TPM = self.hmm_model.transmat_
        self.pi = self.hmm_model.startprob_

        # mean activation of states
        self.set_mean_activity(time_series)

        # record time
        self.set_FCS_fit_time(time.time() - tic)

        return self

    def estimate_dFC(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."
        
        assert len(time_series.subj_id_lst)==1, \
            'this function takes only one subject as input.'

        time_series = self.manipulate_time_series4dFC(time_series)

        # start timing
        tic = time.time()

        Z = self.hmm_model.predict(time_series.data.T)

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=self.FCS_, FCS_idx=Z, TS_info=time_series.info_dict)

        return dFC
    
################################################################################