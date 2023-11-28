"""
Implementation of dFC methods.
the parent dFC method class

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import numpy as np
from copy import deepcopy

from ..dfc_utils import SW_downsample, visualize_FCS

################################# BaseDFCMethod class ####################################

"""
todo:
- type annotation
"""

class BaseDFCMethod:

    TF_methods_name_lst = [ \
        'CWT_mag', \
        'CWT_phase_r', \
        'CWT_phase_a', \
        'WTC' \
    ]

    sw_methods_name_lst = [ \
        'pear_corr', \
        'MI', \
        'GraphLasso', \
    ]

    base_methods_name_lst = ['SlidingWindow', 'Time-Freq']

    def __init__(self):
        self.measure_name = ''
        self.is_state_based = bool()
        self._stat = []
        self.TPM = []
        self.params = {}
        self.TS_info_ = {}
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.logs_ = ''

    @property
    def FCS_fit_time(self):
        return self.FCS_fit_time_

    @property
    def dFC_assess_time(self):
        return self.dFC_assess_time_

    @property
    def TS_info(self):
        # info of the time series used to train/estimate FCSs
        return self.TS_info_

    @property
    def is_state_based(self):
        return self.params['is_state_based']

    @property
    def FCS(self):
        return self.FCS_

    # test
    @property
    def FCS_dict(self):
        # returns a dict including FCS matrices

        if not self.is_state_based:
            return None

        C_A = self.FCS
        FCSs = {}
        for k in range(C_A.shape[0]):
            FCSs['FCS'+str(k+1)] = C_A[k,:,:]
            
        return FCSs

    @property
    def info(self):
        return self.params

    @property
    def logs(self):
        print(self.logs_)

    def issame(self, dFC):
        if type(self)==type(dFC):
            for param_name in self.params:
                if self.params[param_name] != dFC.params[param_name]:
                    return False
        else:
            return False
        return True

    #test
    def param_match(self, **param_dict):
        for param in param_dict:
            if param in self.params:
                if type(param_dict[param]) is list:
                    if not self.params[param] in param_dict[param]:
                        return False
                else:
                    if self.params[param]!=param_dict[param]:
                        return False
        return True

    def set_FCS_fit_time(self, time):
        self.FCS_fit_time_ = time

    def set_dFC_assess_time(self, time):
        self.dFC_assess_time_ = time

    def set_mean_activity(self, time_series):
        # mean activity of regions at each state
        if self.is_state_based:
            if 'sw_method' in self.params_name_lst:
                SUBJECTs = time_series.subj_id_lst
                TS_data = None
                for subject in SUBJECTs:
                    subj_TS = time_series.get_subj_ts(subjs_id=subject).data
                    new_TS_data = SW_downsample(data=subj_TS.T, \
                        Fs=time_series.Fs, W=self.params['W'], \
                        n_overlap=self.params['n_overlap'], \
                        tapered_window=self.params['tapered_window'] \
                    ).T
                    if TS_data is None:
                        TS_data = new_TS_data
                    else:
                        TS_data = np.concatenate((TS_data, new_TS_data), axis=1)
            else:
                TS_data = time_series.data
            mean_act = list()
            for i in np.unique(self.Z):
                ids = np.array([int(state==i) for state in self.Z])
                mean_act.append(np.average(TS_data, weights=ids, axis=1))
            self.mean_act = np.array(mean_act)
        else:
            self.mean_act = None

    def estimate_FCS(self, time_series=None):
        pass

    def estimate_dFC(self, time_series=None):
        pass

    def manipulate_time_series4FCS(self, time_series):
        '''
        passing None to params will not change the time series
        num_realization is not implemented yet
        '''

        new_time_series = deepcopy(time_series)

        # SUBJECTs
        if not self.params['num_subj'] is None:
            new_time_series.select_subjs(num_subj=self.params['num_subj'])
        # SPATIAL RESOLUTION
        if not self.params['num_select_nodes'] is None:
            new_time_series.spatial_downsample(num_select_nodes=self.params['num_select_nodes'], rand_node_slct=False)
        # TEMPORAL RESOLUTION
        if not self.params['Fs_ratio'] is None:
            new_time_series.Fs_resample(Fs_ratio=self.params['Fs_ratio'])
        # NORMALIZE
        if self.params['normalization']:
            new_time_series.normalize()
        # NOISE
        if not self.params['noise_ratio'] is None:
            new_time_series.add_noise(noise_ratio=self.params['noise_ratio'], mean_noise=0)
        # NUMBER OF TIME POINTS
        if not self.params['num_time_point'] is None:
            new_time_series.truncate(start_point=0, end_point=self.params['num_time_point']-1)

        self.TS_info_ = new_time_series.info_dict

        return new_time_series

    def manipulate_time_series4dFC(self, time_series):
        '''
        passing None to params will not change the time series
        num_realization is not implemented yet
        '''

        new_time_series = deepcopy(time_series)

        # SPATIAL RESOLUTION
        if not self.params['num_select_nodes'] is None:
            new_time_series.spatial_downsample(num_select_nodes=self.params['num_select_nodes'], rand_node_slct=False)
        # TEMPORAL RESOLUTION
        if not self.params['Fs_ratio'] is None:
            new_time_series.Fs_resample(Fs_ratio=self.params['Fs_ratio'])
        # NORMALIZE
        if self.params['normalization']:
            new_time_series.normalize()
        # NOISE
        if not self.params['noise_ratio'] is None:
            new_time_series.add_noise(noise_ratio=self.params['noise_ratio'], mean_noise=0)
        # NUMBER OF TIME POINTS
        if not self.params['num_time_point'] is None:
            new_time_series.truncate(start_point=0, end_point=self.params['num_time_point']-1)

        return new_time_series
    
    def visualize_states(self):
        pass

    # todo : use FCS_dict func in this func
    def visualize_FCS(
            self,
            normalize=True, fix_lim=True, 
            save_image=False, output_root=None
        ):
        
        visualize_FCS(
            self,
            normalize=normalize, fix_lim=fix_lim, 
            save_image=save_image, output_root=output_root
        )


################################## NEW METHOD ##################################

'''
by : web link

Reference: ##

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:

import needed_toolbox

class method_name(dFC):

    def __init__(self, **params):
        self.FCS_ = []
        self.logs_ = ''

        self.params_name_lst = ['measure_name', 'is_state_based', 'n_states',
            'normalization', 'num_subj', 'num_select_nodes', 'num_time_point',
            'Fs_ratio', 'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None

        self.params['specific_param'] = value
        self.params['measure_name'] = 'method_name'
        self.params['is_state_based'] = True/False
    
    @property
    def measure_name(self):
        return self.params['measure_name'] 

    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # start timing
        tic = time.time()

        # calc FCSs

        # calc self.Z

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

        # calc FCSs and FCS_idx

        # record time
        self.set_dFC_assess_time(time.time() - tic)
            
        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=self.FCS_, FCS_idx=FCS_idx, TS_info=time_series.info_dict)
        return dFC
'''
