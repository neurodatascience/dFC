
"""
dFC class

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import numpy as np

from .dfc_utils import node_info2network, node_labels2networks, rank_norm_dFC_dict, visualize_conn_mat_dict, SW_downsample

################################# DFC class ######################################

"""
Parameters
    ----------
    TR_array : an array labeling 
        timepoints by their TRs
        starts from 0

Variables
    ----------
    FCSs : Functional Connecitivity 
        States patterns
    FCS_idx : the  index of the 
        FCS that corresponds to each 
        timepoint
    

todo:
- 
"""

class DFC():
    def __init__(self, measure=None):

        # assert not measure is None, \
        #     "measure arg must be provided."
        self.measure_ = measure
        self.FCSs_ = None # is a dict
        self.FCS_idx_ = None # is a dict
        # info of the time series used for dFC estimation
        self.TS_info_ = None
        self.TR_array_ = None
        self.n_regions_ = None
        self.n_time_ = -1
    
    @classmethod
    def from_numpy(cls, array=None):
        pass

    @property
    def measure(self):
        return self.measure_

    @property
    def TR_array(self):
        return self.TR_array_.astype(int)

    @property
    def TR_keys(self):
        TRs_lst = list()
        for TR in self.TR_array:
            TRs_lst.append('TR'+str(TR))
        return TRs_lst

    @property
    def n_regions(self):
        return self.n_regions_

    @property
    def n_time(self):
        return self.n_time_

    # test this
    @property
    def FCSs(self):
        return self.FCSs_

    # test this
    @property
    def FCS_idx(self):
        return self.FCS_idx_

    # test this
    @property
    def FCS_idx_array(self):
        return np.array([int(self.FCS_idx[TR][self.FCS_idx[TR].find('S')+1:])-1 for TR in self.FCS_idx])

    @property
    def TS_info(self):
        # info of the time series used for dFC estimation
        return self.TS_info_

    
    # test
    def state_TC(self, TRs=None, \
        state_match=False, state_match_dict=None \
        ):
        # returns a np array of state indices over TRs in TRs

        if TRs is None:
            TRs = self.TR_array

        if not type(TRs[0]) is str:
            TRs_lst = list()
            for TR in TRs:
                TRs_lst.append('TR'+str(TR))
        else:
            TRs_lst = TRs

        state_TC = list()
        for key in self.FCS_idx:
            if key in TRs_lst:
                state = self.FCS_idx[key]
                if state_match:
                    match = state_match_dict['FCS_match'][state]['match']
                    state_TC.append(int(match[match.find('FCS')+3:]))
                else:
                    state_TC.append(int(state[state.find('FCS')+3:]))

        state_TC = np.array(state_TC)
        return state_TC
    
    # test
    def state_act_dict(self, TRs=None):
        # returns a dict including each FCS and its activation times
        # the TRs arg can be used to set a common set of TRs

        if TRs is None:
            TRs = self.TR_array

        TRs_lst = list()
        for TR in TRs:
            TRs_lst.append('TR'+str(TR))

        state_act_dict = {}
        state_act_dict['state_TC'] = {}
        state_act_dict['TR_array'] = TRs
        for FCS_key in self.FCSs:
            state_act_dict['state_TC'][FCS_key] = {}
            state_act_dict['state_TC'][FCS_key]['FCS'] = self.FCSs[FCS_key]
            state_act_dict['state_TC'][FCS_key]['act_TC'] = np.zeros((len(TRs),))
        t=0
        for TR in self.FCS_idx:
            if TR in TRs_lst:
                state_act_dict['state_TC'][self.FCS_idx[TR]]['act_TC'][t] = 1
                t=t+1
        assert t==len(TRs), 'error!'

        return state_act_dict

    # test
    def dFC2dict(self, TRs=None):
        # return dFC samples as a dictionary
        if TRs is None:
            TRs = self.TR_array
        if type(TRs) is list:
            TRs = np.array(TRs)
        TRs = TRs.astype(int)
        dFC_mat = self.get_dFC_mat(TRs=TRs)
        dFC_dict = {}
        for k, TR in enumerate(TRs):
            dFC_dict['TR'+str(TR)] = dFC_mat[k, :, :]
        return dFC_dict

    # test this
    def get_dFC_mat(self, TRs=None, num_samples=None):
        '''
        get dFC matrices corresponding to 
        the specified TRs 
        TRs should be list/ndarray not necessarily in order ?
        if num_samples specified, it will downsample 
        TRs to reach that number of samples and will also
        return picked TRs
        if num_samples > len(TRs) -> picks all TRs
        '''

        if TRs is None:
            TRs = self.TR_array

        if type(TRs) is np.int32 or type(TRs) is np.int64 or type(TRs) is int:
            TRs = [TRs]

        if not num_samples is None:
            if num_samples < len(TRs):
                TRs = TRs[np.linspace(0, len(TRs), num_samples, endpoint=False, dtype=int)]

        dFC_mat = list()
        for TR in TRs:
            dFC_mat.append(self.FCSs[self.FCS_idx['TR'+str(TR)]])

        dFC_mat = np.array(dFC_mat)

        if num_samples is None:
            return dFC_mat
        else:
            return dFC_mat, TRs

    def SWed_dFC_mat(self, W=None, n_overlap=None, tapered_window=False):
        '''
        the time samples will be picked after 
        averaging over a window which slides
        W is in sec
        '''
        dFC_mat = self.get_dFC_mat()

        # method not applicable to SW-based methods
        if 'sw_method' in self.measure.info:
            return dFC_mat

        dFC_mat_new = SW_downsample(data=dFC_mat, \
            Fs=self.TS_info['Fs'], W=W, n_overlap=n_overlap, tapered_window=tapered_window \
        )
        
        return dFC_mat_new


    def set_dFC(self, FCSs, FCS_idx=None, TS_info=None, TR_array=None):
        
        if len(FCSs.shape)==2:
            FCSs = np.expand_dims(FCSs, axis=0)

        if FCS_idx is None:
            # usually for state-free methods like sliding window when we don't have FCSs
            # we consider each FC a FCS
            FCS_idx = np.arange(start=0, stop=FCSs.shape[0], step=1, dtype=int)

        if type(FCS_idx) is list:
            FCS_idx = np.array(FCS_idx)

        if len(FCS_idx.shape)>1:
            FCS_idx = np.squeeze(FCS_idx)
        
        assert FCSs.shape[1] == FCSs.shape[2], \
                "FC matrices must be square."

        assert self.n_time==-1, \
            'why n_time is not -1 ? Are you adding a dFC to an existing dFC ?'
        
        if TR_array is None:
            # self.n_time is -1 at first. if it is not -1, it means that a dFC is already set and
            # we are adding a new dFC to it. 
            TR_array = np.arange(start=self.n_time+1, stop=self.n_time+len(FCS_idx)+1, step=1, dtype=int)

        assert np.sum(np.abs(np.sort(TR_array)-TR_array))==0.0, \
            'TRs not sorted !'

        # the input FCS_idx is ranged from 0 to len(FCS)-1 but we shift it to 1 to len(FCS)
        self.FCSs_ = {}
        for i, FCS in enumerate(FCSs):
            self.FCSs_['FCS'+str(i+1)] = FCS

        self.FCS_idx_ = {}
        for i, idx in enumerate(FCS_idx):
            self.FCS_idx_['TR'+str(TR_array[i])] = 'FCS'+str(idx+1)

        self.TS_info_ = TS_info
        self.n_regions_ = FCSs.shape[1]
        self.n_time_ = len(self.FCS_idx_)
        self.TR_array_ = TR_array


    def visualize_dFC(self, TRs=None, normalize=False,
                      show_networks=False,
                      rank_norm=False,
                      threshold=0.0, 
                      fix_lim=False,
                      save_image=False, fig_name=None, 
    ):

        assert not self.measure is None, \
            'Measure is not provided.'

        if TRs is None:
            TRs = self.TR_array

        if show_networks:
            if 'nodes_info' in self.TS_info:
                node_networks = node_info2network(self.TS_info['nodes_info'])
            elif 'node_labels' in self.TS_info:
                node_networks = node_labels2networks(self.TS_info['node_labels'])
        else:
            node_networks = None

        if rank_norm:
            dFC_dict = rank_norm_dFC_dict(self.dFC2dict(TRs=TRs))
            cmap = 'plasma'
            center_0 = False
        else:
            dFC_dict = self.dFC2dict(TRs=TRs)
            cmap = 'seismic'
            center_0 = True
        
        visualize_conn_mat_dict(data=dFC_dict, 
            title=self.measure.measure_name+' dFC', 
            fix_lim=fix_lim, normalize=normalize,
            node_networks=node_networks,
            cmap=cmap, center_0=center_0,
            save_image=save_image, 
            output_root=fig_name, 
        )
