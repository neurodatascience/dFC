
"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from copy import deepcopy
import os

from .dfc_utils import print_dict

################################# Parameters ####################################

## visualization parameters
fig_dpi = 120
fig_bbox_inches = 'tight'
fig_pad = 0.1
show_title = False
save_fig_format = 'png'

################################# TIME_SERIES class ######################################

"""

todo:
- select nodes for visualizer
- default node list is chosen by arange !
"""

class TIME_SERIES():
    def __init__(self, data=None, subj_id=None, Fs=None, time_array=None, \
                locs=None, node_labels=None, TS_name='', session_name=''):
        
        '''
        subj_id is an id to identify the subjects
        all properties are applied to every subject separately
        for instance interval applies to TS of each subj separately

        time_array of all subjects must be equal
        '''

        assert (not data is None) and (not Fs is None) and (not subj_id is None), \
            "data, subj_id, and Fs args must be provided."
        
        assert type(locs) is np.ndarray, 'locs must be a numpy array'
        assert type(node_labels) is list, 'node_labels must be a list'
        assert locs.shape[0] == len(node_labels), 'locs and node_labels must have the same length'
        assert locs.shape[1] == 3, 'locs must have 3 columns'

        self.data_dict_ = {}
        self.data_dict_[subj_id] = {}
        self.data_dict_[subj_id]['data'] = data 
        self.data_ = None
        self.Fs_ = Fs
        self.Fs_ratio_ = 1.00
        self.noise_ratio = 0.0
        self.normalized = False
        self.TS_name_ = TS_name
        self.session_name_ = session_name
        self.n_regions_ = data.shape[0]
        self.n_time_ = data.shape[1]

        # assert self.n_regions_ < self.n_time_, \
        #     "Probably you have to transpose the time_series."

        if time_array is None:
            self.time_array_ = 1/self.Fs_ + np.arange(0, data.shape[1]/self.Fs_, 1/self.Fs_)
        else:
            self.time_array_ = time_array

        self.locs_ = locs
        self.node_labels_ = node_labels

        self.interval_ = np.arange(0, self.n_time_, dtype=int)
        self.nodes_selection_ = list(range(self.n_regions_))

    @property
    def info(self):
        print_dict(self.info_dict)

    @property
    def info_dict(self):
        info_dict = {}
        info_dict['n_time'] = self.n_time
        info_dict['n_regions'] = self.n_regions
        info_dict['Fs'] = self.Fs
        info_dict['Fs_ratio'] = self.Fs_ratio_
        info_dict['noise_ratio'] = self.noise_ratio
        info_dict['nodes_lst'] = self.nodes_lst
        info_dict['node_labels'] = self.node_labels
        info_dict['nodes_locs'] = self.locs
        info_dict['subj_id_lst'] = self.subj_id_lst
        info_dict['interval'] = self.interval
        info_dict['time'] = self.time

        return info_dict

    @property
    def data_dict(self):
        return self.data_dict_

    @property
    def data(self):
        if self.data_ is None:
            self.update_data()
        return self.data_

    def update_data(self):
        # after any change in data_dict, self.data_ is 
        # set to None and needs an update before being used
        data = None
        for subj in self.data_dict:
            if data is None:
                data = self.data_dict[subj]['data']
            else:
                data = np.concatenate((data, self.data_dict[subj]['data']), axis=1)
        self.data_ = data
        return

    @property
    def subj_id_lst(self):
        return [subj_id for subj_id in self.data_dict]

    @property
    def nodes_lst(self):
        # output shape is (n_region,) 
        return np.array(self.nodes_selection_)
        
    @property
    def interval(self):
        # output shape is (n_time,) 
        return self.interval_

    @property
    def locs(self):
        if self.locs_ is None:
            return None
        else:
            return self.locs_[self.nodes_lst, :]

    @property
    def node_labels(self):
        if self.node_labels_ is None:
            return None
        else:
            return [self.node_labels_[i] for i in self.nodes_lst] 

    @property
    def Fs(self):
        return self.Fs_ * self.Fs_ratio_

    @property
    def n_time(self):
        return len(self.time)

    @property
    def n_regions(self):
        return len(self.nodes_lst)

    @property
    def time(self):
        return self.time_array_[self.interval]

    @property
    def TS_name(self):
        return self.TS_name_

    def get_subj_ts(self, subjs_id=None):
        """
        you can select time samples by their subj_id
        ! be careful about the original properties of TS hidden in new TS
        node selection will be kept.

        subjs_id is one str or a list of str
        """

        flag = 0
        if not type(subjs_id) is list:
            subjs_id = [subjs_id]
            flag = 1

        new_TS = deepcopy(self)

        SUBJECTS = [subj_id for subj_id in new_TS.data_dict_]
        for subj in SUBJECTS:
            if not subj in subjs_id:
                new_TS.data_dict_.pop(subj, None)

        if flag == 1:
            new_TS.TS_name_ = self.TS_name+' subject '+subjs_id[0]

        return new_TS


    def append_ts(self, new_time_series, time_array=None, subj_id=None):
        # append new time series to existing ones
        # truncate and node selection , etc will be automatically applied to new TS;
        # However, at first the new TS must have the same properties as the original properties of 
        # the existing TSs 

        assert self.n_regions_ == new_time_series.shape[0], \
            "Number of nodes mismatch."

        assert not subj_id is None, \
            "subj_id must be provided."

        assert not subj_id in self.data_dict_, \
            "subj_id already exists."

        self.data_dict_[subj_id] = {}

        if not time_array is None:
            assert self.time_array_ == time_array, \
                'time array mismatch!'

        self.data_dict_[subj_id]['data'] = new_time_series

        self.data_ = None


    def truncate(self, start_time=None, end_time=None, start_point=None, end_point=None):

        # truncates TS of every subj separately
        # based on either time or samples
        # if all None -> whole time_series
        #check if not out of total interval

        start = 0
        end = self.n_time

        if not start_point is None:
            start = start_point
        
        if not end_point is None:
            end = end_point + 1

        if not start_time is None:
            start = np.argwhere(self.time_array_>=start_time)[0,0]

        if not end_time is None:
            end = np.argwhere(self.time_array_<=end_time)[-1,0] + 1

        if start > self.interval_[0] or end < self.interval_[-1]:
            # make sure the interval is not out of range
            start = max(start, 0)
            end = min(end, self.n_time)

            self.interval_ = np.arange(start, end, dtype=int)
            
            for subj in self.data_dict_:
                self.data_dict_[subj]['data'] = self.data_dict_[subj]['data'][:, self.interval]

            self.data_ = None

    def normalize(self):
        # normalization
        if not self.normalized:
            for subj_id in self.data_dict:
                new_time_series = self.data_dict[subj_id]['data']
                for n in range(new_time_series.shape[0]):
                    new_time_series[n, :] = new_time_series[n, :] - np.mean(new_time_series[n, :])
                    new_time_series[n, :] = np.divide(new_time_series[n, :], np.std(new_time_series[n, :]))
                self.data_dict_[subj_id]['data'] = new_time_series

            self.normalized = True
            self.data_ = None

    def spatial_downsample(self, num_select_nodes, rand_node_slct=False):
        if num_select_nodes < self.n_regions:
            if rand_node_slct:
                np.random.seed(0)
                nodes_idx = np.random.choice(range(self.n_regions), size=num_select_nodes, replace=False)
                nodes_idx.sort()
            else:
                # nodes_idx = np.array(list(range(self.num_select_nodes)))
                nodes_idx = np.arange(0, self.n_regions, np.ceil(self.n_regions/num_select_nodes), dtype=int)
            self.select_nodes(nodes_idx=nodes_idx)

            self.data_ = None

    def Fs_resample(self, Fs_ratio=None):
        # downsample frequency
        if Fs_ratio != 1 and self.Fs_ratio_ == 1:
            for subj_id in self.data_dict:
                new_time_series = self.data_dict[subj_id]['data']
                downsampled_time_series = np.zeros((new_time_series.shape[0], int(new_time_series.shape[1]*Fs_ratio)))
                for n in range(new_time_series.shape[0]):
                    downsampled_time_series[n, :] =  signal.resample(new_time_series[n, :], int(new_time_series.shape[1]*Fs_ratio))
                self.data_dict_[subj_id]['data'] = downsampled_time_series
            self.Fs_ratio_ = Fs_ratio

            start_time = self.time_array_[self.interval_[0]]
            end_time = self.time_array_[self.interval_[-1]]

            _, self.time_array_ =  signal.resample(new_time_series[n, :], int(new_time_series.shape[1]*Fs_ratio), t=self.time_array_)

            start = np.argwhere(self.time_array_>=start_time)[0,0]
            end = np.argwhere(self.time_array_<=end_time)[-1,0] + 1
            self.interval_ = np.arange(start, end, dtype=int)

            self.data_ = None

    def add_noise(self, noise_ratio, mean_noise=0):
        # adding noise perturbation 
        if noise_ratio > 0 and self.noise_ratio == 0 :
            for subj_id in self.data_dict:
                new_time_series = self.data_dict[subj_id]['data']
                power_signal = np.mean(new_time_series ** 2)
                power_noise = power_signal * noise_ratio
                new_time_series += np.random.normal(mean_noise, np.sqrt(power_noise), (new_time_series.shape[0], new_time_series.shape[1]))
                self.data_dict_[subj_id]['data'] = new_time_series

            self.noise_ratio = noise_ratio
            self.data_ = None

    def select_subjs(self, num_subj):
        # selects the first num_subj subjects in self.subj_id_lst

        SUBJECTS = [subj_id for subj_id in self.data_dict_]
        if num_subj < len(SUBJECTS):
            for subj_id in SUBJECTS:
                if not subj_id in SUBJECTS[:num_subj]:
                    self.data_dict_.pop(subj_id, None)

            self.data_ = None

    def select_nodes(self, nodes_idx=None):
        # select the nodes indexed by numbers in nodes_idx. nodes_idx is a numpy 1D array
        # if nodes_idx is None -> all the nodes will be considered (resets node selection)
        # if nodes_idx is not sorted, it can be used to reorder the nodes
        # this function can be used only once (you cannot select the nodes again)

        if nodes_idx is None:
            self.nodes_selection_ = np.arange(0, self.n_regions_, 1, dtype=int)
        else:
            self.nodes_selection_ = nodes_idx  

        for subj in self.data_dict_:
            self.data_dict_[subj]['data'] = self.data_dict_[subj]['data'][self.nodes_lst, :]

        self.data_ = None

        

    def visualize(self, start_time=None, end_time=None, 
        nodes_lst=None, 
        save_image=False, output_root=None):
        '''
        time in seconds
        nodes_lst is a list of indices
        '''

        start = 0
        end = self.n_time

        if not start_time is None:
            start = np.argwhere(self.time>=start_time)[0,0]

        if not end_time is None:
            end = np.argwhere(self.time<=end_time)[-1,0] + 1
        
        interval = list(range(start, end))

        if nodes_lst is None:
            nodes_lst = self.nodes_lst
        else:
            nodes_lst = np.array(nodes_lst)[:, np.newaxis]

        plt.figure(figsize=(15, 5))
        plt.plot(self.time[interval], self.data[nodes_lst, interval].T)
        plt.xlabel('time (sec)')
        title = self.TS_name_ + ' ' + self.session_name_
        if show_title:
            plt.title(title)
        if save_image:
            folder = output_root[:output_root.rfind('/')]
            file_name = title.replace(" ", "_")
            if not os.path.exists(folder):
                os.makedirs(folder)
            plt.savefig(output_root+file_name+'.'+save_fig_format, 
                dpi=fig_dpi, bbox_inches=fig_bbox_inches, pad_inches=fig_pad, format=save_fig_format
            ) 
            plt.close()
        else:
            plt.show()