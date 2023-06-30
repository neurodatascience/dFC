
"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import numpy as np
import hdf5storage
import scipy.io as sio
import os

from dfc_utils import intersection
from .time_series import TIME_SERIES

################################# DATA_LOADER class ######################################

"""
Parameters
    ----------
    DATA_type : ICA , Gordon , sample , 
        or simulated 

Variables
    ----------
    Var1 : Functional Connecitivity 
        States patterns
    Var2 : the  index of the 
        FCS that corresponds to each 
        timepoint
    

todo:
- add SUBJECTS of sample and simulation DATA_type
- factors are implemented only for gordon data type
"""

class DATA_LOADER():
    def __init__(self, **params):

        self.params = params

        if self.params['DATA_type']=='ICA': 
            self.BOLD_Fs_ = 1/0.72
            self.SUBJECTS = self.find_subj_list()

        if self.params['DATA_type']=='Gordon': 
            self.BOLD_Fs_ = 1/0.72
            self.SUBJECTS = self.find_subj_list()

        if self.params['DATA_type']=='sample': 
            self.BOLD_Fs_ = 1/0.5

        if self.params['DATA_type']=='simulated': 
            self.BOLD_Fs_ = 1/0.5

    
    @property
    def BOLD_Fs(self):
        return self.BOLD_Fs_

    def load(self, subj_id2load=None):

        BOLD = {}
        if self.params['DATA_type']=='ICA': 
            BOLD = self.load_ica(subj_id2load)

        if self.params['DATA_type']=='Gordon': 
            BOLD = self.load_gordon(subj_id2load)

        if self.params['DATA_type']=='sample': 
            BOLD = self.load_sample(subj_id2load)

        if self.params['DATA_type']=='simulated': 
            BOLD = self.load_simulated(subj_id2load)

        return BOLD
    
    def find_subj_list(self):
        '''
        if any of data_root_gordon or data_root_ica
        is None, it will be excluded.
        '''

        # ICA
        SUBJECTS_ica = None
        if not self.params['data_root_ica'] is None:
            ALL_RECORDS = os.listdir(self.params['data_root_ica'])
            ALL_RECORDS = [i for i in ALL_RECORDS if '.txt' in i]
            ALL_RECORDS.sort()
            SUBJECTS_ica = list()
            for s in ALL_RECORDS:
                num = s[:s.find('.')]
                SUBJECTS_ica.append(num)
            SUBJECTS_ica = list(set(SUBJECTS_ica))
            SUBJECTS_ica.sort()

        # GORDON
        SUBJECTS_gordon = None
        if not self.params['data_root_gordon'] is None:
            ALL_RECORDS = os.listdir(self.params['data_root_gordon'])
            ALL_RECORDS = [i for i in ALL_RECORDS if 'Rest' in i]
            ALL_RECORDS.sort()
            SUBJECTS_gordon = list()
            for s in ALL_RECORDS:
                num = s[:s.find('_')]
                SUBJECTS_gordon.append(num)
            SUBJECTS_gordon = list(set(SUBJECTS_gordon))
            SUBJECTS_gordon.sort()

        SUBJECTS = list()
        if SUBJECTS_gordon is None:
            SUBJECTS = SUBJECTS_ica
        if SUBJECTS_ica is None:
            SUBJECTS = SUBJECTS_gordon
        if (not SUBJECTS_gordon is None) and (not SUBJECTS_ica is None):
            SUBJECTS = intersection(SUBJECTS_gordon, SUBJECTS_ica)

        print( str(len(SUBJECTS)) + ' subjects were found. ')

        # print( str(len(SUBJECTS)) + ' subjects were found. ' + str(self.params['num_subj']) + ' subjects were selected.')

        # SUBJECTS = SUBJECTS[0:self.params['num_subj']]

        return SUBJECTS

    def load_gordon(self, subj_id2load=None):

        SESSIONs = self.params['SESSIONs'] #['Rest1_LR' , 'Rest1_RL', 'Rest2_LR', 'Rest2_RL']
        if subj_id2load is None:
            SUBJECTS = self.SUBJECTS
        else:
            SUBJECTS = [subj_id2load]

        # LOAD Region Location DATA

        locs = sio.loadmat(self.params['data_root_gordon']+'Gordon333_LOCS.mat')
        locs = locs['locs']

        # LOAD Region Data

        file = self.params['data_root_gordon']+'Gordon333_Key.txt'
        f = open(file, 'r')

        atlas_data = []
        for line in f:
            row = line.split()
            atlas_data.append(row)

        # apply networks2include
        nodes2include = [i-1 for i, x in enumerate(atlas_data) if x[3] in self.params['networks2include']]
        locs = locs[nodes2include, :]
        atlas_data = [x for node, x in enumerate(atlas_data) if node-1 in nodes2include]

        BOLD = {}
        for session in SESSIONs:
            BOLD[session] = None
            for subject in SUBJECTS:

                subj_fldr = subject + '_' + session

                # LOAD BOLD Data

                DATA = hdf5storage.loadmat(self.params['data_root_gordon']+subj_fldr+'/ROI_data_Gordon_333_surf.mat')
                time_series = DATA['ROI_data']

                # change time_series.shape to (nodes, time)
                time_series = time_series.T

                # apply networks2include
                time_series = time_series[nodes2include, :]

                if BOLD[session] is None:
                    BOLD[session] = TIME_SERIES(data=time_series, subj_id=subject, \
                                        Fs=self.BOLD_Fs, \
                                        locs=locs, nodes_info=atlas_data, \
                                        TS_name='BOLD Real', session_name=session \
                                    )
                else:
                    BOLD[session].append_ts(new_time_series=time_series, subj_id=subject)

            print( '*** Session ' + session + ': ' )
            print( 'number of regions= '+str(BOLD[session].n_regions) + ', number of TRs= ' + str(BOLD[session].n_time) )

        return BOLD

    def load_ica(self, subj_id2load=None):

        SESSIONs = self.params['SESSIONs'] #['session_1']
        if subj_id2load is None:
            SUBJECTS = self.SUBJECTS
        else:
            SUBJECTS = [subj_id2load]

        BOLD = {}
        for session in SESSIONs:
            BOLD[session] = None
            for subject in SUBJECTS:
                time_series = np.loadtxt( \
                    self.params['data_root_ica'] + subject + '.txt', dtype='float64' \
                    )
                time_series = time_series.T
                
                # time_series = time_series - np.repeat(np.mean(time_series, axis=1)[:,None], time_series.shape[1], axis=1) # ???????????????????????

                if BOLD[session] is None:
                    BOLD[session] = TIME_SERIES(data=time_series, subj_id=subject, Fs=self.BOLD_Fs, TS_name='BOLD ICA', session_name=session)
                else:
                    BOLD[session].append_ts(new_time_series=time_series, subj_id=subject)

            print(BOLD[session].n_regions, BOLD[session].n_time)

        return BOLD

    def load_sample(self, subj_id2load=None):

        ###### BOLD DATA ######
        time_BOLD = np.load(self.params['data_root_sample']+'bold_time.npy')/1e3    
        time_series = np.load(self.params['data_root_sample']+'bold_data.npy')

        time_series = time_series.T

        BOLD = None
        for subject in range(5):
            if BOLD is None:
                BOLD = TIME_SERIES( \
                    data=time_series[:, (subject)*1200:(subject+1)*1200], \
                    subj_id=str(subject+1), Fs=self.BOLD_Fs, \
                    time_array=time_BOLD[(subject)*1200:(subject+1)*1200], \
                    TS_name='BOLD Sample' \
                )
            else:
                BOLD.append_ts( \
                    new_time_series=time_series[:, (subject)*1200:(subject+1)*1200], \
                    time_array=time_BOLD[(subject)*1200:(subject+1)*1200],
                    subj_id=str(subject+1) \
                )

        print(BOLD.n_regions, BOLD.n_time)

        return BOLD

    def load_simulated(self, subj_id2load=None):

        ################################# Load Simulated BOLD data #################################

        time_BOLD = np.load(self.params['data_root_simul']+'bold_time.npy')/1e3    
        time_series_BOLD = np.load(self.params['data_root_simul']+'bold_data.npy')

        BOLD = TIME_SERIES(data=time_series_BOLD.T, subj_id='1', Fs=self.BOLD_Fs, time_array=time_BOLD, TS_name='BOLD Simulation')

        # ################################# Load Simulated Tavg data #################################

        # time_Tavg = np.load(self.params['data_root_simul']+'TVB data/tavg_time.npy')/1e3    
        # time_series_Tavg = np.load(self.params['data_root_simul']+'TVB data/tavg_data.npy')

        # TAVG = TIME_SERIES(data=time_series_Tavg.T, subj_id='1', Fs=200, time_array=time_Tavg, TS_name='Tavg Simulation')

        return BOLD
