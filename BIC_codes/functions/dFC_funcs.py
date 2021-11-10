#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:34:49 2021

@author: mte
"""

import numpy as np
from scipy import signal
from copy import deepcopy
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
# import warnings

# warnings.simplefilter('ignore')

################################# Parameters ####################################

fig_dpi = 120

################################# Other Functions ####################################

# test
def get_subj_ts_dict(time_series_dict, subj_id):
    subj_ts_dict = {}
    for session in time_series_dict:
        subj_ts_dict[session] = time_series_dict[session].get_subj_ts(subj_id=subj_id)
    return subj_ts_dict

# test
def common_subj_lst(time_series_dict):
    SUBJECTs = None
    for session in time_series_dict:
        if SUBJECTs is None:
            SUBJECTs = list(set(time_series_dict[session].subj_id_array))
        else:
            SUBJECTs = intersection(SUBJECTs, list(set(time_series_dict[session].subj_id_array)))
    return SUBJECTs

def intersection(lst1, lst2): # input is a list 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def TR_intersection(dFCM_lst): # input is a list of dFCM objs
    TRs_lst_old = dFCM_lst[0].TR_array
    for dFCM in dFCM_lst:
        TRs_lst_new = intersection(TRs_lst_old, dFCM.TR_array)
        TRs_lst_old = TRs_lst_new
    TRs_lst_old.sort()
    if len(TRs_lst_old)==0:
        print('No TR intersection.')
    return TRs_lst_old

def dFC_dict_slice(data, idx_lst):
    data_sliced = {}
    for i, k in enumerate(data):
        if i in idx_lst:
            data_sliced[k] = data[k]
    return data_sliced

def visualize_state_TC(TC_lst, \
    TRs, \
    state_lst, \
    TC_name_lst, \
    title='', \
    save_image=None, output_root=None\
    ):

    color_lst = ['k', 'b', 'g', 'r']

    if 'on' in state_lst and 'off' in state_lst:
        ticks = range(2)
    else:
        ticks = range(1, len(state_lst)+1)

    plt.figure(figsize=(25, 5))
    for i, TC in enumerate(TC_lst):
        plt.plot(TRs, TC, color_lst[i], linewidth=2)
    plt.xlabel('TR')
    plt.yticks(ticks=ticks, labels=state_lst)
    plt.legend(TC_name_lst)
    plt.title(title)
    if save_image:
        plt.savefig(output_root + '.png', dpi=fig_dpi)  
        plt.close()
    else:
        plt.show()

    return

def visualize_conn_mat(data, title='', \
    name_lst_key=None, mat_key=None, \
    cmap='viridis',\
    disp_diag=True,\
    save_image=False, output_root=None, \
        fix_lim=True \
    ):

    '''
    - name_lst_key can be a list of names or the key to list of names
    - data must be a dict of correlation/connectivity matrices
    sample:
    Suptitle1
        corr_mat
            0.00 0.31 0.76 
            0.31 0.00 0.43 
            0.76 0.43 0.00 
        measure_lst
            ContinuousHMM
            Windowless
            Clustering_pear_corr
    Suptitle1
        corr_mat
            0.00 0.32 0.76 
            0.32 0.00 0.45 
            0.76 0.45 0.00 
        measure_lst
            ContinuousHMM
            Windowless
            Clustering_pear_corr
    '''

    if name_lst_key is None:
        fig_width = 25*(len(data)/10)
    else:
        fig_width = 35*(len(data)/10) + 4
    fig_height = 10

    fig, axs = plt.subplots(1, len(data), figsize=(fig_width, fig_height), \
        facecolor='w', edgecolor='k')

    if not type(axs) is np.ndarray:
        axs = np.array([axs])

    fig.suptitle(title) #, fontsize=20, size=20

    axs = axs.ravel()

    for i, key in enumerate(data):

        name_lst = None
        if not name_lst_key is None:
            if type(name_lst_key) is str:
                name_lst = data[key][name_lst_key]
            if type(name_lst_key) is list:
                name_lst = name_lst_key

        if mat_key is None:
            C = data[key]
        else:
            C = data[key][mat_key]

        C = np.abs(C) # ?????? should we do this?

        if not disp_diag:
            C = np.multiply(C, 1-np.eye(len(C)))
            C = C + np.mean(C.flatten()) * np.eye(len(C))

        if np.any(C<0): # ?????? should we do this?
            V_MIN = -1
            V_MAX = 1
        else: # ?????? should we do this?
            V_MIN = 0
            V_MAX = 1

        if not fix_lim:
            V_MAX = np.max(C)
            V_MIN = np.min(C)

        if name_lst is None:
            axs[i].set_axis_off()

        im = axs[i].imshow(C, interpolation='nearest', aspect='equal', cmap=cmap,    # 'viridis' or 'jet'
            vmin=V_MIN, vmax=V_MAX)
        
        if not name_lst is None:
            axs[i].set_xticks(np.arange(len(name_lst)))
            axs[i].set_yticks(np.arange(len(name_lst)))
            axs[i].set_xticklabels(name_lst, rotation=90, fontsize=9)
            axs[i].set_yticklabels(name_lst, fontsize=9)
        axs[i].set_title(key)

    fig.subplots_adjust(
        bottom=0.1, \
        top=1.5, \
        left=0.1, \
        right=0.9,
        # wspace=0.02, \
        # hspace=0.02\
    )

    if not name_lst is None:
        fig.subplots_adjust(
            wspace=0.85 
        )
        
    if name_lst is None:
        cb_ax = fig.add_axes([0.91, 0.75, 0.007, 0.1])
    else:
        cb_ax = fig.add_axes([0.91, 0.75, 0.02, 0.1])
    cbar = fig.colorbar(im, cax=cb_ax, shrink=0.8) # shrink=0.8??

    # # set the colorbar ticks and tick labels
    # cbar.set_ticks(np.arange(0, 1.1, 0.5))
    # cbar.set_ticklabels(['0', '0.5', '1'])

    if save_image:
        plt.savefig(output_root + '.png', dpi=fig_dpi)  
        plt.close()
    else:
        plt.show()

def dFC_dict_normalize(D, global_normalization=False, threshold=0.0):

    C = list()
    for key in D:
        C.append(D[key])
    C = np.array(C)

    C_z = dFC_mat_normalize(C, \
        global_normalization=global_normalization, \
        threshold=threshold \
    )

    D_z = {}
    for i, key in enumerate(D):
        D_z[key] = C_z[i,:,:]

    return D_z

def dFC_mat_normalize(C_t, global_normalization=False, threshold=0.0):

    # threshold is ratio of connections wanted to be zero
    C_t_z = deepcopy(C_t)
    if len(C_t_z.shape)<3:
        C_t_z = np.expand_dims(C_t_z, axis=0)

    if global_normalization:

        # transform the whole abs(dFC mat) to [0, 1] 

        signs = np.sign(C_t_z)
        C_t_z = np.abs(C_t_z)

        miN = list()
        for i in range(C_t_z.shape[0]):
            slice = C_t_z[i,:,:]
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0],dtype=bool))]
            miN.append(np.min(slice_non_diag))

        C_t_z = C_t_z - np.min(miN)

        maX = list()
        for i in range(C_t_z.shape[0]):
            slice = C_t_z[i,:,:]
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0],dtype=bool))]
            maX.append(np.max(slice_non_diag))

        if np.max(maX) != 0:
            C_t_z = np.divide(C_t_z, np.max(maX))

        # thresholding
        d = deepcopy(np.ravel(C_t_z))
        d.sort()
        new_threshold = d[int(threshold*len(d))]
        C_t_z = np.multiply(C_t_z, (C_t_z>=new_threshold))
        C_t_z = np.multiply(C_t_z, signs)

    else:

        # transform abs of each time slice to [0, 1]

        signs = np.sign(C_t_z)
        C_t_z = np.abs(C_t_z)
        
        for i in range(C_t_z.shape[0]):
            slice = C_t_z[i,:,:]
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0],dtype=bool))]
            slice = slice - np.min(slice_non_diag)
            slice_non_diag = slice[np.where(~np.eye(slice.shape[0],dtype=bool))]
            if np.max(slice_non_diag) != 0:
                slice = np.divide(slice, np.max(slice_non_diag))

            # thresholding
            d = deepcopy(np.ravel(slice))
            d.sort()
            new_threshold = d[int(threshold*len(d))]
            slice = np.multiply(slice, (slice>=new_threshold))

            C_t_z[i,:,:] = slice

        C_t_z = np.multiply(C_t_z, signs)

    # removing self connections
    for i in range(C_t_z.shape[1]):
        C_t_z[:, i, i] = np.mean(C_t_z) # ?????????????????

    return C_t_z

def print_mat(mat, s=0):
    if len(mat.shape)==1:
        mat = np.expand_dims(mat, axis=0)
    for i in mat:
        print('\t'*s,  end=" ")
        for j in i:
            print("{:.2f}".format(j), end=" ")
        print()

def print_dict(t, s=0):
    if not isinstance(t,dict) and not isinstance(t,list):
        if isinstance(t,np.ndarray):
            print_mat(t, s)
        else:
            if isinstance(t,float):
                print('\t'*s+"{:.2f}".format(t))
            else:
                print('\t'*s+str(t))
    else:
        for key in t:
            print('\t'*s+str(key))
            if not isinstance(t,list):
                print_dict(t[key],s+1)
############################# dFC Analyzer class ################################

"""

todo:
- 
"""

import time

class DFC_ANALYZER:
    # if self.n_jobs is None => no parallelization

    def __init__(self, MEASURES_lst, analysis_name='', **params):

        self.analysis_name = analysis_name
        # self.MEASURES_lst_ = MEASURES_lst
        self.MEASURES_lst_ = self.DD_MEASURES_lst(MEASURES_lst) + self.SB_MEASURES_lst(MEASURES_lst)
        self.MEASURES_fit_lst_ = {}
        
        self.params = params
        if not 'vis_TR_idx' in self.params:
            self.params['vis_TR_idx'] = None
        if not 'save_image' in self.params:
            self.params['save_image'] = False
        if not 'output_root' in self.params:
            self.params['output_root'] = None
        if not 'n_jobs' in self.params:
            self.params['n_jobs'] = -1
        if not 'verbose' in self.params:
            self.params['verbose'] = 1
        if not 'backend' in self.params:
            self.params['backend'] = 'loky'

        self.sim_assess_params = {}
        if 'sim_assess_params' in params:
            self.sim_assess_params = params['sim_assess_params']

        # self.dyn_conn_det_params = {}
        # if 'dyn_conn_det_params' in params:
            # self.dyn_conn_det_params = params['dyn_conn_det_params']

        self.methods_corr_dict_lst_ = list()

    @property
    def methods_corr(self):
        # it assumes all subjects have all sessions and the order of their measures is the same ?
        
        methods_corr_dict = {}
        for session in self.methods_corr_dict_lst[0]:
            methods_corr_dict[session] = {}
            methods_corr = list()
            
            # dFC corr
            for methods_corr_subj_dict in self.methods_corr_dict_lst:
                methods_corr.append(methods_corr_subj_dict[session]['corr_mat'])
            
            methods_corr = np.array(methods_corr)
            methods_corr_dict[session]['corr_mat'] = np.mean(methods_corr, axis=0)
            methods_corr_dict[session]['measure_lst'] = methods_corr_subj_dict[session]['measure_lst']

        return methods_corr_dict

    @property
    def methods_corr_dict_lst(self):
        return self.methods_corr_dict_lst_

    @property
    def MEASURES_lst(self):
        return self.MEASURES_lst_

    # test
    @property
    def MEASURES_dict(self):
        dict = {}
        for measure in self.MEASURES_lst:
            assert not measure.measure_name in dict, \
                'duplicate measure name.'
            dict[measure.measure_name] = measure
        return dict

    @property
    def MEASURES_fit_lst(self):
        return self.MEASURES_fit_lst_

    # test
    @property
    def MEASURES_fit_dict(self):
        dict = {}
        for session in self.MEASURES_fit_lst:
            dict[session] = {}
            for measure in self.MEASURES_fit_lst[session]:
                assert not measure.measure_name in dict[session], \
                    'duplicate measure name.'
                dict[session][measure.measure_name] = measure
        return dict

    @property
    def FCS_dict(self):
        #FCS_dict[session][measure.measure_name]['state_TC'][FCS_i]['FCS']
        FCS_dict = {}
        for session in self.MEASURES_fit_lst:
            FCS_dict[session] = {}
            for measure in self.SB_MEASURES_lst(self.MEASURES_fit_lst[session]):
                FCS_dict[session][measure.measure_name] = measure.FCS_dict
        return FCS_dict

    @property
    def FCS_sim_dict(self):
        ## normalize ??
        normalize = False
        # FCS_sim_dict[session][measure_i][measure_j][FCS_i]
        FCS_sim_dict = {}
        for session in self.FCS_dict:
            FCS_sim_dict[session] = {}
            for measure_i in self.FCS_dict[session]:
                FCS_sim_dict[session][measure_i] = {}
                for measure_j in self.FCS_dict[session]:

                    FCS_sim_dict[session][measure_i][measure_j] = self.FCS_sim_calc( \
                        FCS_dict_i = self.FCS_dict[session][measure_i], \
                        FCS_dict_j = self.FCS_dict[session][measure_j], \
                        normalize=normalize \
                    )

        return FCS_sim_dict

    def FCS_sim_calc(self, FCS_dict_i, FCS_dict_j, normalize=False):
        # FCS_sim_dict[FCS_i] = FCS_similarity_vec

        FCS_sim_dict = {}
        for FCS_i in FCS_dict_i['state_TC']:
            FCS_similarity_vec = np.zeros((len(FCS_dict_j['state_TC'])))
            for j, FCS_j in enumerate(FCS_dict_j['state_TC']):
                if normalize:
                    FCS_similarity_vec[j] = self.FC_mat_corr( \
                        dFC_mat_normalize(C_t=FCS_dict_i['state_TC'][FCS_i]['FCS'], \
                            global_normalization=False), \
                        dFC_mat_normalize(C_t=FCS_dict_j['state_TC'][FCS_j]['FCS'], \
                            global_normalization=False) \
                    ) 
                else:
                    FCS_similarity_vec[j] = self.FC_mat_corr( \
                        FCS_dict_i['state_TC'][FCS_i]['FCS'], \
                        FCS_dict_j['state_TC'][FCS_j]['FCS']\
                    ) 
            FCS_sim_dict[FCS_i] = FCS_similarity_vec

        return FCS_sim_dict

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

    def FC_mat_corr(self, A, B):
        # it excludes diagonal values
        A = np.multiply(A, 1-np.eye(len(A)))
        B = np.multiply(B, 1-np.eye(len(B)))
        return np.corrcoef(A.flatten(), B.flatten())[0,1]

    def similarity(self, D_A, D_B, normalize=False, return_matched=False):
        # searchs D_A FC_mats in D_B
        # this functions is recommended to be used only for inter session similarity 
        # assessment since it does not take into account the temporal transition
        similarity_dict = {}
        for key_a in D_A:
            similarity_vec = np.zeros((len(D_B)))
            b_keys = [key for key in D_B]
            for b, key_b in enumerate(D_B):
                if normalize:
                    similarity_vec[b] = self.FC_mat_corr( \
                        dFC_mat_normalize(C_t=D_B[key_b], \
                            global_normalization=False), \
                        dFC_mat_normalize(C_t=D_A[key_a], \
                            global_normalization=False) \
                    ) 
                else:
                    similarity_vec[b] = self.FC_mat_corr( \
                        D_B[key_b], \
                        D_A[key_a]\
                    ) 
            similarity_dict[key_a] = {}
            similarity_dict[key_a]['match'] = b_keys[np.argmax(similarity_vec)]
            similarity_dict[key_a]['score'] = np.max(similarity_vec)
        similarity_score = np.mean([similarity_dict[key_a]['score'] for key_a in similarity_dict])

        if return_matched:
            return similarity_score, similarity_dict
        else:
            return similarity_score

    def FCS_session_similarity(self):
        # measures inter session similarity
        
        FCS_session_sim_dict = {}
        SESSIONs = [session for session in self.MEASURES_fit_lst]
        for m in range(len(self.MEASURES_fit_lst[SESSIONs[0]])):
            # if the measure is DD -> continue to the next loop
            if not self.MEASURES_fit_lst[SESSIONs[0]][m].is_state_based:
                continue
            # this measure is a name/string
            measure=self.MEASURES_fit_lst[SESSIONs[0]][m].measure_name
            FCS_session_sim_dict[measure] = {}
            FCS_session_sim_dict[measure]['session_lst'] = SESSIONs
            FCS_session_sim_dict[measure]['sim_mat'] = np.zeros((len(SESSIONs),len(SESSIONs)))
            for i, session_i in enumerate(SESSIONs):
                for j, session_j in enumerate(SESSIONs):

                    assert self.MEASURES_fit_lst[session_i][m].measure_name==self.MEASURES_fit_lst[session_j][m].measure_name, \
                        'measures mismatch!'
                    
                    C_A = self.MEASURES_fit_lst[session_i][m].FCS
                    C_B = self.MEASURES_fit_lst[session_j][m].FCS
                    D_A = {}
                    for k in range(C_A.shape[0]):
                        D_A['FCS'+str(i+1)] = C_A[k,:,:]
                    D_B = {}
                    for k in range(C_B.shape[0]):
                        D_B['FCS'+str(i+1)] = C_B[k,:,:]
                    # self.similarity(dFC_dict_normalize(D_A), dFC_dict_normalize(D_B), return_matched=False, normalize=True)
                    FCS_session_sim_dict[measure]['sim_mat'][i, j] = self.similarity(D_A, D_B, return_matched=False, normalize=False) # normalize ??
        return FCS_session_sim_dict

    def FCS_measure_similarity(self):
        
        # this function returns similarity of different measures' FCSs regardless of state transitions
        FCS_measure_sim_dict = {}
        for session in self.MEASURES_fit_lst:
            # we only keep SB measures:
            sb_measures_lst = self.SB_MEASURES_lst(self.MEASURES_fit_lst[session])
            FCS_measure_sim_dict[session] = {}
            FCS_measure_sim_dict[session]['measure_lst'] = [measure.measure_name for measure in sb_measures_lst]
            FCS_measure_sim_dict[session]['sim_mat'] = np.zeros((len(sb_measures_lst),len(sb_measures_lst)))
            for i, measure_i in enumerate(sb_measures_lst):
                for j, measure_j in enumerate(sb_measures_lst):
                    C_A = measure_i.FCS
                    C_B = measure_j.FCS
                    D_A = {}
                    for k in range(C_A.shape[0]):
                        D_A['FCS'+str(k+1)] = C_A[k,:,:]
                    D_B = {}
                    for k in range(C_B.shape[0]):
                        D_B['FCS'+str(k+1)] = C_B[k,:,:]
                    # self.similarity(dFC_dict_normalize(D_A), dFC_dict_normalize(D_B), return_matched=False, normalize=True)
                    FCS_measure_sim_dict[session]['sim_mat'][i, j] = self.similarity(D_A, D_B, return_matched=False, normalize=False) # normalize ??

        return FCS_measure_sim_dict
    
    def dFC_session_similarity(self, dFCM_dict):
        # measures inter session similarity

        dFC_session_sim_dict = {}
        SESSIONs = [session for session in dFCM_dict]
        for measure in dFCM_dict[SESSIONs[0]]:
            dFC_session_sim_dict[measure] = {}
            dFC_session_sim_dict[measure]['session_lst'] = SESSIONs
            dFC_session_sim_dict[measure]['sim_mat'] = np.zeros((len(SESSIONs),len(SESSIONs)))
            for i, session_i in enumerate(SESSIONs):
                for j, session_j in enumerate(SESSIONs):

                    # dFCM_dict[session_i][measure] is a dFCM object
                    # it only considers num_samples number of samples for measuring similarity (uniformly picked)

                    D_A = dFCM_dict[session_i][measure].dFC2dict(num_samples=self.sim_assess_params['num_samples'])
                    D_B = dFCM_dict[session_j][measure].dFC2dict(num_samples=self.sim_assess_params['num_samples'])

                    # self.similarity(dFC_dict_normalize(D_A), dFC_dict_normalize(D_B), return_matched=False, normalize=True)
                    dFC_session_sim_dict[measure]['sim_mat'][i, j] = self.similarity(D_A, D_B, return_matched=False, normalize=False) # normalize ??
        return dFC_session_sim_dict

    def dFC_measure_similarity(self, dFCM_dict):
        # this is not useful, instead we are using dFC_corr
        pass

    # todo: add save image
    def similarity_analyze(self, SUBJs_dFC_session_sim_dict, verb=False, show_all_subj=False):

        ############ inter session dFC similarity averaged ############
        
        # dFC_session_sim_dict contains similarity between different sessions 
        # and in different measures in each subject
        avg_dFC_session_sim_dict = {}
        SUBJECTs = [subject for subject in SUBJs_dFC_session_sim_dict]
        for measure in SUBJs_dFC_session_sim_dict[SUBJECTs[0]]:
            avg_dFC_session_sim_dict[measure] = {}
            avg_dFC_session_sim_dict[measure]['sim_mat'] = 0
            for subject in SUBJECTs:
                avg_dFC_session_sim_dict[measure]['sim_mat'] += np.divide(SUBJs_dFC_session_sim_dict[subject][measure]['sim_mat'], len(SUBJECTs))
                avg_dFC_session_sim_dict[measure]['session_lst'] = SUBJs_dFC_session_sim_dict[subject][measure]['session_lst']
            
        visualize_conn_mat(avg_dFC_session_sim_dict, \
            title='avg inter session dFC similarity', \
            name_lst_key='session_lst', mat_key='sim_mat', \
            save_image=self.params['save_image'], output_root=self.params['output_root']+'similarity/inter_session_dFC' \
        )

        avg_measure_repro = {}
        for measure in avg_dFC_session_sim_dict:
            avg_measure_repro[measure] = np.mean(avg_dFC_session_sim_dict[measure]['sim_mat'])

        if verb:
            print('average inter session dFC similarity:')
            print_dict(avg_measure_repro)

        ############ inter session dFC similarity for all subjects ############

        if show_all_subj:
            if verb:
                print('inter session dFC similarity dict:')
                print_dict(SUBJs_dFC_session_sim_dict)

            for subject in SUBJs_dFC_session_sim_dict:
                visualize_conn_mat(SUBJs_dFC_session_sim_dict[subject], \
                    title='inter session dFC similarity subj '+subject, \
                    name_lst_key='session_lst', mat_key='sim_mat', \
                    save_image=self.params['save_image'], output_root=self.params['output_root']+'similarity/inter_session_dFC'+subject \
                    )

        ############ inter session measures FCS similarity ############

        FCS_session_sim_dict = self.FCS_session_similarity()
        if verb:
            print_dict(FCS_session_sim_dict)

        visualize_conn_mat(FCS_session_sim_dict, \
            title='inter session measures FCS similarity', \
            name_lst_key='session_lst', mat_key='sim_mat', \
            save_image=self.params['save_image'], output_root=self.params['output_root']+'similarity/inter_session_FCS' \
        )

        avg_measure_repro = {}
        for measure in FCS_session_sim_dict:
            avg_measure_repro[measure] = np.mean(FCS_session_sim_dict[measure]['sim_mat'])

        if verb:
            print('average measures FCS reproducibility:')
            print_dict(avg_measure_repro)

        ############ intra session measures FCS similarity ############

        # FCS_measure_sim_dict = self.FCS_measure_similarity()
        # if verb:
        #     print_dict(FCS_measure_sim_dict)
        # visualize_conn_mat(FCS_measure_sim_dict, title='measures FCS similarity matrix', name_lst_key='measure_lst', mat_key='sim_mat')

        return

    def dFCM_state_match(self, trans_sim_dict, FCS_dict_i, FCS_dict_j, FCS_sim, matching_method='score'):

        # matching_method can be 'FCS', 'transition' or 'score' (combination of both)
        # trans_sim_dict[FCS_i]['trans_sim_vec'] = list(transition_similarity_vec) of subjs
        # returns 

        state_match_dict= {}
        state_match_dict['FCS_match'] = {}
        state_match_dict['orig_FCSs'] = {}
        state_match_dict['matched_FCSs'] = {}
        
        FCS_j_lst = [FCS for FCS in FCS_dict_j['state_TC']]
        for FCS_i in trans_sim_dict:

            # trans_sim_dict[FCS_i]['trans_sim_vec'] is list of trans_sim_vec of all subjects

            score = list()
            for trans_sim_vec in trans_sim_dict[FCS_i]['trans_sim_vec']:
                score.append(np.multiply(FCS_sim[FCS_i], trans_sim_vec))
            
            trans_sim = np.array(trans_sim_dict[FCS_i]['trans_sim_vec'])
            score = np.array(score)

            trans_sim_avg = np.nanmean(trans_sim, axis=0)
            score_avg = np.nanmean(score, axis=0)

            # choose match based on score/FCS/transition 
            if matching_method=='score':
                FCS_j_max = np.nanargmax(score_avg)
            if matching_method=='transition':
                FCS_j_max = np.nanargmax(trans_sim_avg)
            if matching_method=='FCS':
                FCS_j_max = np.argmax(FCS_sim[FCS_i])

            state_match_dict['FCS_match'][FCS_i] = {}
            state_match_dict['FCS_match'][FCS_i]['match'] = FCS_j_lst[FCS_j_max]
            state_match_dict['FCS_match'][FCS_i]['trans_corr'] = trans_sim_avg[FCS_j_max]
            state_match_dict['FCS_match'][FCS_i]['FCS_corr'] = FCS_sim[FCS_i][FCS_j_max]
            state_match_dict['FCS_match'][FCS_i]['score'] = score_avg[FCS_j_max]

            state_match_dict['orig_FCSs'][FCS_i] = FCS_dict_i['state_TC'][FCS_i]['FCS']
            state_match_dict['matched_FCSs'][FCS_i+'->'+FCS_j_lst[FCS_j_max]] = FCS_dict_j['state_TC'][FCS_j_lst[FCS_j_max]]['FCS']

        # avg over all FCSs of measure_i
        FCS_match_dict = state_match_dict['FCS_match']
        state_match_dict['avg_score'] = np.nanmean([FCS_match_dict[FCS_i]['score'] for FCS_i in FCS_match_dict])
        state_match_dict['avg_trans_corr'] = np.nanmean([FCS_match_dict[FCS_i]['trans_corr'] for FCS_i in FCS_match_dict])
        state_match_dict['avg_FCS_corr'] = np.mean([FCS_match_dict[FCS_i]['FCS_corr'] for FCS_i in FCS_match_dict])

        return state_match_dict
    
    def state_match(self):
        # this function matches states of two different method
        # matching_method can be 'FCS', 'transition' or 'score' (combination of both)

        state_match = {}
        for session in self.FCS_sim_dict:

            FCS_dict = self.FCS_dict[session]
            FCS_sim_dict = self.FCS_sim_dict[session]

            # collecting similarity scores across subjects
            score_dict = {}
            for subj_i, subj_dict in enumerate(self.methods_corr_dict_lst_):
                for measure_i in subj_dict[session]['state_match']:
                    if subj_i==0:
                        score_dict[measure_i] = {}
                    for measure_j in subj_dict[session]['state_match'][measure_i]:
                        if subj_i==0:
                            score_dict[measure_i][measure_j] = {}
                        for FCS_i in subj_dict[session]['state_match'][measure_i][measure_j]:
                            trans_sim_vec = subj_dict[session]['state_match'][measure_i][measure_j][FCS_i]['trans_sim_vec']
            
                            if subj_i==0:
                                score_dict[measure_i][measure_j][FCS_i] = {}
                                score_dict[measure_i][measure_j][FCS_i]['trans_sim_vec'] = list()

                            score_dict[measure_i][measure_j][FCS_i]['trans_sim_vec'].append(trans_sim_vec)

            # averaging collected similarity scores
            state_match_dict = {}
            state_match_dict['final'] = {}
            state_match_dict['method_pairs'] = {}
            state_match_dict['final']['score'] = {}
            state_match_dict['final']['trans_corr'] = {}
            state_match_dict['final']['FCS_corr'] = {}
            state_match_dict['final']['score']['corr_mat'] = np.zeros((len(score_dict), len(score_dict)))
            state_match_dict['final']['trans_corr']['corr_mat'] = np.zeros((len(score_dict), len(score_dict)))
            state_match_dict['final']['FCS_corr']['corr_mat'] = np.zeros((len(score_dict), len(score_dict)))
            for measure_i_iter, measure_i in enumerate(score_dict):
                state_match_dict['method_pairs'][measure_i] = {}
                for measure_j_iter, measure_j in enumerate(score_dict):

                    trans_sim_dict = score_dict[measure_i][measure_j]
                    state_match_dict['method_pairs'][measure_i][measure_j] = self.dFCM_state_match( \
                        trans_sim_dict=trans_sim_dict, \
                        FCS_dict_i=FCS_dict[measure_i], \
                        FCS_dict_j=FCS_dict[measure_j], \
                        FCS_sim=FCS_sim_dict[measure_i][measure_j], \
                        matching_method=self.sim_assess_params['matching_method'] \
                    )
                        
                    # avg over all FCSs of measure_i as a matrix
                    state_match_dict['final']['score']['corr_mat'][measure_i_iter, measure_j_iter] = state_match_dict['method_pairs'][measure_i][measure_j]['avg_score']
                    state_match_dict['final']['trans_corr']['corr_mat'][measure_i_iter, measure_j_iter] = state_match_dict['method_pairs'][measure_i][measure_j]['avg_trans_corr']
                    state_match_dict['final']['FCS_corr']['corr_mat'][measure_i_iter, measure_j_iter] = state_match_dict['method_pairs'][measure_i][measure_j]['avg_FCS_corr']

            state_match[session] = state_match_dict

        ###### results visualization ######

        for session in state_match:
            visualize_conn_mat(state_match[session]['final'], \
                title='intra session state match results ('+session+')', \
                name_lst_key=[measure for measure in state_match['Rest1_LR']['method_pairs']], \
                mat_key='corr_mat', \
                cmap='viridis',\
                save_image=self.params['save_image'], output_root=self.params['output_root']+'state_match/results_'+session, \
                fix_lim=True \
            )
        
        return state_match


    def state_transition_analyze(self, dFCM_i, dFCM_j, \
        state_match_dict=None, \
        matching_method='score', \
        subject='', \
        session='', \
        verb=False \
        ):

        normalize=False

        output_root = self.params['output_root']+'post_analysis/'+ \
            subject+'/state_match_'+ \
            dFCM_i.measure.measure_name+'_'+dFCM_j.measure.measure_name+ \
            '_'+session

        TRs = TR_intersection([dFCM_i, dFCM_j])
        TRs_lst = list()
        for TR in TRs:
            TRs_lst.append('TR'+str(TR))

        TC_name_lst = list()
        TC_name_lst.append(dFCM_i.measure.measure_name)
        TC_name_lst.append(dFCM_j.measure.measure_name)

        num_FCS = max(len(dFCM_i.FCSs), len(dFCM_j.FCSs))
        FCS_lst = ['FCS'+str(i+1) for i in range(num_FCS)]

        ############ state transition without matching ############

        state_TC_i = dFCM_i.state_TC(TRs=TRs_lst, \
                        state_match=False, state_match_dict=None \
                    )

        state_TC_j = dFCM_j.state_TC(TRs=TRs_lst, \
                        state_match=False, state_match_dict=None \
                    )

        visualize_state_TC([state_TC_i, state_TC_j], \
            TRs=TRs, TC_name_lst=TC_name_lst , \
            state_lst=FCS_lst, \
            title='state transition without matching', \
            save_image=self.params['save_image'], output_root=output_root+'without_match' \
        )

        ############ state matching ############
            
        # here the FCSs are matched using state_match function which can be 
        # by FCS similarity, transition similarity, or combination of both

        if state_match_dict is None:

            #trans_sim[FCS_i]['trans_sim_vec'] = transition_similarity_vec
            trans_sim = self.dFCM_trans_sim( \
                dFCM_i=dFCM_i, \
                dFCM_j=dFCM_j, \
                common_TRs=None \
            )

            trans_sim_dict = {}
            for FCS_i in trans_sim:
                trans_sim_dict[FCS_i] = {}
                trans_sim_vec = trans_sim[FCS_i]['trans_sim_vec']
                trans_sim_dict[FCS_i]['trans_sim_vec'] = [trans_sim_vec]

            FCS_dict_i = {}
            FCS_dict_i['state_TC'] = {}
            for FCS in dFCM_i.FCSs:
                FCS_dict_i['state_TC'][FCS] = {}
                FCS_dict_i['state_TC'][FCS]['FCS'] = dFCM_i.FCSs[FCS]

            FCS_dict_j = {}
            FCS_dict_j['state_TC'] = {}
            for FCS in dFCM_j.FCSs:
                FCS_dict_j['state_TC'][FCS] = {}
                FCS_dict_j['state_TC'][FCS]['FCS'] = dFCM_j.FCSs[FCS]

            FCS_sim = self.FCS_sim_calc( \
                FCS_dict_i=FCS_dict_i, \
                FCS_dict_j=FCS_dict_j, \
                normalize=False \
            )
            # FCS_sim_dict[FCS_i] = FCS_similarity_vec
            state_match_dict = self.dFCM_state_match( \
                trans_sim_dict=trans_sim_dict, \
                FCS_dict_i=FCS_dict_i, \
                FCS_dict_j=FCS_dict_j, \
                FCS_sim=FCS_sim, \
                matching_method=matching_method \
            )

        # visualization
        D_A = state_match_dict['orig_FCSs']
        D_B = state_match_dict['matched_FCSs']

        if normalize:
            visualize_conn_mat(dFC_dict_normalize(D_A), \
                disp_diag=False, cmap='viridis', \
                save_image=self.params['save_image'], output_root=output_root+'original_FCSs' \
            )
            visualize_conn_mat(dFC_dict_normalize(D_B), \
                disp_diag=False, cmap='viridis', \
                save_image=self.params['save_image'], output_root=output_root+'matched_FCSs' \
            )
        else:
            visualize_conn_mat(D_A, \
                disp_diag=False, cmap='viridis', \
                save_image=self.params['save_image'], output_root=output_root+'original_FCSs' \
            )
            visualize_conn_mat(D_B, \
                disp_diag=False, cmap='viridis', \
                save_image=self.params['save_image'], output_root=output_root+'matched_FCSs' \
            )

        print('state TC corr', [state_match_dict['FCS_match'][FCS_i]['trans_corr'] for FCS_i in  state_match_dict['FCS_match']])
        print('FCS corr', [state_match_dict['FCS_match'][FCS_i]['FCS_corr'] for FCS_i in  state_match_dict['FCS_match']])
        print('score', [state_match_dict['FCS_match'][FCS_i]['score'] for FCS_i in  state_match_dict['FCS_match']])

        ##### print matched scores #####

        print('average state TC corr: ' + str(state_match_dict['avg_trans_corr']))
        print('average FCS corr: ' + str(state_match_dict['avg_FCS_corr']))
        print('average score: ' + str(state_match_dict['avg_score']))

        ############ state transition after matching ############

        state_TC_i = dFCM_i.state_TC(TRs=TRs_lst, \
                        state_match=True, state_match_dict=state_match_dict \
                    )

        state_TC_j = dFCM_j.state_TC(TRs=TRs_lst, \
                        state_match=False, state_match_dict=None \
                    )

        visualize_state_TC([state_TC_i, state_TC_j], \
            TRs=TRs, TC_name_lst=TC_name_lst, \
            state_lst=FCS_lst, \
            title='state transition after matching', \
            save_image=self.params['save_image'], output_root=output_root+'after_match' \
        )

        if verb:
            print("state Time Course Equality: {:.2f}".format(np.sum(state_TC_i == state_TC_j)/len(state_TC_i)))

        ##### visualize matched states co transitions #####

        for key_a in state_match_dict['FCS_match']:

            key_b = state_match_dict['FCS_match'][key_a]['match']

            state_act_dict_i = dFCM_i.state_act_dict(TRs=TRs)
            state_act_dict_j = dFCM_j.state_act_dict(TRs=TRs)

            state_TC_i = state_act_dict_i['state_TC'][key_a]['act_TC']
            state_TC_j = state_act_dict_j['state_TC'][key_b]['act_TC']

            visualize_state_TC([state_TC_i, state_TC_j], \
                TRs=TRs, TC_name_lst=TC_name_lst, \
                state_lst=['off', 'on'], \
                title='state transition of ' + key_a + ' and ' + key_b, \
                save_image=self.params['save_image'], output_root=output_root+'trans_'+key_a+'_and_'+key_b \
            )

            if verb:
                print("state Time Course Equality: {:.2f}".format(state_match_dict['FCS_match'][key_a]['trans_corr']))

        return state_match_dict

    def time_analyze(self, time_series=None):

        print('Total Time = estimate_FCS Time + estimate_dFCM Time')
        
        time_lst = list()
        estimate_FCS_time_lst = list()
        for measure in self.MEASURES_lst:

            tic = time.time()
            if measure.is_state_based:
                tic_FCS = time.time()
                measure.estimate_FCS(time_series=time_series)
                estimate_FCS_time_lst.append(time.time() - tic_FCS)
            else:
                estimate_FCS_time_lst.append(0.0)

            measure.estimate_dFCM(time_series=time_series)

            time_lst.append(time.time() - tic)

        for i, measure in enumerate(self.MEASURES_lst):
            if measure.is_state_based:
                print('Measure '+measure.measure_name+' time = ' + \
                    str(time_lst[i]) + ' = ' + \
                    str(estimate_FCS_time_lst[i]) + ' + ' + \
                    str(time_lst[i] - estimate_FCS_time_lst[i]) \
                    )
            else:
                print('Measure '+measure.measure_name+' time = ' + \
                    str(time_lst[i]) \
                    )

    def post_analyze(self):

        # # todo add session
        # self.visualize_dFCMs(dFCM_lst=dFCM_lst, \
        #     TR_idx=self.params['vis_TR_idx'], \
        #     subj_id=time_series.subj_id_array[0], \
        #     )

        pass


    def analyze(self, time_series_dict):

        #time_series_dict is a dict of time_series

        ### estimate FCS ###

        print("FCS estimation started...")
        self.estimate_all_FCS(time_series_dict=time_series_dict)
        print("FCS estimation done.")

        ### Visualize FCS ###

        self.visualize_FCS(normalize=True, \
                                threshold=0.0, \
                                )
        
        ### estimate dFCM ###

        print("dFCM estimation started...")
        SUBJs_dFC_session_sim_dict = self.estimate_all_dFCM(time_series_dict=time_series_dict)
        print("dFCM estimation done.")

        #### Methods dFC Corr MAT ###

        fig_name = None
        if self.params['save_image']:
            output_root = self.params['output_root']+'dFC/'
            fig_name = output_root + 'avg_dFC_corr' 

        visualize_conn_mat(self.methods_corr, \
            title='intra session dFC correlation', \
            name_lst_key='measure_lst', mat_key='corr_mat', \
            cmap='viridis',\
            save_image=self.params['save_image'], output_root=fig_name, \
                fix_lim=True \
        )

        #### State Transitions in Measured dFC ###

        # ?? 

        ### DYNAMIC CONN DETEC ###

        # todo not corrected for time_series_dict
        
        # if self.dyn_conn_det_params['run_analysis']:

        #     dyn_conn_detector = DYN_CONN_DETECTOR(**self.dyn_conn_det_params)

        #     print("Dynamic Connection Detection started...")
        #     dyn_conn_detector.train_VAR(time_series=time_series, p=self.dyn_conn_det_params['p'])
        #     SUBJs_TH_mask = dyn_conn_detector.calc_subj_TH_mask(time_series, self.MEASURES_lst, \
        #         N=self.dyn_conn_det_params['N'], L=self.dyn_conn_det_params['L'])

        #     SUBJs_dyn_conn = dyn_conn_detector.mask_SUBJs_dFC(SUBJs_dFC_var, SUBJs_TH_mask)
        #     print("Dynamic Connection Detection done.")

        #     self.visualize_dyn_conns(SUBJs_dyn_conn)

        #     return SUBJs_dyn_conn

        return SUBJs_dFC_session_sim_dict

    def dFCM_var(self, MEASURES_dFCM):

        MEASURES_dFC_var = {}
        for measure in MEASURES_dFCM:
            dFC_mat = MEASURES_dFCM[measure].get_dFC_mat(TRs = MEASURES_dFCM[measure].TR_array)
            V = np.var(dFC_mat, axis=0)
            MEASURES_dFC_var[measure] = V
        return MEASURES_dFC_var

    def estimate_all_FCS(self, time_series_dict):

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

    def estimate_all_dFCM(self, time_series_dict):

        # time_series_dict is a dict of time_series
        
        SUBJECTs = common_subj_lst(time_series_dict) 

        if self.params['n_jobs'] is None:
            OUT = list()
            for subject in SUBJECTs:
                OUT.append( \
                    self.subj_lvl_analysis( \
                    time_series_dict=get_subj_ts_dict(time_series_dict, subj_id=subject), \
                    ))
        else:
            OUT = Parallel( \
                        n_jobs=self.params['n_jobs'], \
                        verbose=self.params['verbose'], \
                        backend=self.params['backend'])( \
                    delayed(self.subj_lvl_analysis)( \
                        time_series_dict=get_subj_ts_dict(time_series_dict, subj_id=subject), \
                        ) \
                        for subject in SUBJECTs)
            
        # out[0] are dFC_corr_assess_dict of different SUBJECTs
        # out[1] are dFC_session_sim_dict of different SUBJECTs
        SUBJs_dFC_session_sim_dict = {}
        for s, out in enumerate(OUT):
            dFC_session_sim_dict = out[1]
            # dFC_session_sim_dict contains similarity between different sessions and in different measures in each subject
            SUBJs_dFC_session_sim_dict[SUBJECTs[s]] = dFC_session_sim_dict

        # SUBJs_dFC_var = {}
        # todo add session
        # for s, out in enumerate(OUT):
        #     MEASURES_dFC_var = out[0]
        #     # MEASURES_dFC_var contains dFC_var of different measures of a subject
        #     SUBJs_dFC_var[SUBJECTs[s]] = MEASURES_dFC_var
                        
        self.methods_corr_dict_lst_ = [out[0] for out in OUT]

        return SUBJs_dFC_session_sim_dict

    def subj_lvl_analysis(self, time_series_dict):

        # time_series_dict is a dict of time_series

        dFCM_dict = {}
        dFC_corr_assess_dict = {}
        for session in time_series_dict:
            time_series = time_series_dict[session]
            # dFCM_lst = self.estimate_dFCM(time_series=time_series)
            if self.params['n_jobs'] is None:
                dFCM_lst = list()
                for measure in self.MEASURES_fit_lst_[session]:
                    dFCM_lst.append( \
                        measure.estimate_dFCM(time_series=time_series) \
                    )
            else:
                dFCM_lst = Parallel( \
                    n_jobs=self.params['n_jobs'], verbose=self.params['verbose'], backend=self.params['backend'])( \
                    delayed(measure.estimate_dFCM)(time_series=time_series) \
                        for measure in self.MEASURES_fit_lst_[session])

            MEASURES_dFCM = {}
            for dFCM in dFCM_lst:
                # test if self.MEASURES_lst[m].measure_name=dFCM.measure.measure_name
                MEASURES_dFCM[dFCM.measure.measure_name] = dFCM

            dFCM_dict[session] = MEASURES_dFCM

            # self.dFC_corr_assess returns a dict with 'corr_mat', 
            # 'measure_lst', 'sb_measure_lst', and 'state_match' keys
            dFC_corr_assess_dict[session] = self.dFC_corr_assess(dFCM_lst=dFCM_lst)

        dFC_session_sim_dict = self.dFC_session_similarity(dFCM_dict)

        # return MEASURES_dFC_var, dFC_corr_assess_dict
        return dFC_corr_assess_dict, dFC_session_sim_dict

    def dFC_corr(self, dFCM_i, dFCM_j, TRs=None):

        # returns correlation of dFC measures over time

        if TRs is None:
            TRs = TR_intersection([dFCM_i, dFCM_j])
        dFC_mat_i = dFCM_i.get_dFC_mat(TRs=TRs)
        dFC_mat_j = dFCM_j.get_dFC_mat(TRs=TRs)
        corr = list()
        for t in range(len(TRs)):
            corr.append(np.corrcoef(dFC_mat_i[t,:,:].flatten(), dFC_mat_j[t,:,:].flatten())[0,1])
        corr= np.array(corr)
        return corr

    def dFCM_trans_sim(self, dFCM_i, dFCM_j, common_TRs=None):

        # returns, for each pair of FCS of the two dFCMs, the portion 
        # of time that they have been both active as a dict of 
        # trans_sim[FCS_i]['trans_sim_vec'] = transition_similarity_vec
        # it only considers TRs in common_TRs

        if common_TRs is None:
            common_TRs = TR_intersection([dFCM_i, dFCM_j])

        state_act_dict_i = dFCM_i.state_act_dict(TRs=common_TRs)
        state_act_dict_j = dFCM_j.state_act_dict(TRs=common_TRs)

        # state_act_dict['state_TC'][FCS_key]['act_TC']
        # state_act_dict['state_TC'][FCS_key]['FCS']
        # state_act_dict['TR_array']

        trans_sim_dict = {}
        for FCS_i in state_act_dict_i['state_TC']:
            trans_sim_dict[FCS_i] = {}
            # FCS_similarity_vec = np.zeros((len(state_act_dict_j['state_TC'])))
            # if coat==False, transition_similarity_vec won't effect the final multiplication 
            # by FCS_similarity_vec
            transition_similarity_vec = np.ones((len(state_act_dict_j['state_TC'])))
            for j, FCS_j in enumerate(state_act_dict_j['state_TC']):

                # transition_similarity_vec[j] = np.mean( \
                #     state_act_dict_i['state_TC'][FCS_i]['act_TC']== \
                #     state_act_dict_j['state_TC'][FCS_j]['act_TC'] \
                # )

                # how many times out of all the times i has been on, j has been on too
                if np.sum(state_act_dict_i['state_TC'][FCS_i]['act_TC'])==0:
                    transition_similarity_vec[j] = float("NaN")
                else:
                    transition_similarity_vec[j] = np.divide( \
                        np.sum(np.multiply( \
                            state_act_dict_i['state_TC'][FCS_i]['act_TC'], \
                            state_act_dict_j['state_TC'][FCS_j]['act_TC'] )), \
                        np.sum(state_act_dict_i['state_TC'][FCS_i]['act_TC']) \
                    )

            trans_sim_dict[FCS_i]['trans_sim_vec'] = transition_similarity_vec

        return trans_sim_dict
    
    def dFC_corr_assess(self, dFCM_lst):

        ########## dFCM corr ##########
        # returns averaged correlation of dFC measures in a dict format
        # with 'corr_mat' and 'measure_lst' keys

        a = 0.1 # portion of the dFCs to ignore from the beginning and the end

        measure_lst = list()
        common_TRs = dFCM_lst[0].TR_array
        for dFCM in dFCM_lst:
            measure_lst.append(dFCM.measure.measure_name)
            common_TRs = intersection(common_TRs, dFCM.TR_array)
        common_TRs.sort()
        assert len(common_TRs)!=0, \
            'No TR intersection.'

        methods_corr = {}
        corr_mat = np.zeros((len(measure_lst), len(measure_lst)))
        for i in range(len(measure_lst)):
            for j in range(i+1, len(measure_lst)):

                # assert dFCM_lst[i].measure.measure_name==self.MEASURES_lst[i].measure_name and \
                #     dFCM_lst[j].measure.measure_name==self.MEASURES_lst[j].measure_name, \
                #     'mismatch in MEASURES_lst order'

                corr_ij = self.dFC_corr( \
                    dFCM_lst[i], dFCM_lst[j], \
                    TRs=common_TRs \
                        )
                corr_mat[i,j] = np.mean(corr_ij[ \
                    int(len(corr_ij)*a) : int(len(corr_ij)*(1-a)) \
                        ])
                corr_mat[j,i] = corr_mat[i,j] 

        ########## state coactivation corr ##########

        sb_dFCM_dict = {}
        for dFCM in dFCM_lst:
            if dFCM.measure.is_state_based:
                sb_dFCM_dict[dFCM.measure.measure_name] = dFCM

        ## using the same common_TRs as dFC corr

        state_match = {}
        for sb_measure_i in sb_dFCM_dict:
            state_match[sb_measure_i] = {}
            for sb_measure_j in sb_dFCM_dict:

                state_match[sb_measure_i][sb_measure_j] = self.dFCM_trans_sim( \
                    dFCM_i=sb_dFCM_dict[sb_measure_i], \
                    dFCM_j=sb_dFCM_dict[sb_measure_j], \
                    common_TRs=common_TRs \
                )
                
        methods_corr['corr_mat'] = corr_mat
        methods_corr['measure_lst'] = measure_lst
        methods_corr['state_match'] = state_match

        return methods_corr

    def visualize_dyn_conns(self, SUBJs_dyn_conn):
        
        for subject in SUBJs_dyn_conn:

            visualize_conn_mat(data=SUBJs_dyn_conn[subject], \
                title='Subject '+subject+' Dynamic Connections', \
                save_image=self.params['save_image'], \
                output_root=self.params['output_root']+'DYN_CONN/'+'subject'+subject+'_dyn_conn', \
                fix_lim=True \
            )

    def visualize_dFCMs(self, dFCM_lst=None, TR_idx=None, normalize=True, threshold=0.0, \
                            fix_lim=True, subj_id=''):
        
        # TR_idx is not TR values, but their indices!

        TRs = TR_intersection(dFCM_lst)
        if not TR_idx is None:
            assert not np.any(np.array(TR_idx)>=len(TRs)), \
                'TR_idx out of range.'
            TRs = [TRs[i] for i in TR_idx]

        for dFCM in dFCM_lst:
            if self.params['save_image']:
                output_root = self.params['output_root']+'dFC/'
                dFCM.visualize_dFC(TRs=TRs, normalize=normalize, threshold=threshold, \
                    fix_lim=fix_lim, \
                    save_image=self.params['save_image'], \
                    fig_name= output_root+'subject'+subj_id+'_'+dFCM.measure.measure_name+'_dFC')
            else:
                dFCM.visualize_dFC(TRs=TRs, normalize=normalize, threshold=threshold, fix_lim=fix_lim)

    def visualize_FCS(self, normalize=True, threshold=0.0):

        for session in self.MEASURES_fit_lst:
            for measure in self.MEASURES_fit_lst[session]:  
                if self.params['save_image']:
                    output_root = self.params['output_root'] + 'FCS/'
                    measure.visualize_FCS(normalize=normalize, threshold=threshold, save_image=True, \
                        fig_name= output_root + measure.measure_name+'_FCS_'+session)
                    # measure.visualize_TPM(normalize=normalize)
                else:
                    measure.visualize_FCS(normalize=normalize, threshold=threshold) # normalize?
                    # measure.visualize_TPM(normalize=normalize)
                

############################# Dynamic Connection Detector class ################################

"""

todo:
- 
"""

from statsmodels.tsa.api import VAR
from scipy.stats import norm

class DYN_CONN_DETECTOR:

    a = 0.95

    def __init__(self, **params):
        self.VAR_model = None
        self.lag_order = None
        self.TH_mask = None

    # @property
    # def methods_corr(self):
    #     return np.mean(self.methods_corr_lst, axis=0)

    def train_VAR(self, time_series, p=None):
        self.VAR_model = VAR(time_series.data.T)

        if p is None:
            self.VAR_model = self.VAR_model.fit(maxlags=10, ic='aic')
        else:
            self.VAR_model = self.VAR_model.fit(p)

        self.lag_order = self.VAR_model.k_ar

    def subj_lvl_calc_TH_mask(self, time_series, MEASURES_lst, N, L):

        SURROGATE = self.gen_surrogate( \
            time_series=time_series, \
            N=N, L=L, verbose=self.params['verbose']  \
        )
        dFCM_var = self.calc_dFC_var(time_series=SURROGATE, MEASURES_lst=MEASURES_lst)
        TH_mask = self.calc_TH_mask(dFCM_var, a=self.a)
        return TH_mask

    def calc_subj_TH_mask(self, time_series, MEASURES_lst, \
        N, L=None):

        SUBJECTs = list(set(time_series.subj_id_array))
        
        if self.params['n_jobs'] is None:
            SUBJs_TH_mask_lst = list()
            for subject in SUBJECTs:
                SURROGATE = self.gen_surrogate( \
                    time_series=time_series.get_subj_ts(subj_id=subject), \
                    N=N, L=L, verbose=self.params['verbose']  \
                )
                dFCM_var = self.calc_dFC_var(time_series=SURROGATE, MEASURES_lst=MEASURES_lst)
                TH_mask = self.calc_TH_mask(dFCM_var, a=self.a)
                SUBJs_TH_mask_lst.append(TH_mask)
        else:
            SUBJs_TH_mask_lst = Parallel( \
                    n_jobs=self.params['n_jobs'], \
                    verbose=self.params['verbose'] , \
                    backend=self.params['backend'])( \
                delayed(self.subj_lvl_calc_TH_mask)( \
                    time_series=time_series.get_subj_ts(subj_id=subject), \
                    MEASURES_lst=MEASURES_lst, \
                    N=N, L=L \
                    ) \
                    for subject in SUBJECTs)

        SUBJs_TH_mask = {}
        for i, TH_mask in enumerate(SUBJs_TH_mask_lst):
            SUBJs_TH_mask[SUBJECTs[i]] = TH_mask

        return SUBJs_TH_mask

    def gen_surrogate(self, time_series, N, L=None, verbose=0):
        if L is None:
            L = time_series.n_time

        SURROGATE = None
        for n in range(N):

            t0 = np.random.choice(\
                range(time_series.n_time-self.lag_order), \
                size=1, replace=False)[0]

            simul = self.VAR_model.forecast(time_series.data.T[t0:t0+self.lag_order, :], L)

            if SURROGATE is None:
                SURROGATE = TIME_SERIES(data=simul.T, subj_id='surrogate'+str(n+1), Fs=1/0.72, TS_name='BOLD Surrogate')
            else:
                SURROGATE.append_ts(new_time_series=simul.T, subj_id='surrogate'+str(n+1))

        if verbose==1:
            print(SURROGATE.n_regions, SURROGATE.n_time)

        return SURROGATE

    def calc_dFC_var(self, time_series, MEASURES_lst):

        # OUTPUT shape = [sample, measure, node, node]

        dFC_analyzer = DFC_ANALYZER(MEASURES_lst=MEASURES_lst, \
            n_jobs=self.params['n_jobs'], verbose=self.params['verbose'] , backend=self.params['backend'] \
            )

        print("FCS estimation started...")
        dFC_analyzer.estimate_all_FCS(time_series=time_series)
        print("FCS estimation done.")

        ### estimate dFCM ###

        print("dFCM estimation started...")
        # SUBJs_dFC_var for SURROGATE is dFC_var of different bootstrap SAMPLEs
        SAMPLEs_dFC_var = dFC_analyzer.estimate_all_dFCM( \
            time_series=time_series \
            )
        print("dFCM estimation done.")

        MEASURES_sample_dFC_var = {}
        for sample in SAMPLEs_dFC_var:
            for measure in SAMPLEs_dFC_var[sample]:
                # SAMPLEs_dFC_var[sample][measure] is a n_region x n_region dFC_var_mat
                if measure in MEASURES_sample_dFC_var:
                    MEASURES_sample_dFC_var[measure].append(SAMPLEs_dFC_var[sample][measure])
                else:
                    MEASURES_sample_dFC_var[measure] = []
                    MEASURES_sample_dFC_var[measure].append(SAMPLEs_dFC_var[sample][measure])

        for measure in MEASURES_sample_dFC_var:
            MEASURES_sample_dFC_var[measure] = np.array(MEASURES_sample_dFC_var[measure])

        return MEASURES_sample_dFC_var

    def calc_TH_mask(self, MEASURES_sample_dFC_var, a=0.95):

        # returns list of TH_masks for different measures

        TH_mask = {}
        for measure in MEASURES_sample_dFC_var:
            n_regions = MEASURES_sample_dFC_var[measure].shape[1]
            TH_mask_mat = np.zeros((n_regions, n_regions))
            for i in range(n_regions):
                for j in range(n_regions):
                    C = np.squeeze(MEASURES_sample_dFC_var[measure][:, i, j])
                    mu, sigma = norm.fit(C)
                    TH_mask_mat[i, j] = norm.ppf(a, mu, sigma)
            TH_mask[measure] = TH_mask_mat

        return TH_mask

    def mask_SUBJs_dFC(self, SUBJs_dFC_var, SUBJs_TH_mask):

        SUBJs_dyn_conn = {}
        for subject in SUBJs_dFC_var:
            dyn_conn = {}
            for measure in SUBJs_dFC_var[subject]:
                dyn_conn[measure] = (SUBJs_dFC_var[subject][measure]>=SUBJs_TH_mask[subject][measure])*[1] 
            SUBJs_dyn_conn[subject] = dyn_conn

        return SUBJs_dyn_conn


################################# dFC class ####################################

"""

todo:
- separate the matrix visualizing function
- brain or brain graph class
- add an updating behavior -> we can segment subjects and time_series and update the model gradually ?
- type annotation
- remove sliding window type dFC visualization 
- normalization: C_t_z[:, i, i] = np.mean(C_t_z) # ?????????????????
"""

class dFC:

    methods_name_lst = [ \
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

    base_methods_name_lst = sw_methods_name_lst + methods_name_lst

    def __init__(self):
        self.measure_name = ''
        self.is_state_based = bool()
        self._stat = []
        self.TPM = []

    @property
    def FCS(self):
        return self.FCS_

    # test
    @property
    def FCS_dict(self):
        # returns a dict inclusing each FCS to be fed to similarity assess

        if not self.is_state_based:
            return None

        C_A = self.FCS
        state_act_dict = {}
        state_act_dict['state_TC'] = {}
        for k in range(C_A.shape[0]):
            state_act_dict['state_TC']['FCS'+str(k+1)] = {}
            state_act_dict['state_TC']['FCS'+str(k+1)]['FCS'] = C_A[k,:,:]
            
        return state_act_dict

    def estimate_FCS(self, time_series=None):
        pass

    def estimate_dFCM(self, time_series=None):
        pass

    def visualize_states(self):
        pass

    # todo : use FCS_dict func in this func
    def visualize_FCS(self, normalize=True, threshold=0.0, save_image=False, fig_name=None):
        
        if self.FCS == []:
            return

        if normalize:
            C = dFC_mat_normalize(C_t=self.FCS, threshold=threshold)
        else:
            C = self.FCS

        FCS_dict = {}
        for i in range(C.shape[0]):
            FCS_dict['FCS '+str(i+1)] = C[i]

        visualize_conn_mat(data=FCS_dict, \
            title=self.measure_name+' FCS', \
            save_image=save_image, \
            output_root=fig_name, \
            fix_lim=True \
        )

    def visualize_TPM(self, normalize=True, save_image=False, fig_name=None):
        
        if self.TPM == []:
            return
        if normalize:
            C = dFC_mat_normalize(C_t=np.expand_dims(self.TPM, axis=0), threshold=0.0)
        else:
            C = np.expand_dims(self.TPM, axis=0)

        plt.figure(figsize=(5, 5))
        plt.imshow(np.squeeze(C), interpolation='nearest', aspect='equal', cmap='jet')
        cb=plt.colorbar(shrink=0.8)
        plt.title(self.measure_name + ' TPM')
        
        if save_image:
            plt.savefig(fig_name + '.png', dpi=fig_dpi)  
            plt.close()
        else:
            plt.show()

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

from hmmlearn import hmm

class HMM_CONT(dFC):

    def __init__(self, **params):
        self.measure_name = 'ContinuousHMM'
        self.is_state_based = True
        self.TPM = []
        self.FCS_ = []
        self.n_states = params['n_states']

    def estimate_FCS(self, time_series=None):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        Models, Scores = [], []
        for i in range(10):
            model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full")
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

        return self

    def estimate_dFCM(self, time_series=None):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        Z = self.hmm_model.predict(time_series.data.T)
        dFCM = DFCM(measure=self)
        dFCM.add_FC(FCSs=self.FCS_, FCS_idx=Z, subj_id_array=time_series.subj_id_array)

        return dFCM

################################## Windowless ##################################

"""
by : https://github.com/nel215/ksvd

Reference: Rubinstein, R., Zibulevsky, M. and Elad, M., Efficient Implementation 
of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit Technical 
Report - CS Technion, April 2008

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
"""
from ksvd import ApproximateKSVD

class WINDOWLESS(dFC):

    def __init__(self, **params):
        self.measure_name = 'Windowless'
        self.is_state_based = True
        self.TPM = []
        self.FCS_ = []
        self.n_states = params['n_states']
    
    def estimate_FCS(self, time_series=None):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        # time_series ~ gamma.dot(dictionary)
        self.aksvd = ApproximateKSVD(n_components=self.n_states, transform_n_nonzero_coefs=1)
        self.dictionary = self.aksvd.fit(time_series.data.T).components_
        self.gamma = self.aksvd.transform(time_series.data.T)

        self.FCS_ = np.zeros([self.n_states, time_series.n_regions, time_series.n_regions])
        for i in range(self.n_states):
            self.FCS_[i, :, :] = np.multiply(np.expand_dims(self.dictionary[i,:], axis=0).T, np.expand_dims(self.dictionary[i,:], axis=0))

        self.Z = list()
        for i in range(time_series.n_time):
            self.Z.append(np.argwhere(self.gamma[i, :] != 0)[0,0])

        return self

    def estimate_dFCM(self, time_series=None):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        gamma = self.aksvd.transform(time_series.data.T)

        Z = list()
        for i in range(time_series.n_time):
            Z.append(np.argwhere(gamma[i, :] != 0)[0,0])
            
        dFCM = DFCM(measure=self)
        dFCM.add_FC(FCSs=self.FCS_, FCS_idx=Z, subj_id_array=time_series.subj_id_array)
        return dFCM

################################# Time-Frequency #################################

"""
PyCWT: 
Authors
Sebastian Krieger, Nabil Freij, Alexey Brazhe, Christopher Torrence, 
Gilbert P. Compo and contributors.

Disclaimer
This module is based on routines provided by C. Torrence and G. P. Compo available 
at http://paos.colorado.edu/research/wavelets/, on routines provided by A. Grinsted, 
J. Moore and S. Jevrejeva available at 
http://noc.ac.uk/using-science/crosswavelet-wavelet-coherence, and on routines 
provided by A. Brazhe available at http://cell.biophys.msu.ru/static/swan/.

This software is released under a BSD-style open source license. Please read 
the license file for further information. This routine is provided as is 
without any express or implied warranties whatsoever.

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.
    dj : float, optional
        Spacing between discrete scales. Default value is 1/12.
        Smaller values will result in better scale resolution, but
        slower calculation and plot.
    s0 : float, optional
        Smallest scale of the wavelet. Default value is 2*dt.
    J : float, optional
        Number of scales less one. Scales range from s0 up to
        s0 * 2**(J * dj), which gives a total of (J + 1) scales.
        Default is J = (log2(N*dt/so))/dj.
    sig : bool 
        set to compute signficance, default is True
    significance_level (float, optional) :
        Significance level to use. Default is 0.95.
    normalize (boolean, optional) :
        If set to true, normalizes CWT by the standard deviation of
        the signals.

- if n_jobs is None => no parallelization

todo:

- consider COI and edge effect in averaging:
    => should we truncate the time points having at less than 20 freqs as done in Savva et al. ?

"""
import pycwt as wavelet

class TIME_FREQ(dFC):

    def __init__(self, method='WTC', coi_correction=True, **params):
        
        assert method in self.methods_name_lst, \
            "Time-frequency method not recognized."

        self.measure_name_ = 'Time-Freq '
        self.is_state_based = False
        self.TPM = []
        self.FCS_ = []
        self.method_ = method
        self.coi_correction_ = coi_correction
    
    @property
    def coi_correction(self):
        return self.coi_correction_

    @property
    def method(self):
        return self.method_

    @property
    def measure_name(self):
        return self.measure_name_ + '_' + self.method

    def coi_correct(self, X, coi, freqs):
        # correct the edge effect in matrix X = [freqs, time] using coi
        # if self.coi_correction=True

        if not self.coi_correction:
            return X
        periods = 1/freqs
        periods = np.repeat(periods[:, None], X.shape[1], axis=1)
        coi = np.repeat(coi[None, :], X.shape[0], axis=0)
        X_corrected = np.multiply(X, (coi>=periods))
        return X_corrected

    def WT_dFC(self, Y1, Y2, Fs, J, s0, dj):
        if self.method_=='CWT_mag' or self.method_=='CWT_phase_r' or self.method_=='CWT_phase_a':
            # Cross Wavelet Transform
            WT_xy, coi, freqs, _ = wavelet.xwt(Y1, Y2, dt=1/Fs, dj=dj, s0=s0, J=J, 
                significance_level=0.95, wavelet='morlet', normalize=True)

            if self.method_=='CWT_mag':
                WT_xy_corrected = self.coi_correct(WT_xy, coi, freqs)
                wt = np.abs(np.mean(WT_xy_corrected, axis=0))

            if self.method_=='CWT_phase_r' or self.method_=='CWT_phase_a':
                cosA = np.cos(np.angle(WT_xy))
                sinA = np.sin(np.angle(WT_xy))

                cosA_corrected = self.coi_correct(cosA, coi, freqs)
                sinA_corrected = self.coi_correct(sinA, coi, freqs)

                A = (cosA_corrected + sinA_corrected * 1j)

                if self.method_=='CWT_phase_r':
                    wt = np.abs(np.mean(A, axis=0))
                else:
                    wt = np.angle(np.mean(A, axis=0))
        
        if self.method_=='WTC':
            # Wavelet Transform Coherence
            WT_xy, _, coi, freqs, _ = wavelet.wct(Y1, Y2, dt=1/Fs, dj=dj, s0=s0, J=J, 
                sig=False, significance_level=0.95, wavelet='morlet', normalize=True)
            WT_xy_corrected = self.coi_correct(WT_xy, coi, freqs)
            wt = np.abs(np.mean(WT_xy_corrected, axis=0))

        return wt

    def estimate_dFCM(self, time_series=None):
        
        '''
        we assume calc is applied on subjects separately
        '''

        # params
        J = 50 # -1
        s0 = 1 # -1
        dj = 1/8 # 1/12

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        WT = np.zeros((time_series.n_time, \
            time_series.n_regions, time_series.n_regions))

        for i in range(time_series.n_regions):
            if self.params['n_jobs'] is None:
                Q = list()
                for j in range(time_series.n_regions):
                    Q.append(self.WT_dFC( \
                        Y1=time_series.data[i, :], \
                        Y2=time_series.data[j, :], \
                        Fs=time_series.Fs, \
                        J=J, s0=s0, dj=dj))
            else:
                Q = Parallel( \
                    n_jobs=self.params['n_jobs'], verbose=self.params['verbose'], backend=self.params['backend'])( \
                    delayed(self.WT_dFC)( \
                                    Y1=time_series.data[i, :], \
                                    Y2=time_series.data[j, :], \
                                    Fs=time_series.Fs, \
                                    J=J, s0=s0, dj=dj) \
                                    for j in range(time_series.n_regions) \
                                                                )
            WT[:, i, :] = np.array(Q).T

        dFCM = DFCM(measure=self)
        dFCM.add_FC(FCSs=WT, subj_id_array=time_series.subj_id_array)
        return dFCM

################################# Sliding-Window #################################

"""

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
"""

from sklearn.covariance import GraphicalLassoCV

class SLIDING_WINDOW(dFC):

    def __init__(self, sw_method='pear_corr', tapered_window=True, **params):

        assert sw_method in self.sw_methods_name_lst, \
            "sw_method not recognized."

        self.measure_name_ = 'SlidingWindow'
        self.is_state_based = False
        self.sw_method_ = sw_method
        self.TPM = []
        self.FCS_ = []
        self.W = params['W']
        self.n_overlap = params['n_overlap']
        self.tapered_window = tapered_window
    
    @property
    def measure_name(self):
        return self.measure_name_ + '_' + self.sw_method
        
    @property
    def sw_method(self):
        return self.sw_method_

    def shan_entropy(self, c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        H = -sum(c_normalized* np.log2(c_normalized))  
        return H

    def calc_MI(self, X, Y):
        
        bins = 20
        
        c_XY = np.histogram2d(X,Y,bins)[0]
        c_X = np.histogram(X,bins)[0]
        c_Y = np.histogram(Y,bins)[0]
        
        H_X = self.shan_entropy(c_X)
        H_Y = self.shan_entropy(c_Y)
        H_XY = self.shan_entropy(c_XY)
        
        MI = H_X + H_Y - H_XY
        return MI

    def FC(self, time_series):
    
        if self.sw_method=='GraphLasso':
            model = GraphicalLassoCV()
            model.fit(time_series.T)
            C = model.covariance_
        else:
            C = np.zeros((time_series.shape[0], time_series.shape[0]))
            for i in range(time_series.shape[0]):
                for j in range(i, time_series.shape[0]):
                    
                    X = time_series[i, :]
                    Y = time_series[j, :]

                    if self.sw_method=='MI':
                        ########### Mutual Information ##############
                        C[j, i] = self.calc_MI(X, Y)
                    else:
                        ########### Pearson Correlation ##############
                        C[j, i] = np.corrcoef(X, Y)[0, 1]

                    C[i, j] = C[j, i]   
                
        return C

    def dFC(self, time_series, subj_id, W=None, n_overlap=None, tapered_window=False):
        # W is in time samples
        
        L = time_series.shape[1]
        step = int((1-n_overlap)*W)
        if step == 0:
            step = 1

        window_taper = signal.windows.gaussian(W, std=3*W/22)
        C = DFCM(measure=self)
        for l in range(0, L-W+1, step):

            ######### creating a rectangel window ############
            window = np.zeros((L))
            window[l:l+W] = 1
            
            ########### tapering the window ##############
            if tapered_window:
                window = signal.convolve(window, window_taper, mode='same') / sum(window_taper)

            window = np.repeat(np.expand_dims(window, axis=0), time_series.shape[0], axis=0)

            # int(l-W/2):int(l+3*W/2) is the nonzero interval after tapering
            C.add_FC(FCSs=self.FC( \
                        np.multiply(time_series, window)[ \
                            :,max(int(l-W/2),0):min(int(l+3*W/2),L) \
                                ] \
                                    ), \
                        subj_id_array = subj_id, \
                        TR_array=np.array( [ int((l + (l+W)) / 2) ] ) \
                        )
            # print('dFC step = %d' %(l))

        return C
    
    def estimate_dFCM(self, time_series=None):
        
        '''
        we assume calc is applied on subjects separately
        '''
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        # W is converted from sec to samples
        dFCM = self.dFC(time_series=time_series.data, \
            subj_id=time_series.subj_id_array[:1], \
            W=int(self.W * time_series.Fs) , \
            n_overlap=self.n_overlap, \
            tapered_window=self.tapered_window \
            )

        return dFCM


########################### Sliding_Window + Clustering ############################

"""
- We used a tapered window as in Allen et al., created by convolving a rectangle (width = 22 TRs = 44s) 
  with a Gaussian (σ = 3 TRs) and slid in steps of 1 TR, resulting in W= 126 windows (Allen et al., 2014).
- Kmeans Clustering is repeated 500 times to escape local minima (Allen et al., 2014)
- for clustering, we have a 2-level kmeans clustering. First, we cluster FCSs of each subject. Then, we
    cluster all clustering centers from all subjects. the final estimate_dFCM is using the second kmeans
    model (Allen et al., 2014; Ou et al., 2015). 

Parameters
    ----------
    y1, y2 : numpy.ndarray, list
        Input signals.
    dt : float
        Sample spacing.

todo:
- pyclustering(manhattan) has a problem when suing predict
"""

from sklearn.cluster import KMeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric

class SLIDING_WINDOW_CLUSTR(dFC):

    def __init__(self, base_method='pear_corr', clstr_distance='euclidean',
    tapered_window=True, **params):

        assert clstr_distance=='euclidean' or clstr_distance=='manhattan', \
            "Clustering distance not recognized. It must be either \
                euclidean or manhattan."

        assert base_method in self.base_methods_name_lst, \
            "Base method not recognized."
    
        self.measure_name_ = 'Clustering'
        self.is_state_based = True
        self.clstr_distance = clstr_distance
        self.TPM = []
        self.FCS_ = []
        self.base_method_ = base_method
        self.n_states = params['n_states']
        self.n_subj_clstrs = params['n_subj_clstrs']
        self.W = params['W']
        self.n_overlap = params['n_overlap']
        self.tapered_window = tapered_window
    
    @property
    def base_method(self):
        return self.base_method_

    @property
    def measure_name(self):
        return self.measure_name_ + '_' + self.base_method

    def dFC_mat2vec(self, C_t):
        F = list()
        for t in range(C_t.shape[0]):
            C = C_t[t, : , :]
            F.append(C[np.triu_indices(C_t.shape[1])])

        F = np.array(F)
        return F

    def dFC_vec2mat(self, F, N):
        C = list()
        iu = np.triu_indices(N)
        for i in range(F.shape[0]):
            K = np.zeros((N, N))
            K[iu] = F[i,:]
            K = K + np.multiply(K.T, 1-np.eye(N))
            C.append(K)
        C = np.array(C)
        return C

    def clusters_lst2idx(self, clusters):
        Z = np.zeros((self.F.shape[0],))
        for i, cluster in enumerate(clusters):
            for sample in cluster:
                Z[sample] = i
        return Z.astype(int)

    def cluster_FC(self, FCS_raw, n_clusters, n_regions):

        F = self.dFC_mat2vec(FCS_raw)

        if self.clstr_distance=='manhattan':
            pass
            # ########### Manhattan Clustering ##############
            # # Prepare initial centers using K-Means++ method.
            # initial_centers = kmeans_plusplus_initializer(F, self.n_states).initialize()
            # # create metric that will be used for clustering
            # manhattan_metric = distance_metric(type_metric.MANHATTAN)
            # # Create instance of K-Means algorithm with prepared centers.
            # kmeans_ = kmeans(F, initial_centers, metric=manhattan_metric)
            # # Run cluster analysis and obtain results.
            # kmeans_.process()
            # Z = self.clusters_lst2idx(kmeans_.get_clusters())
            # F_cent = np.array(kmeans_.get_centers())
        else:
            ########### Euclidean Clustering ##############
            kmeans_ = KMeans(n_clusters=n_clusters, n_init=500).fit(F)
            Z = kmeans_.predict(F)
            F_cent = kmeans_.cluster_centers_

        FCS_ = self.dFC_vec2mat(F_cent, N=n_regions)
        return FCS_, kmeans_

        
    def estimate_FCS(self, time_series=None):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        if self.base_method=='CWT_mag' or self.base_method=='CWT_phase_r' \
            or self.base_method=='CWT_phase_a' or self.base_method=='WTC':
            params = {'n_jobs': self.params['n_jobs'], 'verbose': self.params['verbose'], 'backend': self.params['backend']}
            base_dFC = TIME_FREQ(method=self.base_method, **params)
        else:
            params = {'W': self.W, 'n_overlap': self.n_overlap}
            base_dFC = SLIDING_WINDOW(sw_method=self.base_method, \
                    tapered_window=self.tapered_window, **params)

        # 1-level clustering
        # dFCM_raw = base_dFC.estimate_dFCM( \
        #         time_series=time_series \
        #         )
        # self.FCS_, self.kmeans_ = self.cluster_FC( \
        # dFCM_raw.get_dFC_mat(TRs=self.dFCM_raw.TR_array), \
        # n_regions = dFCM_raw.n_regions \
        # )

        # 2-level clustering
        SUBJECTs = list(set(time_series.subj_id_array))
        FCS_1st_level = None
        for subject in SUBJECTs:
            
            dFCM_raw = base_dFC.estimate_dFCM( \
                time_series=time_series.get_subj_ts(subj_id=subject) \
                )

            if dFCM_raw.n_time<self.n_subj_clstrs:
                print( \
                    'Number of subject-level clusters cannot be more than SW dFCM samples! n_subj_clstrs was changed to ' \
                        + str(dFCM_raw.n_time))
                self.n_subj_clstrs = dFCM_raw.n_time

            FCS, _ = self.cluster_FC( \
                FCS_raw = dFCM_raw.get_dFC_mat(TRs=dFCM_raw.TR_array), \
                n_clusters = self.n_subj_clstrs, \
                n_regions = dFCM_raw.n_regions \
                )
            if FCS_1st_level is None:
                FCS_1st_level = FCS
            else:
                FCS_1st_level = np.concatenate((FCS_1st_level, FCS), axis=0)
        
        self.FCS_, self.kmeans_ = self.cluster_FC( \
            FCS_raw=FCS_1st_level, \
            n_clusters = self.n_states, \
            n_regions = dFCM_raw.n_regions \
            )

        return self

    def estimate_dFCM(self, time_series=None):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        if self.base_method=='CWT_mag' or self.base_method=='CWT_phase_r' \
            or self.base_method=='CWT_phase_a' or self.base_method=='WTC':
            params = {'n_jobs': self.params['n_jobs'], 'verbose': self.params['verbose'], 'backend': self.params['backend']}
            base_dFC = TIME_FREQ(method=self.base_method, **params)
        else:
            params = {'W': self.W, 'n_overlap': self.n_overlap}
            base_dFC = SLIDING_WINDOW(sw_method=self.base_method, \
                    tapered_window=self.tapered_window, **params)
                    
        dFCM_raw = base_dFC.estimate_dFCM(time_series=time_series)

        F = self.dFC_mat2vec(dFCM_raw.get_dFC_mat(TRs=dFCM_raw.TR_array))

        if self.clstr_distance=='manhattan':
            pass
            # ########### Manhattan Clustering ##############
            # self.kmeans_.predict(F)
            # Z = self.clusters_lst2idx(self.kmeans_.get_clusters())
        else:
            ########### Euclidean Clustering ##############
            Z = self.kmeans_.predict(F)

        dFCM = DFCM(measure=self)
        dFCM.add_FC(FCSs=self.FCS_, \
            FCS_idx=Z, \
            subj_id_array=dFCM_raw.subj_id_array, \
            TR_array=dFCM_raw.TR_array \
            )

        return dFCM

################################# HMM Discrete #################################

"""

Parameters
    ----------
    Z : numpy.ndarray
        state time course
    M : int
        (num of observations/n_state) of 16
    N : int
        (num of hidden states) of 24
    self.FCC_ : 
        dFCM estimated by Clustering which is then used to fit Discrete HMM 
    self.FCS_ : 
        collection FCS pattern coded in numbers for Discrete HMM

todo:
- two-level hierarchical clustering ?
- find a better name for FCC
"""
# from HMM_discrete import *
from hmmlearn import hmm

class HMM_DISC(dFC):

    def __init__(self, base_method='pear_corr', tapered_window=True, **params):
        
        assert base_method in self.base_methods_name_lst, \
            "Base method not recognized."
            
        self.measure_name_ = 'DiscreteHMM'
        self.is_state_based = True
        self.TPM = []
        self.FCS_ = []
        self.base_method_ = base_method
        self.swc = None
        self.n_states = params['n_states']
        self.n_subj_clstrs = params['n_subj_clstrs']
        self.n_hid_states = params['n_hid_states']
        self.W = params['W']
        self.n_overlap = params['n_overlap']
        self.tapered_window = tapered_window

    @property
    def base_method(self):
        return self.base_method_

    @property
    def measure_name(self):
        return self.measure_name_ + '_' + self.base_method

    def estimate_FCS(self, time_series=None):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        # self.n_regions = time_series.n_regions
        # self.n_time = time_series.n_time

        params = {'W': self.W, 'n_overlap': self.n_overlap, \
            'n_subj_clstrs': self.n_subj_clstrs, 'n_states': self.n_states, \
            'n_jobs': self.params['n_jobs'], 'verbose': self.params['verbose'], 'backend': self.params['backend']}
        self.swc = SLIDING_WINDOW_CLUSTR(base_method=self.base_method, \
            tapered_window=self.tapered_window, **params)
        self.swc.estimate_FCS(time_series=time_series)
        self.FCC_ = self.swc.estimate_dFCM(time_series=time_series)

        self.hmm_model = hmm.MultinomialHMM(n_components=self.n_hid_states)
        self.hmm_model.fit(self.FCC_.FCS_idx_array.reshape(-1, 1))

        self.Z = self.hmm_model.predict(self.FCC_.FCS_idx_array.reshape(-1, 1))
        self.TPM = self.hmm_model.transmat_
        self.EPM = self.hmm_model.emissionprob_ 

        self.FCS_ = np.zeros((self.n_hid_states, \
            time_series.n_regions, time_series.n_regions))
        for i in range(self.n_hid_states):
            if len(np.argwhere(self.Z==i))>0:
                self.FCS_[i,:,:] = np.mean(self.FCC_.get_dFC_mat(\
                    TRs=self.FCC_.TR_array[np.squeeze(np.argwhere(self.Z==i))]\
                        ), axis=0)  # III

        return self

    def estimate_dFCM(self, time_series=None):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        FCC = self.swc.estimate_dFCM(time_series=time_series)

        Z = self.hmm_model.predict(FCC.FCS_idx_array.reshape(-1, 1))

        dFCM = DFCM(measure=self)
        dFCM.add_FC(FCSs=self.FCS_, \
            FCS_idx=Z, \
            subj_id_array=FCC.subj_id_array, \
            TR_array=FCC.TR_array \
                )

        return dFCM
    
###################################################################################

################################# TIME_SERIES class ######################################

"""

todo:
- select nodes for visualizer
- default node list is chosen by arange !
"""

class TIME_SERIES():
    def __init__(self, data=None, subj_id=None, Fs=None, time_array=None, \
                locs=None, nodes_info=None, TS_name='', session_name=''):
        
        '''
        subj_id is an id to identify the subjects
        '''

        assert (not data is None) and (not Fs is None) and (not subj_id is None), \
            "data, subj_id, and Fs args must be provided."

        self.data_ = data
        self.subj_id_array_ = [subj_id] * data.shape[1] 
        self.Fs_ = Fs
        self.TS_name_ = TS_name
        self.session_name_ = session_name
        self.n_regions_ = self.data_.shape[0]
        self.n_time_ = self.data_.shape[1]

        assert self.n_regions_ < self.n_time_, \
            "Probably you have to transpose the time_series."

        if time_array is None:
            self.time_array_ = 1/self.Fs_ + np.arange(0, self.data_.shape[1]/self.Fs_, 1/self.Fs_)
        else:
            self.time_array_ = time_array

        self.locs_ = locs
        self.nodes_info_ = nodes_info

        self.interval_ = list(range(self.n_time_))
        self.nodes_selection_ = list(range(self.n_regions_))
    
    @classmethod
    def from_numpy(cls):
        pass

    @property
    def data(self):
        return self.data_[self.nodes_lst, self.interval]

    @property
    def subj_id_array(self):
        return [self.subj_id_array_[i] for i in self.interval[0]]

    @property
    def nodes_lst(self):
        # output shape is (n_region, 1) 
        return np.array(self.nodes_selection_)[:, np.newaxis]
        
    @property
    def interval(self):
        # output shape is (1, n_time) 
        return np.array(self.interval_)[np.newaxis, :]

    @property
    def locs(self):
        if self.locs_ is None:
            return None
        else:
            return self.locs_[self.nodes_lst, :]

    @property
    def nodes_info(self):
        if self.nodes_info_ is None:
            return None
        else:
            return [self.nodes_info_[0]] + [self.nodes_info_[i[0]+1] for i in self.nodes_lst] 

    @property
    def Fs(self):
        return self.Fs_

    @property
    def n_time(self):
        return self.data.shape[1]

    @property
    def n_regions(self):
        return self.data.shape[0]

    @property
    def time(self):
        return self.time_array_[self.interval[0]]

    @property
    def TS_name(self):
        return self.TS_name_

    def resample(self):
        # change self.Fs_
        pass

    def get_subj_ts(self, subj_id=None):
        """
        you can select time samples by their subj_id
        ! be careful about the original properties of TS hidden in new TS
        node selection will be kept.
        """
        TS_temp = deepcopy(self)
        idx = [i for i,j in enumerate(self.subj_id_array) if j==subj_id]
        TS_temp.truncate(start_point=idx[0], end_point=idx[-1])

        new_TS = TIME_SERIES(data=TS_temp.data, subj_id=subj_id, Fs=self.Fs_, \
            time_array=TS_temp.time, locs=self.locs, nodes_info=self.nodes_info, \
            TS_name=self.TS_name+' subject '+subj_id)
        new_TS.time_array_ = new_TS.time_array_ - (new_TS.time_array_[0] - 1/new_TS.Fs_)

        return new_TS


    def append_ts(self, new_time_series=None, time_array=None, subj_id=None):
        # append new time series to existing ones
        # truncate will not be considered anymore, while node selection is; 
        # the whole old time series will be concat to new one
        # append_ts resets the truncate but not the node selection
        # the new ts will be appended to the original data and then node_selection is applied again
        # we assume the new time array starts from about 0 (or 1/Fs)

        assert self.n_regions_ == new_time_series.shape[0], \
            "Number of nodes mismatch."

        assert not subj_id is None, \
            "subj_id must be provided."

        if time_array is None:
            time_array = 1/self.Fs_ + np.arange(0, new_time_series.shape[1]/self.Fs_, 1/self.Fs_)
        time_array = self.time_array_[-1] + time_array

        self.data_ = np.concatenate((self.data_, new_time_series), axis=1)
        self.time_array_ = np.concatenate((self.time_array_, time_array), axis=0)
        self.subj_id_array_ = self.subj_id_array_ + [subj_id] * new_time_series.shape[1]
        self.n_time_ = self.data_.shape[1]
        self.interval_ = list(range(self.n_time_))


    def truncate(self, start_time=None, end_time=None, start_point=None, end_point=None):

        # based on either time or samples
        # if all None -> whole time_series
        #check if not out of total interval

        start = 0
        end = self.n_time_

        if not start_point is None:
            start = start_point
        
        if not end_point is None:
            end = end_point + 1

        if not start_time is None:
            start = np.argwhere(self.time_array_>=start_time)[0,0]

        if not end_time is None:
            end = np.argwhere(self.time_array_<=end_time)[-1,0] + 1
        
        self.interval_ = list(range(start, end))
                
    def select_nodes(self, nodes_idx=None):
        # select the nodes indexed by numbers in nodes_idx. nodes_idx is a numpy 1D array
        # if nodes_idx is None -> all the nodes will be considered (resets node selection)
        # if nodes_idx is not sorted, it can be used to reorder the nodes

        if nodes_idx is None:
            self.nodes_selection_ = list(range(self.n_regions_))
        else:
            self.nodes_selection_ = nodes_idx    

    def visualize(self, start_time=None, end_time=None, \
        nodes_lst=None, \
        save_image=False, fig_name=None):

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
        plt.title(self.TS_name_ + ' ' + self.session_name_ )
        if save_image:
            plt.savefig(fig_name + '.png', dpi=fig_dpi)  
            plt.close()
        else:
            plt.show()


################################# dFCM class ######################################

"""
Parameters
    ----------
    TR_array : an array labeling 
        timepoints by their TRs

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

class DFCM():
    def __init__(self, measure=None):

        # assert not measure is None, \
        #     "measure arg must be provided."
        self.measure_ = measure
        self.FCSs_ = None # is a dict
        self.FCS_idx_ = None # is a dict
        self.subj_id_array_ = None
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
    def subj_id_array(self):
        return self.subj_id_array_

    
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
    def dFC2dict(self, TRs=None, num_samples=None):
        # return dFC samples as a dictionary
        if TRs is None:
            TRs = self.TR_array
        if type(TRs) is list:
            TRs = np.array(TRs)
        TRs = TRs.astype(int)
        dFC_mat, TRs_new = self.get_dFC_mat(TRs=TRs, num_samples=num_samples)
        dFC_dict = {}
        for k, TR in enumerate(TRs_new):
            dFC_dict['TR'+str(TR)] = dFC_mat[k, :, :]
        return dFC_dict

    # test this
    def get_dFC_mat(self, TRs=None, num_samples=None):
        # get dFC matrices corresponding to 
        # the specified TRs 
        # TRs should be list not necessarily in order ?
        # if num_samples specified, it will downsample 
        # TRs to reach that number of samples
        # if num_samples > len(TRs) -> picks all TRs

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

    # def concat(self, dFCM):

    #     # test this method

    #     assert type(dFCM) is DFCM, \
    #             "The input must be of DFCM class"

    #     if self.FCPs_ is None:
    #         self.FCPs_ = dFCM.FCPs
    #         self.FCP_idx_ = dFCM.FCP_idx
    #         self.n_regions_ = dFCM.n_regions
    #         self.n_time_ = dFCM.n_time
    #         self.TR_array_ = dFCM.TR_array
    #     else:
    #         assert self.n_regions== dFCM.n_regions, \
    #             "dFCM region numbers missmatch."
    #         FCP_idx = dFCM.FCP_idx + self.FCPs.shape[0]
    #         self.FCPs_ = np.concatenate((self.FCPs_, dFCM.FCPs), axis=0)
    #         self.FCP_idx_ = np.concatenate((self.FCP_idx_, FCP_idx), axis=0)
    #         self.n_time_ = self.FCP_idx.shape[0]
    #         self.TR_array_ = np.concatenate((self.TR_array, dFCM.TR_array))

    def add_FC(self, FCSs, FCS_idx=None, subj_id_array=None, TR_array=None):
        
        if len(FCSs.shape)==2:
            FCSs = np.expand_dims(FCSs, axis=0)

        if FCS_idx is None:
            FCS_idx = np.arange(start=0, stop=FCSs.shape[0], step=1)

        if type(FCS_idx) is list:
            FCS_idx = np.array(FCS_idx)

        if len(FCS_idx.shape)>1:
            FCS_idx = np.squeeze(FCS_idx)

        if not type(subj_id_array) is list:
            subj_id_array = list(subj_id_array)
        
        assert FCSs.shape[1] == FCSs.shape[2], \
                "FC matrices must be square."

        assert len(subj_id_array)==FCS_idx.shape[0], \
            "FCS_idx and subj_id_array length mismatch."

        if TR_array is None:
            TR_array = np.arange(start=self.n_time+1, stop=self.n_time+len(FCS_idx)+1, step=1)

        assert np.sum(np.abs(np.sort(TR_array)-TR_array))==0.0, \
            'TRs not sorted !'

        # the input FCS_idx is ranged from 0 to len(FCS)-1 but we shift it to 1 to len(FCS)
        if self.FCSs_ is None:
            self.FCSs_ = {}
            for i, FCS in enumerate(FCSs):
                self.FCSs_['FCS'+str(i+1)] = FCS

            self.FCS_idx_ = {}
            for i, idx in enumerate(FCS_idx):
                self.FCS_idx_['TR'+str(TR_array[i])] = 'FCS'+str(idx+1)

            self.subj_id_array_ = subj_id_array
            self.n_regions_ = FCSs.shape[1]
            self.n_time_ = len(self.FCS_idx_)
            self.TR_array_ = TR_array
        else:
            # test this part
            assert self.n_regions == FCSs.shape[1], \
                "FCS region numbers mismatch."

            FCS_idx_offset = len(self.FCSs_)+1 # test FCS_idx_offset

            for i, FCS in enumerate(FCSs):
                assert not 'FCS'+str(i+FCS_idx_offset) in self.FCSs_, \
                    'key already exists in self.FCSs_ !' 
                self.FCSs_['FCS'+str(i+FCS_idx_offset)] = FCS

            for i, idx in enumerate(FCS_idx):
                assert not 'TR'+str(TR_array[i]) in self.FCS_idx_, \
                    'key already exists in self.FCS_idx_ !' 
                assert TR_array[i] > self.TR_array_[-1], \
                    'TR overlap !' 
                self.FCS_idx_['TR'+str(TR_array[i])] = 'FCS'+str(idx+FCS_idx_offset) 

            self.subj_id_array_ = self.subj_id_array_ + subj_id_array
            self.n_time_ = len(self.FCS_idx_) 
            self.TR_array_ = np.concatenate((self.TR_array, TR_array))

    def visualize_dFC(self, TRs=None, normalize=True, \
        threshold=0.0, save_image=False, fig_name=None, fix_lim=True):

        assert not self.measure is None, \
            'Measure is not provided.'

        if TRs is None:
            TRs = self.TR_array

        if normalize:
            C = dFC_mat_normalize(C_t=self.get_dFC_mat(TRs=TRs), \
                global_normalization=True, threshold=threshold)
        else:
            C = self.get_dFC_mat(TRs=TRs)

        dFC_dict = {}
        for i, TR in enumerate(TRs):
            dFC_dict['TR'+str(TR)] = C[i]

        visualize_conn_mat(data=dFC_dict, \
            title=self.measure.measure_name+' dFC', \
            save_image=save_image, \
            output_root=fig_name, \
            fix_lim=fix_lim \
        )


