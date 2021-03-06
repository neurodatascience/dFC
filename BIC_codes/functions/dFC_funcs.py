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

def visualize_corr_mat():
    pass

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

############################# dFC Analyzer class ################################

"""

todo:
- 
"""

import time

class DFC_ANALYZER:
    # if self.n_jobs is None => no parallelization

    def __init__(self, MEASURES_lst, vis_TR_idx,\
            save_image=False, output_root=None, \
                n_jobs=-1, verbose=1, backend='loky'):
        self.analysis_name = ''
        self.MEASURES_lst_ = MEASURES_lst
        self.vis_TR_idx = vis_TR_idx # to visualize
        self.save_image = save_image
        self.output_root = output_root
        self.n_jobs = n_jobs
        self.verbose = verbose 
        self.backend = backend

        self.methods_corr_lst_ = list()

    @property
    def methods_corr(self):
        return np.mean(self.methods_corr_lst, axis=0)

    @property
    def methods_corr_lst(self):
        return np.array(self.methods_corr_lst_)

    @property
    def MEASURES_lst(self):
        return self.MEASURES_lst_

    @property
    def SB_MEASURES_lst(self): # returns state_based measures
        SB_MEASURES = list()
        for measure in self.MEASURES_lst:
            if measure.is_state_based:
                SB_MEASURES.append(measure)
        return SB_MEASURES

    @property
    def NSB_MEASURES_lst(self): # returns non_state_based measures
        NSB_MEASURES = list()
        for measure in self.MEASURES_lst:
            if not measure.is_state_based:
                NSB_MEASURES.append(measure)
        return NSB_MEASURES

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

    def dynamic_conns(self):
        pass

    def analyze(self, time_series=None):

        ### estimate FCS ###

        print("FCS estimation started...")
        self.estimate_FCS(time_series=time_series)
        print("FCS estimation done.")

        ### Visualize FCS ###

        self.visualize_FCS(normalize=True, \
                                threshold=0.0, \
                                )

        ### estimate dFCM ###

        SUBJECTs = list(set(time_series.subj_id_array))

        print("dFCM estimation started...")
        if self.n_jobs is None:
            for subject in SUBJECTs:
                self.methods_corr_lst_.append( \
                    self.subj_lvl_analysis( \
                    time_series=time_series.get_subj_ts(subj_id=subject) \
                    ))
        else:
            new_methods_corr_lst = Parallel( \
                        n_jobs=self.n_jobs, \
                        verbose=self.verbose, \
                        backend=self.backend)( \
                    delayed(self.subj_lvl_analysis)( \
                        time_series=time_series.get_subj_ts(subj_id=subject) \
                        ) \
                        for subject in SUBJECTs)
            self.methods_corr_lst_ = new_methods_corr_lst
        print("dFCM estimation done.")

        #### Methods dFC Corr MAT ###

        self.visualize_dFC_corr()

    def subj_lvl_analysis(self, time_series):

        dFCM_lst = self.estimate_dFCM(time_series=time_series)

        self.visualize_dFCMs(dFCM_lst=dFCM_lst, \
            TR_idx=self.vis_TR_idx, \
            subj_id=time_series.subj_id_array[0], \
            )

        return self.dFC_corr_mat(dFCM_lst=dFCM_lst)

    def estimate_FCS(self, time_series=None):
        SB_MEASURES_lst = self.SB_MEASURES_lst
        if self.n_jobs is None:
            SB_MEASURES_lst_NEW = list()
            for measure in SB_MEASURES_lst:
                SB_MEASURES_lst_NEW.append( \
                    measure.estimate_FCS(time_series=time_series) \
                    )
        else:
            SB_MEASURES_lst_NEW = Parallel( \
                n_jobs=self.n_jobs, verbose=self.verbose, backend=self.backend)( \
                delayed(measure.estimate_FCS)(time_series=time_series) \
                    for measure in SB_MEASURES_lst)
        self.MEASURES_lst_ = self.NSB_MEASURES_lst + SB_MEASURES_lst_NEW

    def estimate_dFCM(self, time_series=None):
        if self.n_jobs is None:
            dFCM_lst = list()
            for measure in self.MEASURES_lst:
                dFCM_lst.append( \
                    measure.estimate_dFCM(time_series=time_series) \
                )
        else:
            dFCM_lst = Parallel( \
                n_jobs=self.n_jobs, verbose=self.verbose, backend=self.backend)( \
                delayed(measure.estimate_dFCM)(time_series=time_series) \
                    for measure in self.MEASURES_lst)
        return dFCM_lst

    def dFC_corr(self, dFCM_i, dFCM_j):

        # returns correlation of dFC measures over time

        TRs = TR_intersection([dFCM_i, dFCM_j])
        dFC_mat_i = dFCM_i.get_dFC_mat(TRs=TRs)
        dFC_mat_j = dFCM_j.get_dFC_mat(TRs=TRs)
        corr = list()
        for t in range(len(TRs)):
            corr.append(np.corrcoef(dFC_mat_i[t,:,:].flatten(), dFC_mat_j[t,:,:].flatten())[0,1])
        corr= np.array(corr)
        return corr

    def dFC_corr_mat(self, dFCM_lst):

        # returns averaged correlation of dFC measures 

        a = 0.1 # portion of the dFCs to ignore from the beginning and the end
        methods_corr = np.zeros((len(dFCM_lst), len(dFCM_lst)))
        for i in range(len(dFCM_lst)):
            for j in range(i+1, len(dFCM_lst)):
                corr_ij = self.dFC_corr( \
                    dFCM_lst[i], dFCM_lst[j] \
                        )
                methods_corr[i,j] = np.mean(corr_ij[ \
                    int(len(corr_ij)*a) : int(len(corr_ij)*(1-a)) \
                        ])
                methods_corr[j,i] = methods_corr[i,j] 
        return methods_corr

    def visualize_dFC_corr(self):

        # visualize avergaed dFC corr mat

        measure_list = list()
        for measure in self.MEASURES_lst:
            measure_list.append(measure.measure_name)
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(self.methods_corr, interpolation='nearest', aspect='equal', cmap='jet')
        ax.set_xticks(np.arange(len(measure_list)))
        ax.set_yticks(np.arange(len(measure_list)))
        ax.set_xticklabels(measure_list, rotation=90)
        ax.set_yticklabels(measure_list)
        cb=fig.colorbar(im, shrink=0.8)
        plt.suptitle('Correlation of measured dFC')
        if self.save_image:
            output_root = self.output_root+'dFC/'
            fig_name = output_root + 'avg_dFC_corr'
            plt.savefig(fig_name + '.png', dpi=fig_dpi)  
            plt.close()
        else:
            plt.show()

    def visualize_dFCMs(self, dFCM_lst=None, TR_idx=None, normalize=True, threshold=0.0, \
                            fix_lim=True, subj_id=''):
        
        # TR_idx is not TR values, but their indices!

        TRs = TR_intersection(dFCM_lst)
        if not TR_idx is None:
            assert not np.any(np.array(TR_idx)>=len(TRs)), \
                'TR_idx out of range.'
            TRs = [TRs[i] for i in TR_idx]

        for dFCM in dFCM_lst:
            if self.save_image:
                output_root = self.output_root+'dFC/'
                dFCM.visualize_dFC(TRs=TRs, normalize=normalize, threshold=threshold, \
                    fix_lim=fix_lim, \
                    save_image=self.save_image, \
                    fig_name= output_root+'subject'+subj_id+'_'+dFCM.measure.measure_name+'_dFC')
            else:
                dFCM.visualize_dFC(TRs=TRs, normalize=normalize, threshold=threshold, fix_lim=fix_lim)

    def visualize_FCS(self, normalize=True, threshold=0.0):

        for measure in self.MEASURES_lst:  
            if self.save_image:
                output_root = self.output_root + 'FCS/'
                measure.visualize_FCS(normalize=normalize, threshold=threshold, save_image=True, \
                    fig_name= output_root + measure.measure_name + '_FCS')
                # measure.visualize_TPM(normalize=normalize)
            else:
                measure.visualize_FCS(normalize=normalize, threshold=threshold) # normalize?
                # measure.visualize_TPM(normalize=normalize)
                


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

    def estimate_FCS(self, time_series=None):
        pass

    def estimate_dFCM(self, time_series=None):
        pass

    def visualize_states(self):
        pass

    def visualize_FCS(self, normalize=True, threshold=0.0, save_image=False, fig_name=None):
        
        if self.FCS == []:
            return

        if normalize:
            C = dFC_mat_normalize(C_t=self.FCS, threshold=threshold)
        else:
            C = self.FCS

        fig, axs = plt.subplots(1,C.shape[0], figsize=(25, 10), facecolor='w', edgecolor='k')
        fig.suptitle(self.measure_name+' FCS', fontsize=20, size=20)
        fig.subplots_adjust(hspace = .001, wspace=.2, top=1.5, bottom=0.1)
        axs = axs.ravel()

        for i, c in enumerate(C):
            axs[i].imshow(c, interpolation='nearest', aspect='equal', cmap='jet',\
                vmin=0, vmax=1)
            # axs[i].colorbar(shrink=0.8)
            axs[i].set_title('FCS '+str(i+1))

        # fig.tight_layout()
        # fig.subplots_adjust(top=1.4)

        if save_image:
            plt.savefig(fig_name + '.png', dpi=fig_dpi)  
            plt.close()
        else:
            plt.show()

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
        dFCM.add_FCP(FCPs=self.FCS_, FCP_idx=Z, subj_id_array=time_series.subj_id_array)

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
        dFCM.add_FCP(FCPs=self.FCS_, FCP_idx=Z, subj_id_array=time_series.subj_id_array)
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

        self.measure_name_ = 'Time-Frequency '
        self.is_state_based = False
        self.TPM = []
        self.FCS_ = []
        self.method_ = method
        self.coi_correction_ = coi_correction
        self.n_jobs = params['n_jobs']
        self.verbose = params['verbose']
        self.backend = params['backend']
    
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
            if self.n_jobs is None:
                Q = list()
                for j in range(time_series.n_regions):
                    Q.append(self.WT_dFC( \
                        Y1=time_series.data[i, :], \
                        Y2=time_series.data[j, :], \
                        Fs=time_series.Fs, \
                        J=J, s0=s0, dj=dj))
            else:
                Q = Parallel( \
                    n_jobs=self.n_jobs, verbose=self.verbose, backend=self.backend)( \
                    delayed(self.WT_dFC)( \
                                    Y1=time_series.data[i, :], \
                                    Y2=time_series.data[j, :], \
                                    Fs=time_series.Fs, \
                                    J=J, s0=s0, dj=dj) \
                                    for j in range(time_series.n_regions) \
                                                                )
            WT[:, i, :] = np.array(Q).T

        dFCM = DFCM(measure=self)
        dFCM.add_FCP(FCPs=WT, subj_id_array=time_series.subj_id_array)
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
            C.add_FCP(FCPs=self.FC( \
                        np.multiply(time_series, window)[ \
                            :,max(int(l-W/2),0):min(int(l+3*W/2),L) \
                                ] \
                                    ), \
                        subj_id_array = subj_id, \
                        TR_array=np.array( [ int(l + (l+W)) / 2 ] ) \
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

        dFCM = self.dFC(time_series=time_series.data, \
            subj_id=time_series.subj_id_array[:1], \
            W=self.W, \
            n_overlap=self.n_overlap, \
            tapered_window=self.tapered_window \
            )

        return dFCM


########################### Sliding_Window + Clustering ###########################

"""
- We used a tapered window as in Allen et al., created by convolving a rectangle (width = 22 TRs = 44s) 
  with a Gaussian (?? = 3 TRs) and slid in steps of 1 TR, resulting in W= 126 windows (Allen et al., 2014).
- Kmeans Clustering is repeated 500 times to escape local minima (Allen et al., 2014)
- for clustering, we have a 2-level kmeans clustering. First, we cluster FCPs of each subject. Then, we
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
    
        self.measure_name_ = 'SlidingWindow+Clustering'
        self.is_state_based = True
        self.clstr_distance = clstr_distance
        self.TPM = []
        self.FCS_ = []
        self.base_method_ = base_method
        self.n_states = params['n_states']
        self.n_subj_clstrs = params['n_subj_clstrs']
        self.W = params['W']
        self.n_overlap = params['n_overlap']
        self.n_jobs = params['n_jobs']
        self.verbose = params['verbose']
        self.backend = params['backend']
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
            params = {'n_jobs': self.n_jobs, 'verbose': self.verbose, 'backend': self.backend}
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
            params = {'n_jobs': self.n_jobs, 'verbose': self.verbose, 'backend': self.backend}
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
        dFCM.add_FCP(FCPs=self.FCS_, \
            FCP_idx=Z, \
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
        self.n_jobs = params['n_jobs']
        self.verbose = params['verbose']
        self.backend = params['backend']
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
            'n_jobs': self.n_jobs, 'verbose': self.verbose, 'backend': self.backend}
        self.swc = SLIDING_WINDOW_CLUSTR(base_method=self.base_method, \
            tapered_window=self.tapered_window, **params)
        self.swc.estimate_FCS(time_series=time_series)
        self.FCC_ = self.swc.estimate_dFCM(time_series=time_series)

        self.hmm_model = hmm.MultinomialHMM(n_components=self.n_hid_states)
        self.hmm_model.fit(self.FCC_.FCP_idx.reshape(-1, 1))

        self.Z = self.hmm_model.predict(self.FCC_.FCP_idx.reshape(-1, 1))
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

        Z = self.hmm_model.predict(FCC.FCP_idx.reshape(-1, 1))

        dFCM = DFCM(measure=self)
        dFCM.add_FCP(FCPs=self.FCS_, \
            FCP_idx=Z, \
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
                locs=None, nodes_info=None, TS_name=''):
        
        '''
        subj_id is an id to identify the subjects
        '''

        assert (not data is None) and (not Fs is None) and (not subj_id is None), \
            "data, subj_id, and Fs args must be provided."

        self.data_ = data
        self.subj_id_array_ = [subj_id] * data.shape[1] 
        self.Fs_ = Fs
        self.TS_name_ = TS_name
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
        nodes_lst=None, save_image=False, fig_name=None):

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
        plt.title(self.TS_name_)
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
    FCPs : Functional Connecitivity 
        Patterns
    FCP_idx : the  index of the 
        FCP that corresponds to each 
        timepoint
    

todo:
- 
"""

class DFCM():
    def __init__(self, measure=None):

        # assert not measure is None, \
        #     "measure arg must be provided."
        self.measure_ = measure
        self.FCPs_ = None 
        self.FCP_idx_ = None
        self.subj_id_array_ = None
        self.TR_array_ = None
        self.n_regions_ = None
        self.n_time_ = -1
    
    @classmethod
    def from_numpy(cls, array=None):
        pass

    # @property
    # def dFC_mat(self):
    #     return self.FCPs[self.FCP_idx,:,:]

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

    @property
    def FCPs(self):
        return self.FCPs_

    @property
    def FCP_idx(self):
        return self.FCP_idx_

    @property
    def subj_id_array(self):
        return self.subj_id_array_

    def get_dFC_mat(self, TRs=None):
        # get dFC matrices corresponding to 
        # the specified TRs

        if type(TRs) is np.int32 or type(TRs) is np.int64 or type(TRs) is int:
            TRs = [TRs]

        idxs = list()
        for tr in TRs:
            idxs.append(np.argwhere(self.TR_array==tr)[0,0])

        return self.FCPs[self.FCP_idx[idxs],:,:] 

    def concat(self, dFCM):

        # test this method

        assert type(dFCM) is DFCM, \
                "The input must be of DFCM class"

        if self.FCPs_ is None:
            self.FCPs_ = dFCM.FCPs
            self.FCP_idx_ = dFCM.FCP_idx
            self.n_regions_ = dFCM.n_regions
            self.n_time_ = dFCM.n_time
            self.TR_array_ = dFCM.TR_array
        else:
            assert self.n_regions== dFCM.n_regions, \
                "dFCM region numbers missmatch."
            FCP_idx = dFCM.FCP_idx + self.FCPs.shape[0]
            self.FCPs_ = np.concatenate((self.FCPs_, dFCM.FCPs), axis=0)
            self.FCP_idx_ = np.concatenate((self.FCP_idx_, FCP_idx), axis=0)
            self.n_time_ = self.FCP_idx.shape[0]
            self.TR_array_ = np.concatenate((self.TR_array, dFCM.TR_array))

    def add_FCP(self, FCPs, FCP_idx=None, subj_id_array=None, TR_array=None):
        
        if len(FCPs.shape)==2:
            FCPs = np.expand_dims(FCPs, axis=0)

        if FCP_idx is None:
            FCP_idx = np.arange(start=0, stop=FCPs.shape[0], step=1)

        if type(FCP_idx) is list:
            FCP_idx = np.array(FCP_idx)

        if len(FCP_idx.shape)>1:
            FCP_idx = np.squeeze(FCP_idx)

        if not type(subj_id_array) is list:
            subj_id_array = list(subj_id_array)
        
        assert FCPs.shape[1] == FCPs.shape[2], \
                "FC matrices must be square."

        assert len(subj_id_array)==FCP_idx.shape[0], \
            "FCP_idx and subj_id_array length mismatch."

        if TR_array is None:
            TR_array = np.arange(start=self.n_time+1, stop=self.n_time+len(FCP_idx)+1, step=1)

        if self.FCPs_ is None:
            self.FCPs_ = FCPs
            self.FCP_idx_ = FCP_idx
            self.subj_id_array_ = subj_id_array
            self.n_regions_ = self.FCPs.shape[1]
            self.n_time_ = self.FCP_idx.shape[0]
            self.TR_array_ = TR_array
        else:
            # test this part
            assert self.n_regions == FCPs.shape[1], \
                "FCP region numbers mismatch."
            FCP_idx = FCP_idx + self.FCPs.shape[0]
            self.FCPs_ = np.concatenate((self.FCPs_, FCPs), axis=0)
            self.FCP_idx_ = np.concatenate((self.FCP_idx_, FCP_idx), axis=0)
            self.subj_id_array_ = self.subj_id_array_ + subj_id_array
            self.n_time_ = self.FCP_idx.shape[0]
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

        C = np.abs(C) # ?????? should we do this?

        if np.any(C<0):
            V_MIN = -1
            V_MAX = 1
        else:
            V_MIN = 0
            V_MAX = 1

        if not fix_lim:
            V_MAX = np.max(C)
            V_MIN = np.min(C)

        fig, axs = plt.subplots(1, C.shape[0], figsize=(25, 10), \
            facecolor='w', edgecolor='k')
        fig.suptitle(self.measure.measure_name+' dFC', fontsize=20, size=20)
        axs = axs.ravel()

        for l in range(0, C.shape[0]):
            axs[l].set_axis_off()
            im = axs[l].imshow(C[l, :, :], interpolation='nearest', aspect='equal', cmap='jet',    # 'viridis'
                        vmin=V_MIN, vmax=V_MAX)
            axs[l].set_title('TR '+str(TRs[l]))

        fig.subplots_adjust(bottom=0.1, top=1.5, left=0.1, right=0.9,
                            wspace=0.02, hspace=0.02)

        # [x, y, w, h]
        cb_ax = fig.add_axes([0.91, 0.75, 0.007, 0.1])
        cbar = fig.colorbar(im, cax=cb_ax)

        #set the colorbar ticks and tick labels
        cbar.set_ticks(np.arange(0, 1.1, 0.5))
        cbar.set_ticklabels(['0', '0.5', '1'])
        
        if save_image:
            plt.savefig(fig_name + '.png', dpi=fig_dpi)  
            plt.close()
        else:
            plt.show()


