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

################################# Parameters ####################################

fig_dpi = 120

################################# Other Functions ####################################

def intersection(lst1, lst2): # input is a list 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def TR_intersection(measures_lst): # input is a list of dFC Measure objs
    TRs_lst_old = measures_lst[0].dFCM.TR_array
    for measure in measures_lst:
        TRs_lst_new = intersection(TRs_lst_old, measure.dFCM.TR_array)
        TRs_lst_old = TRs_lst_new
    TRs_lst_old.sort()
    return TRs_lst_old

################################# dFC class ####################################

"""

todo:
- brain or brain graph class
- add an updating behavior -> we can segment subjects and time_series and update the model gradually ?
- type annotation
- remove sliding window type dFC visualization 
- normalization: C_t_z[:, i, i] = np.mean(C_t_z) # ?????????????????
"""

class dFC:

    def __init__(self):
        self.measure_name = ''
        self.dFCM = DFCM()
        self._stat = []
        self.TPM = []

    def calc(self):
        pass

    def visualize_states(self):
        pass

    def visualize_dFC(self, TRs=None, W=1, n_overlap=1, normalize=True, threshold=0.0, save_image=False, fig_name=None):

        # W = 1 and n_overlap = 1 -> normal dFC visualization

        if TRs is None:
            TRs = list(range(self.dFCM.n_time))

        if normalize:
            C = self.dFC_mat_normalize(C_t=self.dFCM.slice_TR(TRs=TRs), \
                global_normalization=True, threshold=threshold)
        else:
            C = self.dFCM.slice_TR(TRs=TRs)

        L = C.shape[0]
        step = int((1-n_overlap)*W)
        if step == 0:
            step = 1

        fig, axs = plt.subplots(1,int((L-W)/step)+1, figsize=(25, 10), \
            facecolor='w', edgecolor='k')
        fig.suptitle(self.measure_name+' dFC', fontsize=20, size=20)
        axs = axs.ravel()

        i=0
        for l in range(0, L-W+1, step):
            idx = int((l + (l+W))/2)
            axs[i].set_axis_off()
            im = axs[i].imshow(C[idx, :, :], interpolation='nearest', aspect='equal', cmap='jet',    # 'viridis'
                        vmin=0, vmax=1)
            axs[i].set_title('TR '+str(TRs[i]))
            i = i + 1

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

    def visualize_FCS(self, normalize=True, threshold=0.0, save_image=False, fig_name=None):
        
        if self.FCS_ == []:
            return

        if normalize:
            C = self.dFC_mat_normalize(C_t=self.FCS_, threshold=threshold)
        else:
            C = self.FCS_

        fig, axs = plt.subplots(1,C.shape[0], figsize=(25, 10), facecolor='w', edgecolor='k')
        fig.suptitle(self.measure_name+' FCS', fontsize=20, size=20)
        fig.subplots_adjust(hspace = .001, wspace=.2)
        axs = axs.ravel()

        for i, c in enumerate(C):
            axs[i].imshow(c, interpolation='nearest', aspect='equal', cmap='jet')
            # axs[i].colorbar(shrink=0.8)
            axs[i].set_title('FCS '+str(i+1))

        fig.tight_layout()
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
            C = self.dFC_mat_normalize(C_t=np.expand_dims(self.TPM, axis=0), threshold=0.0)
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

    

    def dFC_mat_normalize(self, C_t, global_normalization=True, threshold=0.0):

        C_t_z = deepcopy(C_t)
        if len(C_t_z.shape)<3:
            C_t_z = np.expand_dims(C_t_z, axis=0)

        if global_normalization:

            # transform the whole dFC mat to [0, 1]
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

            C_t_z = np.divide(C_t_z, np.max(maX))

        else:

            # transform each time slice to [0, 1]
            for i in range(C_t_z.shape[0]):
                slice = C_t_z[i,:,:]
                slice_non_diag = slice[np.where(~np.eye(slice.shape[0],dtype=bool))]
                slice = slice - np.min(slice_non_diag)
                slice_non_diag = slice[np.where(~np.eye(slice.shape[0],dtype=bool))]
                slice = np.divide(slice, np.max(slice_non_diag))
                slice = slice * (slice>=threshold)

                C_t_z[i,:,:] = slice

        # removing self connections
        for i in range(C_t_z.shape[1]):
            C_t_z[:, i, i] = np.mean(C_t_z) # ?????????????????

        return C_t_z



################################# HMM Continuous ###############################

"""
by hmmlearn
todo:
- number of iter?
- ValueError: 'covars' must be symmetric, positive-definite
"""

from hmmlearn import hmm

class HMM_CONT(dFC):

    def __init__(self, n_states=12):
        self.measure_name = 'Continuous HMM'
        self.dFCM = DFCM()
        self.TPM = []
        self.FCS_ = []
        self.n_states = n_states

    def calc(self, time_series=None):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        self.n_regions = time_series.n_regions
        self.n_time = time_series.n_time

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

        self.dFCM.add_FCM(self.FCS_[self.Z,:,:])
        return

################################## Windowless ##################################

"""
by : https://github.com/nel215/ksvd

Reference: Rubinstein, R., Zibulevsky, M. and Elad, M., Efficient Implementation 
of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit Technical 
Report - CS Technion, April 2008

todo:
"""
from ksvd import ApproximateKSVD

class WINDOWLESS(dFC):

    def __init__(self, n_states=12):
        self.measure_name = 'Windowless'
        self.dFCM = DFCM()
        self.TPM = []
        self.FCS_ = []
        self.n_states = n_states
    
    def calc(self, time_series=None):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        self.n_regions = time_series.n_regions
        self.n_time = time_series.n_time

        # time_series ~ gamma.dot(dictionary)
        aksvd = ApproximateKSVD(n_components=self.n_states, transform_n_nonzero_coefs=1)
        self.dictionary = aksvd.fit(time_series.data.T).components_
        self.gamma = aksvd.transform(time_series.data.T)

        self.FCS_ = np.zeros([self.n_states, self.n_regions, self.n_regions])
        for i in range(self.n_states):
            self.FCS_[i, :, :] = np.multiply(np.expand_dims(self.dictionary[i,:], axis=0).T, np.expand_dims(self.dictionary[i,:], axis=0))

        self.Z = list()
        for i in range(self.n_time):
            self.Z.append(np.argwhere(self.gamma[i, :] != 0)[0,0])
        self.dFCM.add_FCM(self.FCS_[self.Z,:,:])
        return 


################################# Sliding-Window #################################

"""
todo:
- switch between corr and MI
- dFC_mat normalization ? 
_ the problem with corr
"""

class SLIDING_WINDOW(dFC):

    def __init__(self, method=None, W=88, n_overlap=0.5, tapered_window=True):

        if method is None:
            method='pear_corr'
        assert method=='pear_corr' or method=='MI', \
            "method not recognized. It must be either pear_corr \
                or MI."

        self.measure_name_ = 'Sliding Window'
        self.method_ = method
        self.dFCM = DFCM()
        self.TPM = []
        self.FCS_ = []
        self.W = W
        self.n_overlap = n_overlap
        self.tapered_window = tapered_window
    
    @property
    def measure_name(self):
        return self.measure_name_ + '_' + self.method_
        
    @property
    def method(self):
        return self.method_

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
    
        C = np.zeros((time_series.shape[0], time_series.shape[0]))
        for i in range(time_series.shape[0]):
            for j in range(i, time_series.shape[0]):
                
                X = time_series[i, :]
                Y = time_series[j, :]

                if self.method=='MI':
                    C[j, i] = self.calc_MI(X, Y)
                else:
                    C[j, i] = np.corrcoef(X, Y)[0, 1]

                C[i, j] = C[j, i]   
                
        return C

    def dFC(self, time_series, W=None, n_overlap=None, tapered_window=False):
        L = time_series.shape[1]
        step = int((1-n_overlap)*W)
        if step == 0:
            step = 1

        window_taper = signal.windows.gaussian(W, std=3*W/22)
        C = DFCM()
        for l in range(0, L-W+1, step):

            # creating a rectangel window
            window = np.zeros((time_series.shape[1]))
            window[l:l+W] = 1
            
            # tapering the window
            if tapered_window:
                window = signal.convolve(window, window_taper, mode='same') / sum(window_taper)

            window = np.repeat(np.expand_dims(window, axis=0), time_series.shape[0], axis=0)

            C.add_FCM(self.FC(np.multiply(time_series, window)), TR_array=np.array( [ int(l + (l+W)) / 2 ] ) )
            # print('dFC step = %d' %(l))

        return C
    
    def calc(self, time_series=None):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        self.n_regions = time_series.n_regions
        self.n_time = time_series.n_time

        self.dFCM = self.dFC(time_series=time_series.data, W=self.W, n_overlap=self.n_overlap, tapered_window=self.tapered_window)

        # self.dFC_mat = self.dFC_mat_normalize(self.dFC_mat)

        return 

    
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

todo:

- consider COI and edge effect in averaging:
    => should we truncate the time points having at less than 20 freqs as done in Savva et al. ?

"""
import pycwt as wavelet

class TIME_FREQ(dFC):

    def __init__(self, method=None, coi_correction=True):
        
        if method is None:
            method='WTC'
        assert method=='CWT_mag' or method=='CWT_phase_r' \
            or method=='CWT_phase_a' or method=='WTC', \
            "method not recognized. It must be either CWT_mag, \
                CWT_phase_r, CWT_phase_a, or WTC."

        self.measure_name_ = 'Time-Frequency '
        self.dFCM = DFCM()
        self.TPM = []
        self.FCS_ = []
        self.method_ = method
        self.coi_correction_ = coi_correction
    
    @property
    def coi_correction(self):
        return self.coi_correction_
    @property
    def measure_name(self):
        return self.measure_name_ + '_' + self.method_

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

    def calc(self, time_series=None):
        
        # params
        J = 50 # -1
        s0 = 1 # -1
        dj = 1/8 # 1/12

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        self.n_regions = time_series.n_regions
        self.n_time = time_series.n_time

        WT = np.zeros((self.n_time, self.n_regions, self.n_regions))
        for i in range(self.n_regions):
            for j in range(self.n_regions):

                Y1 = time_series.data[i, :]
                Y2 = time_series.data[j, :]

                if self.method_=='CWT_mag' or self.method_=='CWT_phase_r' or self.method_=='CWT_phase_a':
                    # Cross Wavelet Transform
                    WT_xy, coi, freqs, _ = wavelet.xwt(Y1, Y2, dt=1/time_series.Fs, dj=dj, s0=s0, J=J, 
                        significance_level=0.95, wavelet='morlet', normalize=True)

                    if self.method_=='CWT_mag':
                        WT_xy_corrected = self.coi_correct(WT_xy, coi, freqs)
                        WT[:, i, j] = np.abs(np.mean(WT_xy_corrected, axis=0))

                    if self.method_=='CWT_phase_r' or self.method_=='CWT_phase_a':
                        cosA = np.cos(np.angle(WT_xy))
                        sinA = np.sin(np.angle(WT_xy))

                        cosA_corrected = self.coi_correct(cosA, coi, freqs)
                        sinA_corrected = self.coi_correct(sinA, coi, freqs)

                        A = (cosA_corrected + sinA_corrected * 1j)

                        if self.method_=='CWT_phase_r':
                            WT[:, i, j] = np.abs(np.mean(A, axis=0))
                        else:
                            WT[:, i, j] = np.angle(np.mean(A, axis=0))

                if self.method_=='WTC':
                    # Wavelet Transform Coherence
                    WT_xy, _, coi, freqs, _ = wavelet.wct(Y1, Y2, dt=1/time_series.Fs, dj=dj, s0=s0, J=J, 
                        sig=False, significance_level=0.95, wavelet='morlet', normalize=True)
                    WT_xy_corrected = self.coi_correct(WT_xy, coi, freqs)
                    WT[:, i, j] = np.abs(np.mean(WT_xy_corrected, axis=0))

                

        self.dFCM.add_FCM(FCM=WT)
        return 

########################### Sliding_Window + Clustering ###########################

"""
- We used a tapered window as in Allen et al., created by convolving a rectangle (width = 22 TRs = 44s) 
  with a Gaussian (σ = 3 TRs) and slid in steps of 1 TR, resulting in W= 126 windows.
- can use the results from SW

todo:

"""
from sklearn.cluster import KMeans

class SLIDING_WINDOW_CLUSTR(dFC):

    def __init__(self, sliding_window=None, n_states=12, W=88, n_overlap=0.5, tapered_window=True):
        self.measure_name = 'Sliding Window + Clustering'
        self.dFCM = DFCM()
        self.TPM = []
        self.FCS_ = []
        self.sliding_window = sliding_window
        self.n_states = n_states
        self.W = W
        self.n_overlap = n_overlap
        self.tapered_window = tapered_window
    
    def set_sliding_window(self, sliding_window=None):
        self.sliding_window = sliding_window

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
    
    def calc(self, time_series=None):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        self.n_regions = time_series.n_regions
        self.n_time = time_series.n_time

        if self.sliding_window is None:
            self.sliding_window = SLIDING_WINDOW(W=self.W, n_overlap=self.n_overlap, tapered_window=self.tapered_window)
            self.sliding_window.calc(time_series=time_series)
        
        self.dFCM_raw = self.sliding_window.dFCM

        self.F = self.dFC_mat2vec(self.dFCM_raw.dFC_mat)

        self.kmeans_ = KMeans(n_clusters=self.n_states).fit(self.F)

        self.Z = self.kmeans_.predict(self.F)
        self.F_cent = self.kmeans_.cluster_centers_

        self.FCS_ = self.dFC_vec2mat(self.F_cent, N=self.n_regions)
        self.dFCM.add_FCM(self.FCS_[self.Z,:,:], TR_array=self.dFCM_raw.TR_array)

        return 

################################# HMM Discrete #################################

"""
- Z is state time course
- can use the results from SWC

todo:
- two-level hierarchical clustering ?
- add slice method to DFCM for III
- find a better name for FCC
"""
# from HMM_discrete import *
from hmmlearn import hmm

class HMM_DISC(dFC):

    def __init__(self, swc=None, n_states=12, n_hid_states=6, W=88, n_overlap=0.5, tapered_window=True):
        self.measure_name = 'Discrete HMM'
        self.dFCM = DFCM()
        self.TPM = []
        self.FCS_ = []
        self.swc = swc
        self.n_states = n_states
        self.n_hid_states = n_hid_states
        self.W = W
        self.n_overlap = n_overlap
        self.tapered_window = tapered_window

    def set_swc(self, swc=None):
        self.swc = swc

    def calc(self, time_series=None):
        
        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        self.n_regions = time_series.n_regions
        self.n_time = time_series.n_time

        if self.swc is None:
            self.swc = SLIDING_WINDOW_CLUSTR(n_states=self.n_states, W=self.W, n_overlap=self.n_overlap, tapered_window=self.tapered_window)
            self.swc.calc(time_series=time_series)
        
        self.FCC_ = self.swc.dFCM

        model = hmm.MultinomialHMM(n_components=self.n_hid_states)
        model.fit(self.swc.Z.reshape(-1, 1))

        self.Z = model.predict(self.swc.Z.reshape(-1, 1))
        self.TPM = model.transmat_
        self.EPM = model.emissionprob_ 

        self.FCS_ = np.zeros((self.n_hid_states, self.n_regions, self.n_regions))
        for i in range(self.n_hid_states):
            self.FCS_[i,:,:] = np.mean(self.FCC_.dFC_mat[np.squeeze(np.argwhere(self.Z==i)),:,:], axis=0)  # III

        self.dFCM.add_FCM(self.FCS_[self.Z,:,:], TR_array=self.FCC_.TR_array)

        return 
    
###################################################################################

################################# TIME_SERIES class ######################################

"""

todo:
- select nodes for visualizer
"""

class TIME_SERIES():
    def __init__(self, data=None, Fs=None, time_array=None, locs=None, nodes_info=None, TS_name=''):
        self.data_ = data
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
    def nodes_lst(self):
        # output shape is (n_region, 1) 
        return np.array(self.nodes_selection_)[:, np.newaxis]
        
    @property
    def interval(self):
        # output shape is (1, n_time) 
        return np.array(self.interval_)[np.newaxis, :]

    @property
    def locs(self):
        return self.locs_[self.nodes_lst, :]

    @property
    def nodes_info(self):
        return self.nodes_info_[self.nodes_lst]

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
        return self.time_array_[self.interval]

    @property
    def TS_name(self):
        return self.TS_name_

    def resample(self):
        # change self.Fs_
        pass

    def append_ts(self, new_time_series=None):
        # append new time series to existing ones
        # truncate and node selection are not considered; the whole old time series will be concat to new one
        # append_ts resets the truncate and node selection

        assert self.n_regions_ == new_time_series.shape[0], \
            "Number of nodes mismatch."

        self.data_ = np.concatenate((self.data_, new_time_series), axis=1)
        self.n_time_ = self.data_.shape[1]
        self.interval_ = list(range(self.n_time_))


    def truncate(self, start_time=None, end_time=None, start_point=None, end_point=None):

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
        
        self.interval_ = list(range(start, end))
                
    def select_nodes(self, nodes_idx=None):
        # select the nodes indexed by numbers in nodes_idx. nodes_idx is a numpy 1D array
        # if nodes_idx is None -> all the nodes will be considered (resets node selection)
        # if nodes_idx is not sorted, it can be used to reorder the nodes

        if nodes_idx is None:
            self.nodes_selection_ = list(range(self.n_regions_))
        else:
            self.nodes_selection_ = nodes_idx
            

        

    def visualize(self, interval=None, save_image=False, fig_name=None):

        # interval or start+end ?

        plt.figure(figsize=(15, 5))
        plt.plot(self.data_.T[interval,:])
        plt.title(self.TS_name_)
        if save_image:
            plt.savefig(fig_name + '.png', dpi=fig_dpi)  
            plt.close()
        else:
            plt.show()


################################# dFCM class ######################################

"""

todo:
- 
"""

class DFCM():
    def __init__(self, dFC_mat=None, TR_array=None):

        if dFC_mat is None:
            self.dFC_mat_ = None
            self.TR_array_ = None
            self.n_regions_ = None
            self.n_time_ = 0
        else:
            assert dFC_mat.shape[1] == dFC_mat.shape[2], \
                "FC matrices must be square."

            self.dFC_mat_ = dFC_mat
            self.n_regions_ = self.dFC_mat_.shape[1]
            self.n_time_ = self.dFC_mat_.shape[0]
            if TR_array is None:
                self.TR_array_ = np.arange(start=1, stop=self.n_time+1, step=1)
            else:
                self.TR_array_ = TR_array
    
    @classmethod
    def from_numpy(cls, array=None):
        pass

    @property
    def dFC_mat(self):
        return self.dFC_mat_

    @property
    def TR_array(self):
        return self.TR_array_.astype(int)

    @property
    def n_regions(self):
        return self.n_regions_

    @property
    def n_time(self):
        return self.n_time_

    def slice_TR(self, TRs=None):
        idxs = list()
        for tr in TRs:
            idxs.append(np.argwhere(self.TR_array==tr)[0,0])

        return self.dFC_mat[idxs, :, :]

    def concat(self, dFCM):

        assert dFCM is DFCM, \
                "The input must be of DFCM class"

        if self.dFC_mat_ is None:
            self.dFC_mat_=dFCM.dFC_mat
            self.n_regions_ = dFCM.n_regions
            self.n_time_ = dFCM.n_time
        else:
            assert self.dFC_mat_.shape[1] == dFCM.dFC_mat.shape[1], \
                "dFCM region numbers missmatch."
            self.dFC_mat_ = np.concatenate((self.dFC_mat_, dFCM.dFC_mat), axis=0)
            self.n_time_ += dFCM.n_time

    def add_FCM(self, FCM, TR_array=None):
        
        if len(FCM.shape)==2:
            FCM = np.expand_dims(FCM, axis=0)
        
        assert FCM.shape[1] == FCM.shape[2], \
                "FC matrices must be square."

        if TR_array is None:
            TR_array = np.arange(start=self.n_time+1, stop=self.n_time+FCM.shape[0]+1, step=1)

        if self.dFC_mat_ is None:
            self.dFC_mat_ = FCM
            self.n_regions_ = self.dFC_mat_.shape[1]
            self.n_time_ = self.dFC_mat_.shape[0]
            self.TR_array_ = TR_array
        else:
            assert self.dFC_mat_.shape[1] == FCM.shape[1], \
                "FCM region numbers missmatch."
            self.dFC_mat_ = np.concatenate((self.dFC_mat_, FCM), axis=0)
            self.n_time_ += FCM.shape[0]
            self.TR_array_ = np.concatenate((self.TR_array, TR_array))

    
    def expand():
        pass

    def zip():
        pass

