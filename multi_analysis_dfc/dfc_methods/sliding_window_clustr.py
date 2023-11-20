"""
Implementation of dFC methods.

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import numpy as np
import time
from sklearn.cluster import KMeans
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric

from .base_dfc_method import BaseDFCMethod
from ..time_series import TIME_SERIES
from ..dfc import DFC
from .sliding_window import SLIDING_WINDOW
from .time_freq import TIME_FREQ
from ..dfc_utils import dFC_mat2vec, dFC_vec2mat

########################### Sliding_Window + Clustering ############################

"""
- We used a tapered window as in Allen et al., created by convolving a rectangle (width = 22 TRs = 44s) 
  with a Gaussian (σ = 3 TRs) and slid in steps of 1 TR, resulting in W= 126 windows (Allen et al., 2014).
- Kmeans Clustering is repeated 500 times to escape local minima (Allen et al., 2014)
- for clustering, we have a 2-level kmeans clustering. First, we cluster FCSs of each subject. Then, we
    cluster all clustering centers from all subjects. the final estimate_dFC is using the second kmeans
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

class SLIDING_WINDOW_CLUSTR(BaseDFCMethod):

    def __init__(self, clstr_distance='euclidean', **params):

        assert clstr_distance=='euclidean' or clstr_distance=='manhattan', \
            "Clustering distance not recognized. It must be either \
                euclidean or manhattan."
        self.logs_ = ''
        self.TPM = []
        self.FCS_ = []
        self.mean_act = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None

        self.params_name_lst = ['measure_name', 'is_state_based', 'clstr_base_measure', 'sw_method', 'tapered_window',
            'clstr_distance', 'coi_correction',
            'n_subj_clstrs', 'W', 'n_overlap', 'n_states', 'normalization',
            'n_jobs', 'verbose', 'backend',
            'num_subj', 'num_select_nodes', 'num_time_point', 'Fs_ratio',
            'noise_ratio', 'num_realization', 'session']
        self.params = {}
        for params_name in self.params_name_lst:
            if params_name in params:
                self.params[params_name] = params[params_name]
            else:
                self.params[params_name] = None
        
        self.params['measure_name'] = 'Clustering'
        self.params['is_state_based'] = True
        self.params['clstr_distance'] = clstr_distance

        assert self.params['clstr_base_measure'] in self.base_methods_name_lst, \
            "Base method not recognized."

    @property
    def measure_name(self):
        return self.params['measure_name'] #+ '_' + self.base_method

    def dFC_mat2vec(self, C_t):
        return dFC_mat2vec(C_t)
        # if len(C_t.shape)==2:
        #     assert C_t.shape[0]==C_t.shape[1],\
        #         'C is not a square matrix'
        #     return C_t[np.triu_indices(C_t.shape[1], k=0)]

        # F = list()
        # for t in range(C_t.shape[0]):
        #     C = C_t[t, : , :]
        #     assert C.shape[0]==C.shape[1],\
        #         'C is not a square matrix'
        #     F.append(C[np.triu_indices(C_t.shape[1], k=0)])

        # F = np.array(F)
        # return F

    def dFC_vec2mat(self, F, N):
        return dFC_vec2mat(F=F, N=N)
        # C = list()
        # iu = np.triu_indices(N, k=0)
        # for i in range(F.shape[0]):
        #     K = np.zeros((N, N))
        #     K[iu] = F[i,:]
        #     K = K + np.multiply(K.T, 1-np.eye(N))
        #     C.append(K)
        # C = np.array(C)
        # return C

    def clusters_lst2idx(self, clusters):
        Z = np.zeros((self.F.shape[0],))
        for i, cluster in enumerate(clusters):
            for sample in cluster:
                Z[sample] = i
        return Z.astype(int)

    def cluster_FC(self, FCS_raw, n_clusters, n_regions):

        F = self.dFC_mat2vec(FCS_raw)

        if self.params['clstr_distance']=='manhattan':
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
            F_cent = kmeans_.cluster_centers_

        FCS_ = self.dFC_vec2mat(F_cent, N=n_regions)
        return FCS_, kmeans_

        
    def estimate_FCS(self, time_series):

        assert type(time_series) is TIME_SERIES, \
            "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4FCS(time_series)

        # start timing
        tic = time.time()

        base_dFC = None
        if self.params['clstr_base_measure']=='Time-Freq':
            base_dFC = TIME_FREQ(**self.params)
        if self.params['clstr_base_measure']=='SlidingWindow':
            base_dFC = SLIDING_WINDOW(**self.params)

        # 1-level clustering
        # dFC_raw = base_dFC.estimate_dFC( \
        #         time_series=time_series \
        #         )
        # self.FCS_, self.kmeans_ = self.cluster_FC( \
        # dFC_raw.get_dFC_mat(TRs=self.dFC_raw.TR_array), \
        # n_regions = dFC_raw.n_regions \
        # )

        # 2-level clustering
        SUBJECTs = time_series.subj_id_lst
        FCS_1st_level = None
        SW_dFC = None
        for subject in SUBJECTs:
            
            dFC_raw = base_dFC.estimate_dFC( \
                time_series=time_series.get_subj_ts(subjs_id=subject) \
                )

            # test
            if dFC_raw.n_time<self.params['n_subj_clstrs']:
                print( \
                    'Number of subject-level clusters cannot be more than SW dFC samples! n_subj_clstrs was changed to ' \
                        + str(dFC_raw.n_time) + '. This change will cause problems in similarity assessment.')
                self.params['n_subj_clstrs'] = dFC_raw.n_time

            FCS, _ = self.cluster_FC( \
                FCS_raw = dFC_raw.get_dFC_mat(TRs=dFC_raw.TR_array), \
                n_clusters = self.params['n_subj_clstrs'], \
                n_regions = dFC_raw.n_regions \
                )
            
            if SW_dFC is None:
                SW_dFC = dFC_raw.get_dFC_mat(TRs=dFC_raw.TR_array)
            else:
                SW_dFC = np.concatenate((SW_dFC, dFC_raw.get_dFC_mat(TRs=dFC_raw.TR_array)), axis=0)
            if FCS_1st_level is None:
                FCS_1st_level = FCS
            else:
                FCS_1st_level = np.concatenate((FCS_1st_level, FCS), axis=0)
        
        self.FCS_, self.kmeans_ = self.cluster_FC( \
            FCS_raw=FCS_1st_level, \
            n_clusters = self.params['n_states'], \
            n_regions = dFC_raw.n_regions \
            )
        self.Z = self.kmeans_.predict(self.dFC_mat2vec(SW_dFC))

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

        base_dFC = None
        if self.params['clstr_base_measure']=='Time-Freq':
            base_dFC = TIME_FREQ(**self.params)
        if self.params['clstr_base_measure']=='SlidingWindow':
            base_dFC = SLIDING_WINDOW(**self.params)
                    
        dFC_raw = base_dFC.estimate_dFC(time_series=time_series)

        F = self.dFC_mat2vec(dFC_raw.get_dFC_mat(TRs=dFC_raw.TR_array))

        if self.params['clstr_distance']=='manhattan':
            pass
            # ########### Manhattan Clustering ##############
            # self.kmeans_.predict(F)
            # Z = self.clusters_lst2idx(self.kmeans_.get_clusters())
        else:
            ########### Euclidean Clustering ##############
            Z = self.kmeans_.predict(F)

        # record time
        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=self.FCS_, \
            FCS_idx=Z, \
            TS_info=time_series.info_dict, \
            TR_array=dFC_raw.TR_array \
            )

        return dFC


################################################################################