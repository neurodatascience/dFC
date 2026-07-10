# -*- coding: utf-8 -*-
"""
SimilarityAssessment class
functions to assess similarity between dFC results

Created on Jun 29 2023
@author: Mohammad Torabi
"""

from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed
from scipy import stats

from ..dfc_utils import (
    TR_intersection,
    calc_graph_propoerty,
    dFC_mat2vec,
    filter_dFC_lst,
    find_new_order,
    mutual_information,
    normalized_euc_dist,
)

################################# SimilarityAssessment class ####################################


class SimilarityAssessment:

    def __init__(self, dFC_lst):
        self.dFC_lst = dFC_lst

    ##################### dFC FEATURES ######################

    def FO_calc(self, dFC_lst, common_TRs=None):

        # returns, for each state the Fractional Occupancy (FO)
        # see Vidaurre et al., 2017
        # it only considers TRs in common_TRs

        if common_TRs is None:
            common_TRs = TR_intersection(dFC_lst)

        FO_list = list()
        for dFC in dFC_lst:

            FO = {}

            if dFC.measure.is_state_based:

                state_act_dict = dFC.state_act_dict(TRs=common_TRs)

                for FCS_key in state_act_dict["state_TC"]:
                    FO[FCS_key] = np.mean(state_act_dict["state_TC"][FCS_key]["act_TC"])

            FO_list.append(FO)

        return FO_list

    def transition_stats(self, dFC_lst, common_TRs=None):
        # returns the number of total state transition within common_TRs -> trans_freq
        # and the number of total state transitions regardless of common_TRs
        # but normalized by total number of TRs -> trans_norm
        # and a list of all dwell times

        if common_TRs is None:
            common_TRs = TR_intersection(dFC_lst)

        TRs_lst = list()
        for TR in common_TRs:
            TRs_lst.append("TR" + str(TR))

        output_lst = list()
        for dFC in dFC_lst:

            output_dict = {}

            if dFC.measure.is_state_based:

                #  downsampled
                trans_freq = 0
                dwell_time_lst = list()
                dwell_time = 0
                last_TR = None
                for TR in dFC.FCS_idx:
                    if TR in TRs_lst:
                        if not last_TR is None:
                            if dFC.FCS_idx[TR] != dFC.FCS_idx[last_TR]:
                                dwell_time_lst.append(dwell_time)
                                dwell_time = 0
                                trans_freq += 1
                        dwell_time += 1
                        last_TR = TR

                output_dict["dwell_time"] = dwell_time_lst
                output_dict["trans_freq"] = trans_freq

                # normalized (not downsampled)
                trans_norm = 0
                dwell_time_lst = list()
                dwell_time = 0
                last_TR = None
                for TR in dFC.FCS_idx:
                    if not last_TR is None:
                        if dFC.FCS_idx[TR] != dFC.FCS_idx[last_TR]:
                            dwell_time_lst.append(dwell_time / len(dFC.FCS_idx))
                            dwell_time = 0
                            trans_norm += 1
                    dwell_time += 1
                    last_TR = TR
                trans_norm = trans_norm / len(dFC.FCS_idx)

                output_dict["dwell_time_norm"] = dwell_time_lst
                output_dict["trans_norm"] = trans_norm

            output_lst.append(output_dict)

        return output_lst

    def feature_all(self, dFC_mat):
        vectorized_dFC = dFC_mat2vec(dFC_mat).flatten()  # (time*connection, )
        vectorized_dFC = np.expand_dims(vectorized_dFC, axis=0)  # (1, time*connection)
        return vectorized_dFC

    def feature_spatial(self, dFC_mat):
        conn_over_time = dFC_mat2vec(dFC_mat)  # (time, connection)
        return conn_over_time

    def feature_temporal(self, dFC_mat):
        conn_over_time = dFC_mat2vec(dFC_mat)  # (time, connection)
        time_over_conn = conn_over_time.T  # (connection, time)
        return time_over_conn

    def feature_inter_time_corr(self, dFC_mat):
        """
        returns correspondence of inter-time relation between results of dFC
        measures in each subject
        """
        conn_over_time = dFC_mat2vec(dFC_mat)  # (time, connection)
        inter_time_corr = np.corrcoef(conn_over_time)  # (time, time)
        inter_time_corr = np.nan_to_num(inter_time_corr)
        inter_time_corr = dFC_mat2vec(inter_time_corr)  # (time*(time-1)/2, )
        inter_time_corr = np.expand_dims(inter_time_corr, axis=0)  # (1, time*(time-1)/2)
        return inter_time_corr

    def feature_inter_conn_corr(self, dFC_mat):
        conn_over_time = dFC_mat2vec(dFC_mat)  # (time, connection)
        time_over_conn = conn_over_time.T  # (connection, time)
        inter_conn_corr = np.corrcoef(time_over_conn)  # (connection, connection)
        inter_conn_corr = np.nan_to_num(inter_conn_corr)
        inter_conn_corr = dFC_mat2vec(inter_conn_corr)  # (connection*(connection-1)/2, )
        inter_conn_corr = np.expand_dims(
            inter_conn_corr, axis=0
        )  # (1, connection*(connection-1)/2)
        return inter_conn_corr

    def feature_dFC_avg(self, dFC_mat):
        dFC_avg = np.mean(dFC_mat, axis=0)  # (ROI, ROI)
        vectorized_dFC_avg = dFC_mat2vec(dFC_avg)  # (connection, )
        vectorized_dFC_avg = np.expand_dims(vectorized_dFC_avg, axis=0)  # (1, connection)
        return vectorized_dFC_avg

    def feature_dFC_var(self, dFC_mat):
        dFC_var = np.var(dFC_mat, axis=0)  # (ROI, ROI)
        vectorized_dFC_var = dFC_mat2vec(dFC_var)  # (connection, )
        vectorized_dFC_var = np.expand_dims(vectorized_dFC_var, axis=0)  # (1, connection)
        return vectorized_dFC_var

    def feature_graph_spatial(self, dFC_mat, graph_property):
        graph_feature_over_time = list()
        for FC_mat in dFC_mat:
            graph_feature = calc_graph_propoerty(
                FC_mat, property=graph_property, threshold=False, binarize=False
            )
            graph_feature_over_time.append(graph_feature)
        graph_feature_over_time = np.array(graph_feature_over_time)  # (time, ROI)
        return graph_feature_over_time

    def feature_graph_temporal(self, dFC_mat, graph_property):
        graph_feature_over_time = list()
        for FC_mat in dFC_mat:
            graph_feature = calc_graph_propoerty(
                FC_mat, property=graph_property, threshold=False, binarize=False
            )
            graph_feature_over_time.append(graph_feature)
        graph_feature_over_time = np.array(graph_feature_over_time)  # (time, ROI)
        graph_feature_over_node = graph_feature_over_time.T  # (ROI, time)
        graph_feature_avg = np.mean(graph_feature_over_node, axis=0)  # (time, )
        graph_feature_avg = np.expand_dims(graph_feature_avg, axis=0)  # (1, time)
        return graph_feature_avg

    def extract_feature(self, dFC_mat, feature2extract, graph_property=None):
        """
        feature2extract_list = [
            'all',
            'spatial', 'temporal',
            'inter_time_corr', 'inter_conn_corr',
            'dFC_avg', 'dFC_var',
            'graph_spatial', 'graph_temporal'
        ]
        """
        feature = None
        if feature2extract == "all":
            feature = self.feature_all(dFC_mat)
        if feature2extract == "spatial":
            feature = self.feature_spatial(dFC_mat)
        if feature2extract == "temporal":
            feature = self.feature_temporal(dFC_mat)
        if feature2extract == "inter_time_corr":
            feature = self.feature_inter_time_corr(dFC_mat)
        if feature2extract == "inter_conn_corr":
            feature = self.feature_inter_conn_corr(dFC_mat)
        if feature2extract == "dFC_avg":
            feature = self.feature_dFC_avg(dFC_mat)
        if feature2extract == "dFC_var":
            feature = self.feature_dFC_var(dFC_mat)
        if feature2extract == "graph_spatial":
            feature = self.feature_graph_spatial(dFC_mat, graph_property=graph_property)
        if feature2extract == "graph_temporal":
            feature = self.feature_graph_temporal(dFC_mat, graph_property=graph_property)

        return feature

    def dFC_mat_lst_similarity(
        self, dFC_mat_lst, feature2extract, metric, graph_property=None
    ):

        sim_mat_over_sample = None
        for i, dFC_mat_i in enumerate(dFC_mat_lst):
            for j, dFC_mat_j in enumerate(dFC_mat_lst):

                if j <= i:
                    continue

                assert dFC_mat_i.shape == dFC_mat_j.shape, "shape mismatch"

                feature_i = self.extract_feature(
                    dFC_mat_i,
                    feature2extract=feature2extract,
                    graph_property=graph_property,
                )  # (samples, variables)
                feature_j = self.extract_feature(
                    dFC_mat_j,
                    feature2extract=feature2extract,
                    graph_property=graph_property,
                )  # (samples, variables)

                sim_over_sample = list()
                for sample in range(feature_i.shape[0]):
                    if (
                        np.var(feature_i[sample, :]) == 0
                        or np.var(feature_j[sample, :]) == 0
                    ):
                        sim = 0
                    else:
                        if metric == "corr":
                            sim = np.corrcoef(feature_i[sample, :], feature_j[sample, :])[
                                0, 1
                            ]
                        elif metric == "spearman":
                            sim, p = stats.spearmanr(
                                feature_i[sample, :], feature_j[sample, :]
                            )
                        elif metric == "MI":
                            sim = mutual_information(
                                X=feature_i[sample, :], Y=feature_j[sample, :], N_bins=100
                            )
                        elif metric == "euclidean_distance":
                            # normalized euclidean is used
                            sim = normalized_euc_dist(
                                x=feature_i[sample, :], y=feature_j[sample, :]
                            )
                    sim_over_sample.append(sim)

                if sim_mat_over_sample is None:
                    sim_mat_over_sample = np.zeros(
                        (len(sim_over_sample), len(dFC_mat_lst), len(dFC_mat_lst))
                    )
                sim_mat_over_sample[:, i, j] = np.array(sim_over_sample)
                sim_mat_over_sample[:, j, i] = sim_mat_over_sample[:, i, j]

        return sim_mat_over_sample

    def assess_similarity(self, dFC_lst, downsampling_method="default", **param_dict):
        """
        downsampling_method: 'default' picks FCs at common_TRs
        while 'SWed' uses a sliding window to downsample
        """
        methods_assess = {}

        # sort dFC_lst according to methods names
        old_list = [dFC.measure.measure_name for dFC in dFC_lst]
        new_list = deepcopy(old_list)
        new_list.sort()

        new_order = find_new_order(old_list, new_list)
        dFC_lst = [dFC_lst[i] for i in new_order]

        common_TRs = TR_intersection(dFC_lst)

        measure_lst = list()
        TS_info_lst = list()
        dFC_mat_lst = list()
        for dFC in dFC_lst:
            measure_lst.append(dFC.measure)
            TS_info_lst.append(dFC.TS_info)
            if downsampling_method == "SWed":
                dFC_mat_lst.append(
                    dFC.SWed_dFC_mat(
                        W=param_dict["W"],
                        n_overlap=param_dict["n_overlap"],
                        tapered_window=param_dict["tapered_window"],
                    )
                )
            else:
                dFC_mat_lst.append(dFC.get_dFC_mat(TRs=common_TRs))

        methods_assess["measure_lst"] = measure_lst
        methods_assess["TS_info_lst"] = TS_info_lst
        methods_assess["common_TRs"] = common_TRs

        ########## dFC samples ##########

        dFC_samples = {}
        for i, dFC_mat in enumerate(dFC_mat_lst):
            dFC_samples[str(i)] = dFC_mat
        methods_assess["dFC_samples"] = dFC_samples

        ########## time record ##########

        time_record_dict = {}
        for i, dFC in enumerate(dFC_lst):
            time_record = {}
            time_record["FCS_fit"] = dFC.measure.FCS_fit_time
            time_record["dFC_assess"] = dFC.measure.dFC_assess_time
            time_record_dict[str(i)] = time_record
        methods_assess["time_record_dict"] = time_record_dict

        ########## subj_dFC_sim ##########
        # returns correlation/MI/spearman corr/euclidean distance between results of dFC
        # measures in a subject
        feature2extract_list = [
            # 'all',
            "spatial",
            "temporal",
            "inter_time_corr",
            "inter_conn_corr",
            "dFC_avg",
            "dFC_var",
            # 'graph_spatial', 'graph_temporal'
        ]
        metric_list = ["corr", "spearman", "MI", "euclidean_distance"]
        graph_property_list = ["ECM", "shortest_path", "degree", "clustering_coef"]
        methods_assess["all"] = {}
        for metric in metric_list:
            methods_assess["all"][metric] = self.dFC_mat_lst_similarity(
                dFC_mat_lst, feature2extract="all", metric=metric
            )
        methods_assess["feature_based"] = {}
        for feature2extract in feature2extract_list:
            methods_assess["feature_based"][feature2extract] = (
                self.dFC_mat_lst_similarity(
                    dFC_mat_lst, feature2extract=feature2extract, metric="spearman"
                )
            )
        methods_assess["graph_based"] = {}
        methods_assess["graph_based"]["graph_spatial"] = {}
        methods_assess["graph_based"]["graph_temporal"] = {}
        for graph_property in graph_property_list:
            methods_assess["graph_based"]["graph_spatial"][graph_property] = (
                self.dFC_mat_lst_similarity(
                    dFC_mat_lst,
                    feature2extract="graph_spatial",
                    metric="spearman",
                    graph_property=graph_property,
                )
            )
            methods_assess["graph_based"]["graph_temporal"][graph_property] = (
                self.dFC_mat_lst_similarity(
                    dFC_mat_lst,
                    feature2extract="graph_temporal",
                    metric="spearman",
                    graph_property=graph_property,
                )
            )
        # ########## dFC temporal average and variance ##########

        methods_assess["dFC_avg"] = [
            self.feature_dFC_avg(dFC_mat) for dFC_mat in dFC_mat_lst
        ]

        methods_assess["dFC_var"] = [
            self.feature_dFC_var(dFC_mat) for dFC_mat in dFC_mat_lst
        ]

        ########## Fractional Occupancy ##########

        FO_lst = self.FO_calc(dFC_lst, common_TRs=common_TRs)
        methods_assess["FO"] = FO_lst

        ########## transition frequency ##########

        transition_stats_lst = self.transition_stats(dFC_lst, common_TRs=common_TRs)
        methods_assess["transition_stats"] = transition_stats_lst

        ##############################################
        return methods_assess

    def run(self, FILTERS, downsampling_method="default"):
        """
        downsampling_method: 'default' picks FCs at common_TRs
        while 'SWed' uses a sliding window to downsample
        """
        parallelize = True
        output = {}
        if parallelize:
            out_lst = Parallel(n_jobs=4, verbose=0, backend="loky")(
                delayed(self.assess_similarity)(
                    dFC_lst=filter_dFC_lst(self.dFC_lst, **FILTERS[filter]),
                    downsampling_method=downsampling_method,
                    **FILTERS[filter],
                )
                for filter in FILTERS
            )
            for i, filter in enumerate(FILTERS):
                output[filter] = out_lst[i]
        else:
            for filter in FILTERS:
                param_dict = FILTERS[filter]
                dFC_lst2check = filter_dFC_lst(self.dFC_lst, **param_dict)
                output[filter] = self.assess_similarity(
                    dFC_lst=dFC_lst2check,
                    downsampling_method=downsampling_method,
                    **param_dict,
                )

        return output


#################################################################################################
