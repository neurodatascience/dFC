"""
dFC class

Created on Jun 29 2023
@author: Mohammad Torabi
"""

import numpy as np

from .dfc_utils import (
    SW_downsample,
    node_info2network,
    node_labels2networks,
    rank_norm_dFC_dict,
    visualize_conn_mat_dict,
)

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


class DFC:
    def __init__(self, measure=None):

        # assert not measure is None, \
        #     "measure arg must be provided."
        self.measure_ = measure
        self.FCSs_ = None  # is a dict
        self.FCS_idx_ = None  # is a dict
        self.FCS_proba_ = (
            None  # is a 2D numpy array of probabilities for each FCS at each time point
        )
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
            TRs_lst.append("TR" + str(TR))
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

    @property
    def FCS_proba(self):
        """
        FCS_proba is a 2D numpy array of probabilities for each FCS at each time point
        shape = (n_time, n_states)
        """
        return self.FCS_proba_

    # test this
    @property
    def FCS_idx_array(self):
        return np.array(
            [
                int(self.FCS_idx[TR][self.FCS_idx[TR].find("S") + 1 :]) - 1
                for TR in self.FCS_idx
            ]
        )

    @property
    def TS_info(self):
        # info of the time series used for dFC estimation
        return self.TS_info_

    # test
    def state_TC(self, TRs=None, state_match=False, state_match_dict=None):
        # returns a np array of state indices over TRs in TRs

        if TRs is None:
            TRs = self.TR_array

        if not type(TRs[0]) is str:
            TRs_lst = list()
            for TR in TRs:
                TRs_lst.append("TR" + str(TR))
        else:
            TRs_lst = TRs

        state_TC = list()
        for key in self.FCS_idx:
            if key in TRs_lst:
                state = self.FCS_idx[key]
                if state_match:
                    match = state_match_dict["FCS_match"][state]["match"]
                    state_TC.append(int(match[match.find("FCS") + 3 :]))
                else:
                    state_TC.append(int(state[state.find("FCS") + 3 :]))

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
            TRs_lst.append("TR" + str(TR))

        state_act_dict = {}
        state_act_dict["state_TC"] = {}
        state_act_dict["TR_array"] = TRs
        for FCS_key in self.FCSs:
            state_act_dict["state_TC"][FCS_key] = {}
            state_act_dict["state_TC"][FCS_key]["FCS"] = self.FCSs[FCS_key]
            state_act_dict["state_TC"][FCS_key]["act_TC"] = np.zeros((len(TRs),))
        t = 0
        for TR in self.FCS_idx:
            if TR in TRs_lst:
                state_act_dict["state_TC"][self.FCS_idx[TR]]["act_TC"][t] = 1
                t = t + 1
        assert t == len(TRs), "error!"

        return state_act_dict

    # test
    def dFC2dict(self, TRs=None, fuzzy=False):
        # return dFC samples as a dictionary
        if TRs is None:
            TRs = self.TR_array
        if type(TRs) is list:
            TRs = np.array(TRs)
        TRs = TRs.astype(int)

        dFC_mat = self.get_dFC_mat(TRs=TRs, fuzzy=fuzzy)

        dFC_dict = {}
        for k, TR in enumerate(TRs):
            dFC_dict[f"TR{TR}"] = dFC_mat[k, :, :]
        return dFC_dict

    def get_dFC_mat(self, TRs=None, num_samples=None, fuzzy=False):
        """
        get dFC matrices corresponding to
        the specified TRs
        TRs should be list/ndarray not necessarily in order ?
        if num_samples specified, it will downsample
        TRs to reach that number of samples and will also
        return picked TRs
        if num_samples > len(TRs) -> picks all TRs

        ONLY FOR STATE-BASED METHODS:
        if fuzzy is True, it will return dFC matrices based on fuzzy states
        """
        if fuzzy:
            if not self.measure.is_state_based:
                raise ValueError(
                    "This method is only applicable to state-based methods. "
                    "Please use get_dFC_mat() for state-free methods."
                )

        if TRs is None:
            TRs = self.TR_array

        if type(TRs) is np.int32 or type(TRs) is np.int64 or type(TRs) is int:
            TRs = [TRs]

        if not num_samples is None:
            if num_samples < len(TRs):
                TRs = TRs[
                    np.linspace(0, len(TRs), num_samples, endpoint=False, dtype=int)
                ]

        dFC_mat = list()
        for TR in TRs:
            if fuzzy:
                TR_index = np.where(self.TR_array == TR)[0]
                FC_mat = np.zeros((self.n_regions, self.n_regions))
                for i in range(self.FCS_proba.shape[1]):  # iterate over states
                    prob = self.FCS_proba[TR_index, i]
                    FC_mat += prob * self.FCSs[f"FCS{i + 1}"]
            else:
                FC_mat = self.FCSs[self.FCS_idx[f"TR{TR}"]]
            dFC_mat.append(FC_mat)

        dFC_mat = np.array(dFC_mat)

        if num_samples is None:
            return dFC_mat
        else:
            return dFC_mat, TRs

    def SWed_dFC_mat(self, W=None, n_overlap=None, tapered_window=False):
        """
        the time samples will be picked after
        averaging over a window which slides
        W is in sec
        """
        dFC_mat = self.get_dFC_mat()

        # method not applicable to SW-based methods
        if "sw_method" in self.measure.info:
            return dFC_mat

        dFC_mat_new = SW_downsample(
            data=dFC_mat,
            Fs=self.TS_info["Fs"],
            W=W,
            n_overlap=n_overlap,
            tapered_window=tapered_window,
        )

        return dFC_mat_new

    def set_dFC(self, FCSs, FCS_idx=None, FCS_proba=None, TS_info=None, TR_array=None):
        """
        FCSs: a 3D numpy array of FC matrices with shape (n_states, n_regions, n_regions), for state-free methods: (n_time, n_regions, n_regions)
        FCS_idx: a list of indices that correspond to each FC matrix in FCSs over time, used for state-based methods.
        FCS_proba: a 2D numpy array of probabilities for each FCS at each time point, shape = (n_time, n_states), used for state-based methods.
        """

        if len(FCSs.shape) == 2:
            FCSs = np.expand_dims(FCSs, axis=0)

        if FCS_idx is None:
            # usually for state-free methods like sliding window when we don't have FCSs
            # we consider each FC a FCS
            FCS_idx = np.arange(start=0, stop=FCSs.shape[0], step=1, dtype=int)

        if type(FCS_idx) is list:
            FCS_idx = np.array(FCS_idx)

        if len(FCS_idx.shape) > 1:
            FCS_idx = np.squeeze(FCS_idx)

        assert FCSs.shape[1] == FCSs.shape[2], "FC matrices must be square."

        assert (
            self.n_time == -1
        ), "why n_time is not -1 ? Are you adding a dFC to an existing dFC ?"

        if TR_array is None:
            # self.n_time is -1 at first. if it is not -1, it means that a dFC is already set and
            # we are adding a new dFC to it.
            TR_array = np.arange(
                start=self.n_time + 1,
                stop=self.n_time + len(FCS_idx) + 1,
                step=1,
                dtype=int,
            )

        assert np.sum(np.abs(np.sort(TR_array) - TR_array)) == 0.0, "TRs not sorted !"

        if FCS_proba is not None and FCS_idx is not None:
            assert FCS_proba.shape[0] == len(
                FCS_idx
            ), "FCS_proba shape does not match FCSs shape (n_time)."
            assert (
                FCS_proba.shape[1] == FCSs.shape[0]
            ), "FCS_proba shape does not match FCSs shape (n_states)."
            assert np.allclose(
                FCS_proba.sum(axis=1), 1
            ), "FCS_proba probabilities must sum to 1 for each time point."
            assert len(TR_array) == FCS_proba.shape[0], (
                "TR_array length does not match FCS_proba shape (n_time). "
                f"TR_array length: {len(TR_array)}, FCS_proba shape: {FCS_proba.shape}"
            )

        # the input FCS_idx is ranged from 0 to len(FCS)-1 but we shift it to 1 to len(FCS)
        self.FCSs_ = {}
        for i, FCS in enumerate(FCSs):
            self.FCSs_[f"FCS{i + 1}"] = FCS

        self.FCS_idx_ = {}
        for i, idx in enumerate(FCS_idx):
            self.FCS_idx_[f"TR{TR_array[i]}"] = f"FCS{idx + 1}"  # "FCS" + str(idx + 1)

        self.FCS_proba_ = FCS_proba

        self.TS_info_ = TS_info
        self.n_regions_ = FCSs.shape[1]
        self.n_time_ = len(self.FCS_idx_)
        self.TR_array_ = TR_array

    def visualize_dFC(
        self,
        TRs=None,
        fuzzy=False,
        normalize=False,
        show_networks=False,
        rank_norm=False,
        threshold=0.0,
        fix_lim=False,
        save_image=False,
        output_root=None,
    ):

        assert not self.measure is None, "Measure is not provided."

        if TRs is None:
            TRs = self.TR_array

        if show_networks:
            if "nodes_info" in self.TS_info:
                node_networks = node_info2network(self.TS_info["nodes_info"])
            elif "node_labels" in self.TS_info:
                node_networks = node_labels2networks(self.TS_info["node_labels"])
        else:
            node_networks = None

        if rank_norm:
            dFC_dict = rank_norm_dFC_dict(self.dFC2dict(TRs=TRs, fuzzy=fuzzy))
            cmap = "plasma"
            center_0 = False
        else:
            dFC_dict = self.dFC2dict(TRs=TRs, fuzzy=fuzzy)
            cmap = "seismic"
            center_0 = True

        visualize_conn_mat_dict(
            data=dFC_dict,
            title=self.measure.measure_name + " dFC",
            fix_lim=fix_lim,
            normalize=normalize,
            node_networks=node_networks,
            cmap=cmap,
            center_0=center_0,
            save_image=save_image,
            output_root=output_root,
        )
