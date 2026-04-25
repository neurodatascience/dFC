# -*- coding: utf-8 -*-
"""
Functions to facilitate applying ML algorithms to dFC.

Created on Aug 8 2024
@author: Mohammad Torabi
"""
import os
import warnings

import numpy as np
from scipy.spatial import procrustes
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors, kneighbors_graph
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.utils import shuffle

from .dfc_utils import dFC_mat2vec, dFC_vec2mat, rank_norm
from .task_utils import (
    calc_relative_task_on,
    calc_rest_duration,
    calc_task_duration,
    calc_transition_freq,
    extract_task_presence,
)

################################# Feature Loading Functions ####################################


def find_available_subjects(dFC_root, task, run=None, session=None, dFC_id=None):
    """
    Find the subjects that have dFC results for the given task and dFC_id (method).

    If run is specified, the dFC results for that run will be used.
    Otherwise, the subjects that have dFC results at least for one run will returned.

    If session is specified, the dFC results for that session will be used.
    Otherwise, it is considered that the dataset does not have session information.
    Note that not specifying session will cause error if the dataset has session information.
    """
    SUBJECTS = list()
    ALL_SUBJ_FOLDERS = os.listdir(f"{dFC_root}/")
    ALL_SUBJ_FOLDERS = [folder for folder in ALL_SUBJ_FOLDERS if "sub-" in folder]
    ALL_SUBJ_FOLDERS.sort()
    for subj_folder in ALL_SUBJ_FOLDERS:
        if session is None:
            ALL_DFC_FILES = os.listdir(f"{dFC_root}/{subj_folder}/")
        else:
            if not os.path.exists(f"{dFC_root}/{subj_folder}/{session}/"):
                continue
            ALL_DFC_FILES = os.listdir(f"{dFC_root}/{subj_folder}/{session}/")
        ALL_DFC_FILES = [
            dFC_file for dFC_file in ALL_DFC_FILES if f"_{task}_" in dFC_file
        ]
        if dFC_id is not None:
            ALL_DFC_FILES = [
                dFC_file for dFC_file in ALL_DFC_FILES if f"_{dFC_id}.npy" in dFC_file
            ]
        if run is not None:
            ALL_DFC_FILES = [
                dFC_file for dFC_file in ALL_DFC_FILES if f"_{run}_" in dFC_file
            ]
        if session is not None:
            ALL_DFC_FILES = [
                dFC_file for dFC_file in ALL_DFC_FILES if f"_{session}_" in dFC_file
            ]
        ALL_DFC_FILES.sort()
        if len(ALL_DFC_FILES) > 0:
            SUBJECTS.append(subj_folder)
    return SUBJECTS


def load_dFC(dFC_root, subj, task, dFC_id, run=None, session=None):
    """
    Load the dFC results for a given subject, task, dFC_id, run and session.
    """
    if session is None:
        if run is None:
            dFC = np.load(
                f"{dFC_root}/{subj}/dFC_{task}_{dFC_id}.npy", allow_pickle="TRUE"
            ).item()
        else:
            dFC = np.load(
                f"{dFC_root}/{subj}/dFC_{task}_{run}_{dFC_id}.npy", allow_pickle="TRUE"
            ).item()
    else:
        if run is None:
            dFC = np.load(
                f"{dFC_root}/{subj}/{session}/dFC_{session}_{task}_{dFC_id}.npy",
                allow_pickle="TRUE",
            ).item()
        else:
            dFC = np.load(
                f"{dFC_root}/{subj}/{session}/dFC_{session}_{task}_{run}_{dFC_id}.npy",
                allow_pickle="TRUE",
            ).item()

    return dFC


def load_task_data(roi_root, subj, task, run=None, session=None):
    """
    Load the task data for a given subject, task and run.
    """
    if session is None:
        if run is None:
            task_data = np.load(
                f"{roi_root}/{subj}/{subj}_{task}_task-data.npy", allow_pickle="TRUE"
            ).item()
        else:
            task_data = np.load(
                f"{roi_root}/{subj}/{subj}_{task}_{run}_task-data.npy",
                allow_pickle="TRUE",
            ).item()
    else:
        if run is None:
            task_data = np.load(
                f"{roi_root}/{subj}/{session}/{subj}_{session}_{task}_task-data.npy",
                allow_pickle="TRUE",
            ).item()
        else:
            task_data = np.load(
                f"{roi_root}/{subj}/{session}/{subj}_{session}_{task}_{run}_task-data.npy",
                allow_pickle="TRUE",
            ).item()

    return task_data


################################# Feature Extraction Functions ####################################


def extract_task_features(TASKS, RUNS, session, roi_root, dFC_root, no_hrf=False):
    """
    Extract task features from the event data.

    if no_hrf is True, the task presence will be binarized without convolving with HRF.
    Therefore the task features will be extracted based on the event labels and
    without the effect of HRF.
    """
    task_features = {
        "task": list(),
        "run": list(),
        "relative_task_on": list(),
        "avg_task_duration": list(),
        "var_task_duration": list(),
        "avg_rest_duration": list(),
        "var_rest_duration": list(),
        "num_of_transitions": list(),
        "relative_transition_freq": list(),
    }
    for task_id, task in enumerate(TASKS):

        if task == "task-restingstate":
            continue

        for run in RUNS[task]:

            SUBJECTS = find_available_subjects(
                dFC_root=dFC_root, task=task, run=run, session=session
            )

            for subj in SUBJECTS:
                # event data
                task_data = load_task_data(
                    roi_root=roi_root, subj=subj, task=task, run=run, session=session
                )
                Fs_task = task_data["Fs_task"]
                TR_task = 1 / Fs_task

                task_presence, indices = extract_task_presence(
                    event_labels=task_data["event_labels"],
                    TR_task=TR_task,
                    TR_mri=task_data["TR_mri"],
                    binary=True,
                    binarizing_method="GMM",
                    no_hrf=no_hrf,
                )
                task_presence = task_presence[indices]

                relative_task_on = calc_relative_task_on(task_presence)
                # task duration
                avg_task_duration, var_task_duration = calc_task_duration(
                    task_presence, task_data["TR_mri"]
                )
                # rest duration
                avg_rest_duration, var_rest_duration = calc_rest_duration(
                    task_presence, task_data["TR_mri"]
                )
                # freq of transitions
                num_of_transitions, relative_transition_freq = calc_transition_freq(
                    task_presence
                )

                task_features["task"].append(task)
                task_features["run"].append(run)
                task_features["relative_task_on"].append(relative_task_on)
                task_features["avg_task_duration"].append(avg_task_duration)
                task_features["var_task_duration"].append(var_task_duration)
                task_features["avg_rest_duration"].append(avg_rest_duration)
                task_features["var_rest_duration"].append(var_rest_duration)
                task_features["num_of_transitions"].append(num_of_transitions)
                task_features["relative_transition_freq"].append(relative_transition_freq)

    return task_features


def dFC_feature_extraction_subj_lvl(
    dFC,
    task_data,
    dynamic_pred="no",
    normalize_dFC=False,
    FCS_proba_for_SB=True,
):
    """
    Extract features and target for task presence classification
    for a single subject.
    dynamic_pred: "no", "past", "past_and_future"

    FCS_proba_for_SB: if True, use FCS_proba as features for state-based dFC.
    If False, use dFC_vecs (dFC matrix as features).
    """
    # dFC features
    # for state-based dFC, we use the FCS_proba as features
    # for state-free dFC, we use the dFC matrix as features
    if dFC.measure.is_state_based and FCS_proba_for_SB:
        # state-based dFC
        dFC_vecs = dFC.FCS_proba  # shape: (n_time, n_states)
        TR_array = dFC.TR_array

        assert dFC_vecs.shape[0] == len(
            TR_array
        ), "dFC_vecs and TR_array have different number of samples."
        assert (
            dFC_vecs.shape[1] == dFC.measure.params["n_states"]
        ), "dFC_vecs and n_states are not consistent."
    else:
        dFC_mat = dFC.get_dFC_mat()
        TR_array = dFC.TR_array
        if normalize_dFC:
            dFC_mat = rank_norm(dFC_mat, global_norm=False)
        dFC_vecs = dFC_mat2vec(dFC_mat)

    # event data
    task_presence, indices = extract_task_presence(
        event_labels=task_data["event_labels"],
        TR_task=1 / task_data["Fs_task"],
        TR_mri=task_data["TR_mri"],
        TR_array=TR_array,
        binary=True,
        binarizing_method="GMM",
    )

    # features = dFC_vecs
    # target = task_presence.ravel()

    # use absolute task presence
    features = dFC_vecs[indices, :]
    target = task_presence.ravel()[indices]

    assert (
        features.shape[0] == target.shape[0]
    ), "Features and target have different number of samples."

    if dynamic_pred == "past":
        # concat current TR and two TR before of features to predict the current TR of target
        # ignore the edge case of the first two TRs
        features = np.concatenate(
            (features, np.roll(features, 1, axis=0), np.roll(features, 2, axis=0)), axis=1
        )
        features = features[2:, :]
        target = target[2:]
    elif dynamic_pred == "past_and_future":
        # concat current TR and two TR before and after of features to predict the current TR of target
        # ignore the edge case of the first and last two TRs
        features = np.concatenate(
            (
                features,
                np.roll(features, 1, axis=0),
                np.roll(features, 2, axis=0),
                np.roll(features, -1, axis=0),
                np.roll(features, -2, axis=0),
            ),
            axis=1,
        )
        features = features[2:-2, :]
        target = target[2:-2]

    features = features.astype(np.float32, copy=False)
    target = target.astype(np.int8, copy=False)  # labels smaller & faster
    return features, target


def dFC_feature_extraction(
    task,
    train_subjects,
    test_subjects,
    dFC_id,
    roi_root,
    dFC_root,
    run=None,
    session=None,
    dynamic_pred="no",
    normalize_dFC=False,
    FCS_proba_for_SB=True,
):
    """
    Extract features and target for task presence classification
    for all subjects.
    if run is specified, dFC results for that run will be used.

    if FCS_proba_for_SB is True, use FCS_proba as features for state-based dFC.
    If False, use dFC_vecs (dFC matrix as features).
    """
    dFC_measure_name = None
    X_train = None
    y_train = None
    subj_label_train = list()
    for subj in train_subjects:

        dFC = load_dFC(
            dFC_root=dFC_root,
            subj=subj,
            task=task,
            dFC_id=dFC_id,
            run=run,
            session=session,
        )
        task_data = load_task_data(
            roi_root=roi_root, subj=subj, task=task, run=run, session=session
        )

        X_subj, y_subj = dFC_feature_extraction_subj_lvl(
            dFC=dFC,
            task_data=task_data,
            dynamic_pred=dynamic_pred,
            normalize_dFC=normalize_dFC,
            FCS_proba_for_SB=FCS_proba_for_SB,
        )

        # to make computations faster
        X_subj = X_subj.astype(np.float32, copy=False)
        y_subj = y_subj.astype(np.int8, copy=False)

        subj_label_train.extend([subj for i in range(X_subj.shape[0])])
        if X_train is None and y_train is None:
            X_train = X_subj
            y_train = y_subj
        else:
            X_train = np.concatenate((X_train, X_subj), axis=0)
            y_train = np.concatenate((y_train, y_subj), axis=0)

        dFC_measure_name_new = dFC.measure.measure_name
        if dFC_measure_name is None:
            dFC_measure_name = dFC_measure_name_new
        else:
            assert (
                dFC_measure_name == dFC_measure_name_new
            ), "dFC measure is not consistent."

    X_test = None
    y_test = None
    subj_label_test = list()
    for subj in test_subjects:
        dFC = load_dFC(
            dFC_root=dFC_root,
            subj=subj,
            task=task,
            dFC_id=dFC_id,
            run=run,
            session=session,
        )
        task_data = load_task_data(
            roi_root=roi_root, subj=subj, task=task, run=run, session=session
        )

        X_subj, y_subj = dFC_feature_extraction_subj_lvl(
            dFC=dFC,
            task_data=task_data,
            dynamic_pred=dynamic_pred,
            normalize_dFC=normalize_dFC,
            FCS_proba_for_SB=FCS_proba_for_SB,
        )

        # to make computations faster
        X_subj = X_subj.astype(np.float32, copy=False)
        y_subj = y_subj.astype(np.int8, copy=False)

        subj_label_test.extend([subj for i in range(X_subj.shape[0])])
        if X_test is None and y_test is None:
            X_test = X_subj.astype(np.float32, copy=False)
            y_test = y_subj.astype(np.int8, copy=False)
        else:
            X_test = np.concatenate((X_test, X_subj), axis=0)
            y_test = np.concatenate((y_test, y_subj), axis=0)

        dFC_measure_name_new = dFC.measure.measure_name
        if dFC_measure_name is None:
            dFC_measure_name = dFC_measure_name_new
        else:
            assert (
                dFC_measure_name == dFC_measure_name_new
            ), "dFC measure is not consistent."

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    subj_label_train = np.array(subj_label_train)
    subj_label_test = np.array(subj_label_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        subj_label_train,
        subj_label_test,
        dFC_measure_name,
    )


################################# Feature Embedding Functions ####################################


def precheck_for_procruste(X_best, X_subj):
    """
    Check if the two matrices have the same number of rows. if not, make them the same.
    """
    # for the procrustes transformation, the number of samples should be the same
    if X_subj.shape[0] > X_best.shape[0]:
        # add zero rows to the embedding of the best subject
        X_best_new = np.concatenate(
            (
                X_best,
                np.zeros(
                    (
                        X_subj.shape[0] - X_best.shape[0],
                        X_best.shape[1],
                    )
                ),
            ),
            axis=0,
        )
    elif X_subj.shape[0] < X_best.shape[0]:
        # remove extra rows from the embedding of the best subject
        X_best_new = X_best[: X_subj.shape[0], :]
    else:
        X_best_new = X_best

    X_best_new = X_best_new.copy()

    return X_best_new


def generalized_procrustes(X_embed_dict, max_iter=1000, tol=1e-6):
    """
    Generalized Procrustes Analysis

    X_embed_dict: dict
        dict of scans and their embeddings

    returns the mean X across scans to be used as the reference for procrustes transformation
    """
    # initial step
    # not all scans have the same number of samples
    # find the max number of samples among all scans
    max_samples = 0
    for scan in X_embed_dict:
        if X_embed_dict[scan].shape[0] > max_samples:
            max_samples = X_embed_dict[scan].shape[0]

    # find the mean embedding of all scan to use as the reference for procrustes transformation
    X_list = []
    for scan in X_embed_dict:
        X_scan_embed = X_embed_dict[scan]
        # add zero rows to the embedding of the scan with less samples
        if X_scan_embed.shape[0] < max_samples:
            X_scan_embed_new = np.concatenate(
                (
                    X_scan_embed,
                    np.zeros(
                        (
                            max_samples - X_scan_embed.shape[0],
                            X_scan_embed.shape[1],
                        )
                    ),
                ),
                axis=0,
            )
        else:
            X_scan_embed_new = X_scan_embed
        X_list.append(X_scan_embed_new)

    # now iteratively find the mean X for transform
    for _ in range(10):

        try:
            # initialize Procrustes distance
            current_distance = 0

            num_X = len(X_list)

            # initialize a mean X by randomly selecting
            # one of the Xs using np.random.choice
            mean_X = X_list[np.random.choice(num_X)]

            # create array for new Xs, add
            new_Xs = np.zeros(np.array(X_list).shape)

            counter = 0
            flag = False
            while True:
                counter += 1
                if counter > max_iter:
                    # if the algorithm does not converge, break the cycle
                    # to avoid infinite loop
                    flag = True
                    break

                # add the mean X as first element of array
                new_Xs[0] = mean_X

                # superimpose all shapes to current mean
                for i in range(1, num_X):
                    _, new_X, _ = procrustes(mean_X, X_list[i])
                    new_Xs[i] = new_X

                # calculate new mean
                new_mean = np.mean(new_Xs, axis=0)

                _, _, new_distance = procrustes(new_mean, mean_X)

                # if the distance did not change, break the cycle
                if np.abs(new_distance - current_distance) < tol:
                    break

                # align the new_mean to old mean
                _, new_mean, _ = procrustes(mean_X, new_mean)

                # update mean and distance
                mean_X = new_mean
                current_distance = new_distance

            if not flag:
                # if the algorithm converged, return the mean X
                return mean_X
        except:
            continue

    raise RuntimeError("Generalized Procrustes Analysis did not converge.")


def twonn(X, discard_ratio=0.1, n_neighbors=30, eps=1e-12, metric="euclidean"):
    """
    TWO-NN intrinsic dimension estimator.

    Parameters
    ----------
    X : (n_samples, n_features)
    discard_ratio : float in [0,1)
        Fraction of largest mu values to discard (tail trimming).
    n_neighbors : int
        Number of neighbors to query (must be >= 3 ideally, and <= n_samples-1).
    eps : float
        Numerical tolerance for filtering mu values.
    metric : str
        Distance metric for NearestNeighbors.

    Returns
    -------
    d : float
        Estimated intrinsic dimension.
    """
    X = np.asarray(X)
    n = X.shape[0]
    if n < 5:
        raise ValueError("TWO-NN needs more samples (n >= 5 is a practical minimum).")

    k = int(min(max(n_neighbors, 3), n - 1))  # at least 3, at most n-1

    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    distances, _ = nn.kneighbors(X, return_distance=True)

    mu = np.full(n, np.nan, dtype=float)

    for i in range(n):
        # distances[i, 0] is typically 0 (self). Find first two *positive* distances
        pos = distances[i][distances[i] > eps]
        if pos.size >= 2:
            r1, r2 = pos[0], pos[1]
            mu[i] = r2 / r1

    mu = mu[np.isfinite(mu)]
    mu = mu[mu > 1.0 + eps]  # avoid log(1)=0 edge cases

    if mu.size < 5:
        raise ValueError(
            "Too few valid mu values after filtering; check duplicates / ties / eps."
        )

    # discard upper tail (largest mu)
    mu.sort()
    keep = int(np.floor((1.0 - discard_ratio) * mu.size))
    keep = max(5, keep)  # don't keep too few
    mu = mu[:keep]

    N = mu.size
    # plotting positions; i/(N+1) is common and avoids CDF=1 exactly
    F = np.arange(1, N + 1) / (N + 1.0)

    x = np.log(mu).reshape(-1, 1)
    y = (-np.log(1.0 - F)).reshape(-1, 1)

    lr = LinearRegression(fit_intercept=False)
    lr.fit(x, y)
    return float(lr.coef_[0, 0])


def SI_ID(
    X,
    y,
    search_range=range(2, 50, 5),
    n_neighbors_LE=125,
    LE_embedding_method="embed+procrustes",
    measure_is_state_based=False,
):
    """
    Find the intrinsic dimension of the data based on the silhouette score.
    """

    SI_score = {}
    for n_components in search_range:
        if n_components > X.shape[1]:
            # if the number of components is larger than the number of features, break
            break
        try:
            X_train_embed, _ = embed_dFC_features(
                train_subjects=["subj"],
                test_subjects=[],
                X_train=X,
                X_test=None,
                y_train=y,
                y_test=None,
                subj_label_train=np.array(["subj"] * len(y)),
                subj_label_test=None,
                embedding="LE",
                n_components=n_components,
                n_neighbors_LE=n_neighbors_LE,
                LE_embedding_method=LE_embedding_method,
                measure_is_state_based=measure_is_state_based,
            )
        except Exception as e:
            warnings.warn(
                f"Error in SI_ID embedding with n_components={n_components}: {e}. Skipping this n_components."
            )
            continue

        SI_score[n_components] = silhouette_score(X_train_embed, y)

    # find the intrinsic dimension based on the silhouette score
    intrinsic_dim = max(SI_score, key=SI_score.get)

    return intrinsic_dim


import numpy as np
from sklearn.neighbors import NearestNeighbors


def localpca_intrinsic_dim(
    X,
    k=20,
    method="explained_var",  # "explained_var" or "eigengap"
    var_threshold=0.9,  # used for explained_var
    max_dim=None,  # cap returned dim (optional)
    center=True,
    metric="euclidean",
    random_state=0,
    agg="median",  # "median", "mean", "trimmed_mean"
    trim=0.1,  # used if agg="trimmed_mean"
    eps=1e-12,
):
    """
    Local PCA intrinsic dimension estimation.

    Parameters
    ----------
    X : (n_samples, n_features)
    k : int
        Neighborhood size (kNN). Must be < n_samples.
    method : str
        "explained_var": choose smallest d achieving cumulative variance >= var_threshold
        "eigengap": choose d maximizing eigenvalue ratio lambda_d / lambda_{d+1}
    var_threshold : float
        Threshold for explained_var method.
    max_dim : int or None
        Max dimension to consider/return; defaults to min(n_features, k-1).
    center : bool
        Whether to mean-center each neighborhood before PCA.
    metric : str
        Metric for kNN graph.
    agg : str
        Aggregation across points: "median", "mean", "trimmed_mean"
    trim : float
        Trimming fraction for trimmed_mean.
    eps : float
        Numerical stability.

    Returns
    -------
    d_global : float
        Aggregated intrinsic dimension estimate.
    d_local : (n_samples,) int
        Local dimension estimates.
    """
    X = np.asarray(X, dtype=float)
    n, D = X.shape
    if n < 5:
        raise ValueError("Need more samples for localPCA ID.")
    if k >= n:
        raise ValueError(f"k must be < n_samples (got k={k}, n={n}).")

    # Choose max_dim limit
    max_possible = min(D, k - 1)  # local covariance rank limited by k-1 if centered
    if max_dim is None:
        max_dim = max_possible
    else:
        max_dim = int(min(max_dim, max_possible))
        max_dim = max(1, max_dim)

    # kNN indices (exclude self by requesting k+1 and dropping first)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    _, idx = nn.kneighbors(X, return_distance=True)
    nbrs = idx[:, 1:]  # (n, k)

    d_local = np.zeros(n, dtype=int)

    for i in range(n):
        Xi = X[nbrs[i]]  # (k, D)
        if center:
            Xi = Xi - Xi.mean(axis=0, keepdims=True)

        # PCA via SVD of neighborhood matrix
        # Xi = U S Vt ; singular values S relate to eigenvalues of covariance
        # covariance eigenvalues proportional to (S^2) / (k-1)
        # we can work directly with S^2
        try:
            # full_matrices=False keeps it fast
            _, S, _ = np.linalg.svd(Xi, full_matrices=False)
        except np.linalg.LinAlgError:
            d_local[i] = 1
            continue

        lam = S**2  # proportional to variance along PCs
        if lam.size == 0:
            d_local[i] = 1
            continue

        lam = lam[: max_dim + 1]  # for eigengap need d and d+1
        lam = np.maximum(lam, eps)

        if method == "explained_var":
            lam_use = lam[:max_dim]
            cum = np.cumsum(lam_use)
            total = cum[-1]
            if total <= eps:
                d_local[i] = 1
            else:
                frac = cum / total
                d_local[i] = int(np.searchsorted(frac, var_threshold) + 1)

        elif method == "eigengap":
            # need ratios up to max_dim-1: lam[d-1]/lam[d]
            lam_use = lam[: max_dim + 1]  # ensures lam[d] exists
            if lam_use.size < 2:
                d_local[i] = 1
            else:
                ratios = lam_use[:-1] / lam_use[1:]
                # pick d that maximizes ratio, d in [1..max_dim]
                d_local[i] = int(np.argmax(ratios) + 1)
        else:
            raise ValueError(f"Unknown method: {method}")

    # aggregate
    if agg == "median":
        d_global = float(np.median(d_local))
    elif agg == "mean":
        d_global = float(np.mean(d_local))
    elif agg == "trimmed_mean":
        d_sorted = np.sort(d_local)
        m = len(d_sorted)
        lo = int(np.floor(trim * m))
        hi = int(np.ceil((1 - trim) * m))
        hi = max(hi, lo + 1)
        d_global = float(np.mean(d_sorted[lo:hi]))
    else:
        raise ValueError(f"Unknown agg: {agg}")

    return d_global, d_local


def find_intrinsic_dim(
    X,
    y,
    subj_label,
    subjects,
    method="SI",
    n_neighbors_LE=125,
    search_range_SI=range(2, 50, 5),
    LE_embedding_method="embed+procrustes",
    measure_is_state_based=False,
):
    """
    Find the number of components to use for embedding the data using LE.
    Find the average intrinsic dimension across all subjects.

    method: "SI" or "twonn" or "localpca"

    Returns:
    intrinsic_dim: number of components to use for embedding
    """
    if method == "SI":
        intrinsic_dim_all = list()
        for subject in subjects:
            X_subj = X[subj_label == subject, :]
            y_subj = y[subj_label == subject]
            try:
                # some subjects may not have enough samples to estimate the intrinsic dimension
                subj_estim_ID = SI_ID(
                    X_subj,
                    y_subj,
                    search_range=search_range_SI,
                    n_neighbors_LE=n_neighbors_LE,
                    LE_embedding_method=LE_embedding_method,
                    measure_is_state_based=measure_is_state_based,
                )
                intrinsic_dim_all.append(subj_estim_ID)
            except Exception as e:
                warnings.warn(
                    f"Error in SI_ID for subject {subject}: {e}. Skipping this subject."
                )
                continue
        intrinsic_dim = int(np.mean(intrinsic_dim_all))
    elif method == "twonn":
        intrinsic_dim_all = list()
        for subject in subjects:
            X_subj = X[subj_label == subject, :]
            intrinsic_dim_all.append(
                twonn(X_subj, discard_ratio=0.1, metric="correlation")
            )
        intrinsic_dim = int(np.median(intrinsic_dim_all))
    elif method == "localpca":
        intrinsic_dim_all = list()
        for subject in subjects:
            X_subj = X[subj_label == subject, :]
            intrinsic_dim_diff_k = list()
            # seatryrch 0.2 * X_subj.shape[0] and 0.3 * X_subj.shape[0] for k
            for k in range(
                max(5, int(0.1 * X_subj.shape[0])),  # not letting go below 5
                int(0.3 * X_subj.shape[0]),
                5,
            ):
                if k == 1:
                    warnings.warn(
                        f"Warning: k=1 is not valid for localpca_intrinsic_dim. Skipping k=1 for subject {subject}."
                    )
                    continue
                try:
                    d_global, _ = localpca_intrinsic_dim(
                        X_subj,
                        k=k,
                        method="explained_var",
                        var_threshold=0.9,
                        center=True,
                        metric="correlation",
                        agg="median",
                    )
                    if np.isfinite(d_global) and d_global >= 1:
                        intrinsic_dim_diff_k.append(d_global)
                except Exception as e:
                    warnings.warn(
                        f"Error in localpca_intrinsic_dim for subject {subject} with k={k}: {e}."
                    )
                    continue
            if len(intrinsic_dim_diff_k) == 0:
                warnings.warn(
                    f"No valid intrinsic dimensions found for subject {subject}."
                )
                continue
            intrinsic_dim_all.append(int(np.mean(intrinsic_dim_diff_k)))
        if len(intrinsic_dim_all) == 0:
            raise ValueError("No valid intrinsic dimensions found for any subject.")
        intrinsic_dim = int(np.median(intrinsic_dim_all))
    return intrinsic_dim


def LE_transform(X, n_components, n_neighbors, distance_metric="euclidean"):
    """
    Apply Laplacian Eigenmaps (LE) to transform data into a lower dimensional space.

    if n_neighbors >= n_samples, n_neighbors will be changed to the lower limit n_neighbors
    """
    n_neighbors_upper = int(X.shape[0] / 8)

    if n_neighbors > n_neighbors_upper:
        n_neighbors_to_be_used = n_neighbors_upper
        # raise a warning
        warnings.warn(
            f"n_neighbors is larger than the limit. n_neighbors is set to {n_neighbors_to_be_used}."
        )
    else:
        n_neighbors_to_be_used = n_neighbors

    affinity_matrix = kneighbors_graph(
        X,
        n_neighbors=n_neighbors_to_be_used,
        mode="connectivity",
        include_self=False,
        metric=distance_metric,
    )

    # Symmetrize
    affinity_matrix = affinity_matrix.maximum(affinity_matrix.T)

    LE = SpectralEmbedding(
        n_components=n_components,
        affinity="precomputed",
        n_neighbors=n_neighbors_to_be_used,
        eigen_solver="lobpcg",
    )
    X_embed = LE.fit_transform(X=affinity_matrix)
    return X_embed


def LE_transform_dFC(X, n_components, n_neighbors, distance_metric="euclidean"):
    """
    Transform dFC features into a lower dimensional space using Laplacian Eigenmaps (LE).
    This function takes care of the case where the dFC samples are not unique,
    specifically for state-based dFC features.
    """
    unique_samples = np.unique(X, axis=0)
    # if there are repeated samples, we need to apply LE on the unique samples
    if unique_samples.shape[0] < X.shape[0] // 2:
        n_neighbors_LE = int(3 / 5 * unique_samples.shape[0])
        unique_samples_embedded = LE_transform(
            X=unique_samples,
            n_components=n_components,
            n_neighbors=n_neighbors_LE,
            distance_metric=distance_metric,
        )

        # for each entry in X, put the corresponding entry in unique_samples_embedded
        # in the corresponding position in X_embedded
        X_embedded = np.zeros((X.shape[0], unique_samples_embedded.shape[1]))
        for i, sample in enumerate(unique_samples):
            idx = np.where((X == sample).all(axis=1))[0]
            if len(idx) > 0:
                X_embedded[idx] = unique_samples_embedded[i]
    else:
        # if all samples are unique, we can apply LE directly on the data
        X_embedded = LE_transform(
            X=X,
            n_components=n_components,
            n_neighbors=n_neighbors,
            distance_metric=distance_metric,
        )

    return X_embedded


def LE_embed_procustes(
    X_train,
    X_test,
    y_train,
    y_test,
    subj_label_train,
    subj_label_test,
    train_subjects,
    test_subjects,
    n_components=30,
    n_neighbors_LE=125,
    procruste_method="best_SI",
):
    procrustes_limit = int(np.sqrt(2 * X_train.shape[0]))
    if n_components > procrustes_limit:
        warnings.warn(
            f"n_components ({n_components}) is larger than the limit for procrustes method ({procrustes_limit}). Setting n_components to {procrustes_limit}."
        )
        n_components = procrustes_limit - 1
    if procruste_method == "best_SI":
        # first embed the dFC features of each subject into a lower dimensional space using LE separately
        embed_dict = {}
        for subject in train_subjects:
            # assert the samples of the same subject are contiguous
            assert np.all(
                np.diff(np.where(subj_label_train == subject)[0]) == 1
            ), f"Indices of {subject} are not consecutive"
            X_subj = X_train[subj_label_train == subject, :]
            y_subj = y_train[subj_label_train == subject]
            X_subj_embed = LE_transform_dFC(
                X=X_subj,
                n_components=n_components,
                n_neighbors=n_neighbors_LE,
                distance_metric="correlation",
            )
            SI = silhouette_score(X_subj_embed, y_subj)
            embed_dict[subject] = {"X_subj_embed": X_subj_embed, "SI": SI}

        # find the best transformation based on the SI score
        best_SI = -1
        best_subject = None
        for subject in embed_dict:
            if embed_dict[subject]["SI"] > best_SI:
                best_SI = embed_dict[subject]["SI"]
                best_subject = subject

        # apply procrustes transformation to align the embeddings of different subjects
        # use the embeddings of the subject with the highest SI score as the reference
        X_train_embed = None
        for subject in train_subjects:
            X_subj_embed = embed_dict[subject]["X_subj_embed"]
            # procrustes transformation
            if subject == best_subject:
                X_subj_embed_transformed = X_subj_embed
            else:
                # for the procrustes transformation, the number of samples should be the same
                X_best_subj_embed = precheck_for_procruste(
                    embed_dict[best_subject]["X_subj_embed"], X_subj_embed
                )
                _, X_subj_embed_transformed, _ = procrustes(
                    X_best_subj_embed, X_subj_embed
                )
            if X_train_embed is None:
                X_train_embed = X_subj_embed_transformed
            else:
                X_train_embed = np.concatenate(
                    (X_train_embed, X_subj_embed_transformed), axis=0
                )

        # apply the same transformation to the test set
        X_test_embed = None
        for subject in test_subjects:
            # assert the samples of the same subject are contiguous
            assert np.all(
                np.diff(np.where(subj_label_test == subject)[0]) == 1
            ), f"Indices of {subject} are not consecutive"
            X_subj = X_test[subj_label_test == subject, :]
            X_subj_embed = LE_transform_dFC(
                X=X_subj,
                n_components=n_components,
                n_neighbors=n_neighbors_LE,
                distance_metric="correlation",
            )
            # procrustes transformation
            # for the procrustes transformation, the number of samples should be the same
            X_best_subj_embed = precheck_for_procruste(
                embed_dict[best_subject]["X_subj_embed"], X_subj_embed
            )
            _, X_subj_embed_transformed, _ = procrustes(X_best_subj_embed, X_subj_embed)
            if X_test_embed is None:
                X_test_embed = X_subj_embed_transformed
            else:
                X_test_embed = np.concatenate(
                    (X_test_embed, X_subj_embed_transformed), axis=0
                )

    elif procruste_method == "generalized":
        # in this method we use generalized procrustes analysis to align the embeddings of different subjects
        # first embed the dFC features of each subject into a lower dimensional space using LE separately
        embed_dict = {}
        for subject in train_subjects:
            # assert the samples of the same subject are contiguous
            assert np.all(
                np.diff(np.where(subj_label_train == subject)[0]) == 1
            ), f"Indices of {subject} are not consecutive"
            X_subj = X_train[subj_label_train == subject, :]
            X_subj_embed = LE_transform_dFC(
                X=X_subj,
                n_components=n_components,
                n_neighbors=n_neighbors_LE,
                distance_metric="correlation",
            )
            embed_dict[subject] = X_subj_embed

        mean_X_train = generalized_procrustes(embed_dict)

        X_train_embed = None
        for subject in train_subjects:
            X_subj_embed = embed_dict[subject]
            mean_X_train_new_size = precheck_for_procruste(mean_X_train, X_subj_embed)
            _, X_subj_embed_transformed, _ = procrustes(
                mean_X_train_new_size, X_subj_embed
            )
            if X_train_embed is None:
                X_train_embed = X_subj_embed_transformed
            else:
                X_train_embed = np.concatenate(
                    (X_train_embed, X_subj_embed_transformed), axis=0
                )

        X_test_embed = None
        for subject in test_subjects:
            X_subj = X_test[subj_label_test == subject, :]
            X_subj_embed = LE_transform_dFC(
                X=X_subj,
                n_components=n_components,
                n_neighbors=n_neighbors_LE,
                distance_metric="correlation",
            )
            mean_X_train_new_size = precheck_for_procruste(mean_X_train, X_subj_embed)
            _, X_subj_embed_transformed, _ = procrustes(
                mean_X_train_new_size, X_subj_embed
            )
            if X_test_embed is None:
                X_test_embed = X_subj_embed_transformed
            else:
                X_test_embed = np.concatenate(
                    (X_test_embed, X_subj_embed_transformed), axis=0
                )

    return X_train_embed, X_test_embed


def rows_look_redundant(X, sample=100):
    n = X.shape[0]
    if n > sample:
        idx = np.random.choice(n, sample, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    # Hash rows quickly
    h = np.apply_along_axis(lambda r: hash(r.tobytes()), 1, Xs)
    # If more than, say, 50% duplicates -> likely state-based
    return (len(h) - len(set(h))) / len(h) > 0.5


class PLSEmbedder(BaseEstimator, TransformerMixin):
    """
    Supervised dimensionality reduction using PLSRegression.
    Returns X scores (latent components) for downstream models.

    Notes:
    - Works for binary y (0/1) and also continuous y (regression-style PLS).
    - For classification, y should typically be 0/1 or {-1,1}.
    """

    def __init__(self, n_components=10, scale=False):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel().reshape(-1, 1)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}.")

        # optional internal scaling (usually OFF if pipeline already scales)
        if self.scale:
            self.scaler_ = StandardScaler(with_mean=True, with_std=True)
            Xs = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None
            Xs = X

        # safety: cap n_components for this fold
        nmax = min(Xs.shape[0] - 1, Xs.shape[1])
        ncomp = int(self.n_components)
        if ncomp > nmax:
            raise ValueError(
                f"n_components={ncomp} is too large for fold with "
                f"n_samples={Xs.shape[0]}, n_features={Xs.shape[1]} (max {nmax})."
            )

        self.model_ = PLSRegression(n_components=ncomp, scale=False)
        self.model_.fit(Xs, y)
        return self

    def transform(self, X):
        if not hasattr(self, "model_"):
            raise RuntimeError("PLSEmbedder is not fitted yet.")

        X = np.asarray(X)
        Xs = self.scaler_.transform(X) if self.scaler_ is not None else X

        # Out-of-sample scores
        Z = Xs @ self.model_.x_rotations_
        return Z.astype(np.float32, copy=False)


def subject_center(X, subj_labels, mode="zscore"):
    Xc = np.zeros_like(X)
    for subj in np.unique(subj_labels):
        idx = subj_labels == subj
        if mode == "demean":
            Xc[idx] = X[idx] - X[idx].mean(axis=0, keepdims=True)
        elif mode == "zscore":
            mu = X[idx].mean(axis=0, keepdims=True)
            sd = X[idx].std(axis=0, keepdims=True) + 1e-6
            Xc[idx] = (X[idx] - mu) / sd
    return Xc


def select_num_components_binary_groupcv(
    X,
    y,
    groups,
    embedding_method="PLS",
    n_list=(2, 5, 10, 15, 20),
    cv=3,
    random_state=0,
):
    """
    Select number of PLS/PCA components using subject-aware CV.

    Parameters
    ----------
    X : array (n_samples, n_features)
    y : array (n_samples,) binary labels
    groups : array (n_samples,) subject IDs
    embedding_method : "PLS" or "PCA"
    n_list : iterable of candidate n_components
    cv : number of folds
    random_state : int

    Returns
    -------
    best_n : int
        Selected number of PLS/PCA components
    best_score : float
        Mean CV balanced accuracy
    """

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    groups = np.asarray(groups)

    cv_splitter = StratifiedGroupKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )

    best_n, best_score = None, -np.inf

    for n in n_list:
        fold_scores = []

        for tr, va in cv_splitter.split(X, y, groups):
            # ---- embedding (trained ONLY on train fold subjects)
            if embedding_method == "PCA":
                emb = PCA(n_components=n, svd_solver="full", whiten=False)
                Ztr = emb.fit_transform(X[tr])
                Zva = emb.transform(X[va])
            elif embedding_method == "PLS":
                emb = PLSEmbedder(n_components=n, scale=True)
                Ztr = emb.fit_transform(X[tr], y[tr])
                Zva = emb.transform(X[va])

            # ---- classifier in latent space
            clf = make_pipeline(
                StandardScaler(),
                SVC(kernel="rbf", C=1.0, gamma="scale"),
            )
            clf.fit(Ztr, y[tr])
            pred = clf.predict(Zva)

            fold_scores.append(balanced_accuracy_score(y[va], pred))

        mean_score = float(np.mean(fold_scores))

        if mean_score > best_score:
            best_score = mean_score
            best_n = n

    return best_n, best_score


def select_num_components_continuous_groupcv(
    X,
    y,
    groups,
    embedding_method="PLS",
    n_list=(2, 5, 10, 15, 20),
    cv=3,
    score="r2",  # "r2" or "neg_mse"
):
    """
    Select number of PLS/PCA components using subject-aware CV for a CONTINUOUS target.

    Parameters
    ----------
    X : array (n_samples, n_features)
    y : array (n_samples,) continuous target
    groups : array (n_samples,) subject IDs
    embedding_method : "PLS" or "PCA"
    n_list : iterable of candidate n_components
    cv : number of folds
    score : "r2" or "neg_mse"

    Returns
    -------
    best_n : int
        Selected number of PLS/PCA components
    best_score : float
        Mean CV score (higher is better)
        - R² if score="r2"
        - negative MSE if score="neg_mse"
    """

    X = np.asarray(X)
    y = np.asarray(y).ravel()  # regression target must be 1D for SVR
    groups = np.asarray(groups)

    if score not in ("r2", "neg_mse"):
        raise ValueError("score must be 'r2' or 'neg_mse'.")

    cv_splitter = GroupKFold(n_splits=cv)

    best_n, best_score = None, -np.inf

    for n in n_list:
        fold_scores = []

        for tr, va in cv_splitter.split(X, y, groups):
            # ---- embedding (trained ONLY on train fold subjects)
            if embedding_method == "PCA":
                emb = PCA(n_components=n, svd_solver="full", whiten=False)
                Ztr = emb.fit_transform(X[tr])
                Zva = emb.transform(X[va])
            elif embedding_method == "PLS":
                emb = PLSEmbedder(n_components=n, scale=True)
                # PLSRegression expects y 2D
                Ztr = emb.fit_transform(X[tr], y[tr].reshape(-1, 1))
                Zva = emb.transform(X[va])
            # ---- regressor in latent space
            reg = make_pipeline(
                StandardScaler(),
                SVR(kernel="rbf", C=1.0, gamma="scale"),
            )
            reg.fit(Ztr, y[tr])
            pred = reg.predict(Zva)

            if score == "r2":
                fold_scores.append(r2_score(y[va], pred))
            else:
                fold_scores.append(-mean_squared_error(y[va], pred))

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score, best_n = mean_score, n

    return best_n, best_score


def embed_dFC_features(
    train_subjects,
    test_subjects,
    X_train,
    X_test,
    y_train,
    y_test,
    subj_label_train,
    subj_label_test,
    embedding="PCA",
    n_components="auto",
    n_neighbors_LE=125,
    LE_embedding_method="embed+procrustes",
    measure_is_state_based=False,
    y_continuous=False,
):
    """
    Embed the dFC features into a lower dimensional space using PCA,  or PLS. For PLS, it assumes that the samples of the same subject are contiguous.

    for LE, first the LE is applied on each subj separately and then the procrustes transformation is applied to align the embeddings of different subjects.
    All the subjects are transformed into the space of the subject with the highest silhouette score.

    LE_embedding_method: "concat+embed" or "embed+procrustes"
    if the dFC features are not unique (state-based), "embed+procrustes" will not work. So this function will switch to "concat+embed" method.
    """
    # make a copy of the data
    X_train = X_train.copy()
    if X_test is not None:
        X_test = X_test.copy()

    # preprocess the data by standardizing it
    if embedding in ("PCA", "PLS"):
        # center the data by subject before PLS to remove subject effects
        X_train_c = subject_center(X_train, subj_label_train, mode="zscore")
        if X_test is not None:
            X_test_c = subject_center(X_test, subj_label_test, mode="zscore")
        else:
            X_test_c = None
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_preproc = scaler.fit_transform(X_train_c)
        if X_test is not None:
            X_test_preproc = scaler.transform(X_test_c)
        else:
            X_test_preproc = None

        if n_components == "auto":
            if y_continuous:
                best_n, _ = select_num_components_continuous_groupcv(
                    X=X_train_preproc,
                    y=y_train,
                    groups=subj_label_train,
                    embedding_method=embedding,
                    n_list=[
                        2,
                        3,
                        4,
                        5,
                        10,
                        15,
                        20,
                        25,
                        30,
                        40,
                        50,
                    ],  # you can adjust this range based on your data
                    cv=5,  # more stable
                    score="r2",
                )
            else:
                best_n, _ = select_num_components_binary_groupcv(
                    X=X_train_preproc,
                    y=y_train,
                    groups=subj_label_train,
                    embedding_method=embedding,
                    n_list=[
                        2,
                        3,
                        4,
                        5,
                        10,
                        15,
                        20,
                        25,
                        30,
                        40,
                        50,
                    ],  # you can adjust this range based on your data
                    cv=5,  # more stable
                )
            n_components = best_n

    if embedding == "PCA":
        pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
        pca.fit(X_train_preproc)
        X_train_embed = pca.transform(X_train_preproc)
        if X_test is not None:
            X_test_embed = pca.transform(X_test_preproc)
        else:
            X_test_embed = None
    elif embedding == "PLS":
        pls = PLSEmbedder(n_components=n_components, scale=True)
        # fit on train set
        X_train_embed = pls.fit_transform(X_train_preproc, y_train)
        # only transform test set
        if X_test is not None:
            X_test_embed = pls.transform(X_test_preproc)
        else:
            X_test_embed = None
    elif embedding == "LE":
        # if the dFC features are not unique (state-based), set the LE_embedding_method to "concat+embed"
        if measure_is_state_based:
            if LE_embedding_method == "embed+procrustes":
                warnings.warn(
                    "The dFC features are not unique (state-based). Switching to 'concat+embed' method."
                )
                LE_embedding_method = "concat+embed"
        # if n_components is not specified, find the intrinsic dimension of the data using training set and based on the silhouette score
        if n_components == "auto":
            if LE_embedding_method == "embed+procrustes":
                # find the list of time lengths across subjects
                n_time_across_subj = [
                    np.sum(subj_label_train == subj) for subj in train_subjects
                ]
                # find the minimum time length across subjects
                min_time_length = min(n_time_across_subj)
                # set the search range based on the minimum time length
                procrustes_limit = int(np.sqrt(2 * min_time_length))
                if procrustes_limit < 50 and procrustes_limit > 10:
                    search_range_SI = range(2, procrustes_limit, 2)
                elif procrustes_limit <= 10:
                    search_range_SI = range(2, procrustes_limit)
                else:
                    search_range_SI = range(2, 50, 5)
            else:
                if X_train.shape[0] < 7:
                    search_range_SI = range(2, X_train.shape[1] + 1)
                elif X_train.shape[1] < 24:
                    search_range_SI = range(2, X_train.shape[1] + 1, 2)
                else:
                    search_range_SI = range(2, 50, 5)
            n_components = find_intrinsic_dim(
                X=X_train,
                y=y_train,
                subj_label=subj_label_train,
                subjects=train_subjects,
                method="localpca",
                n_neighbors_LE=n_neighbors_LE,
                search_range_SI=search_range_SI,
                LE_embedding_method=LE_embedding_method,
                measure_is_state_based=measure_is_state_based,
            )

        if LE_embedding_method == "embed+procrustes":
            X_train_embed, X_test_embed = LE_embed_procustes(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                subj_label_train=subj_label_train,
                subj_label_test=subj_label_test,
                train_subjects=train_subjects,
                test_subjects=test_subjects,
                n_components=n_components,
                n_neighbors_LE=n_neighbors_LE,
                procruste_method="generalized",
            )
        elif LE_embedding_method == "concat+embed":
            # since SpectralEmbedding does not have transform method, we need to fit the LE on the whole data
            # but note that this method is used mostly for state-based dFC features, and in this case the
            # samples are the same across subjects, so we can concatenate the training and test sets
            # and then apply LE on the concatenated data
            if X_test is not None:
                X_concat = np.concatenate((X_train, X_test), axis=0)
            else:
                X_concat = X_train
            X_concat_embed = LE_transform_dFC(
                X=X_concat,
                n_components=n_components,
                n_neighbors=n_neighbors_LE,
                distance_metric="correlation",
            )
            X_train_embed = X_concat_embed[: X_train.shape[0], :]
            if X_test is not None:
                X_test_embed = X_concat_embed[X_train.shape[0] :, :]
            else:
                X_test_embed = None
    else:
        raise ValueError(f"Unknown embedding method: {embedding}")

    # to make computation faster, we can return the embeddings as float32
    X_train_embed = X_train_embed.astype(np.float32, copy=False)
    if X_test_embed is not None:
        X_test_embed = X_test_embed.astype(np.float32, copy=False)
    return X_train_embed, X_test_embed


################################# Classification Framework Functions ####################################


def get_classification_results(
    X_train,
    X_test,
    y_train,
    y_test,
    classifier_model=None,
):
    """
    Get classification results for a given classifier.
    This function fits the classifier, predicts the labels for train and test sets,
    and calculates the balanced accuracy score, recall, precision, and f1 for both sets.

    cloning ensures that the classifier is not fitted and the original classifier remains unchanged.
    """
    classifier_model = clone(classifier_model)
    classifier_model.fit(X_train, y_train)
    y_train_pred = classifier_model.predict(X_train)
    y_test_pred = classifier_model.predict(X_test)

    RESULT = {
        "model": classifier_model,
        "train": {
            "balanced accuracy": balanced_accuracy_score(y_train, y_train_pred),
            "recall": recall_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred),
            "f1": f1_score(y_train, y_train_pred),
        },
        "test": {
            "balanced accuracy": balanced_accuracy_score(y_test, y_test_pred),
            "recall": recall_score(y_test, y_test_pred),
            "precision": precision_score(y_test, y_test_pred),
            "f1": f1_score(y_test, y_test_pred),
        },
    }
    return RESULT


def logistic_regression_classify(
    X_train,
    y_train,
    X_test,
    y_test,
    subj_label_train=None,
    embedding_method=None,
):

    if embedding_method == "PCA":
        emb = PCA(whiten=False, svd_solver="full", random_state=0)
    elif embedding_method == "PLS":
        emb = PLSEmbedder(scale=False)  # IMPORTANT: avoid double scaling
    elif embedding_method is None:
        emb = None
    else:
        raise ValueError("embedding_method must be 'PCA' or 'PLS'.")

    if emb is not None:
        # Grid (keep small!)
        param_grid = {
            "emb__n_components": [5, 10, 20, 30, 50, 100],
            "lr__C": [0.001, 0.01, 0.1, 1, 10, 100],
        }
    else:
        param_grid = {"lr__C": [0.001, 0.01, 0.1, 1, 10, 100]}

    steps = [("scaler", StandardScaler())]

    if emb is not None:
        steps.append(("emb", emb))

    steps.append(
        ("lr", LogisticRegression(penalty="l1", solver="saga", max_iter=2000, tol=1e-3))
    )

    pipe = Pipeline(steps)

    # CV splitter
    if subj_label_train is None:
        Xs, ys = shuffle(X_train, y_train, random_state=0)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        fit_kwargs = {}
    else:
        Xs, ys, gs = shuffle(X_train, y_train, subj_label_train, random_state=0)
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
        fit_kwargs = {"groups": gs}

    # GridSearch on training subjects only
    gscv = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=1, scoring="balanced_accuracy")
    gscv.fit(Xs, ys, **fit_kwargs)

    # Evaluate with best estimator (already refit on full training set by default)
    model = gscv.best_estimator_

    RESULT = get_classification_results(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classifier_model=model,
    )
    RESULT["best_params"] = gscv.best_params_
    return RESULT


def SVM_classify(
    X_train,
    y_train,
    X_test,
    y_test,
    subj_label_train=None,
    embedding_method=None,
):
    if embedding_method == "PCA":
        emb = PCA(whiten=False, svd_solver="full", random_state=0)
    elif embedding_method == "PLS":
        emb = PLSEmbedder(scale=False)  # IMPORTANT: avoid double scaling
    elif embedding_method is None:
        emb = None
    else:
        raise ValueError("embedding_method must be 'PCA' or 'PLS'.")

    if emb is not None:
        # Grid (keep small!)
        param_grid = {
            "emb__n_components": [5, 10, 20, 30, 50, 100],
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", 0.01, 0.1],
        }
    else:
        param_grid = {
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", 0.01, 0.1],
        }

    steps = [("scaler", StandardScaler())]

    if emb is not None:
        steps.append(("emb", emb))

    steps.append(("svc", SVC(kernel="rbf")))

    pipe = Pipeline(steps)

    # CV splitter
    if subj_label_train is None:
        Xs, ys = shuffle(X_train, y_train, random_state=0)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        fit_kwargs = {}
    else:
        Xs, ys, gs = shuffle(X_train, y_train, subj_label_train, random_state=0)
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
        fit_kwargs = {"groups": gs}

    # GridSearch on training subjects only
    gscv = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=1, scoring="balanced_accuracy")
    gscv.fit(Xs, ys, **fit_kwargs)

    # Evaluate with best estimator (already refit on full training set by default)
    model = gscv.best_estimator_

    RESULT = get_classification_results(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        classifier_model=model,
    )
    RESULT["best_params"] = gscv.best_params_
    return RESULT


def group_permutation(y, groups, permute_groups=True):
    """
    Permute the labels while keeping the group structure intact.
    This is useful for permutation tests where we want to keep the group structure.
    Also permute the order of groups if permute_groups is True.
    If permute_groups is False, the labels within each group are permuted but the order of groups is not changed.
    This function assumes that all samples in a group have the same label.
    """
    # make sure groups is a numpy array
    groups = np.array(groups, copy=True)
    y = np.copy(y)

    unique_groups = np.unique(groups)

    # Step 1: Create a mapping from groups to labels
    group_to_label = {group: y[groups == group] for group in unique_groups}

    # Step 2: Permute each group labels
    group_to_permuted_label = {}
    for group in unique_groups:
        group_to_permuted_label[group] = np.random.permutation(group_to_label[group])

    # Step 3: Reconstruct permuted y based on groups
    # also shuffle the order of groups if permute_groups is True
    if permute_groups:
        unique_groups_permuted = np.random.permutation(unique_groups)
    else:
        unique_groups_permuted = unique_groups
    y_permuted = list()
    for group in unique_groups_permuted:
        # For each group, append the permuted label to y_permuted
        y_permuted.extend(group_to_permuted_label[group])
    # Convert to numpy array
    y_permuted = np.array(y_permuted)

    assert (
        y_permuted.shape == y.shape
    ), f"Permuted labels shape {y_permuted.shape} does not match original labels shape {y.shape}"

    return y_permuted


def get_permutation_scores(
    X_train,
    y_train,
    X_test,
    y_test,
    classifier_model,
    groups_train=None,
    n_permutations=100,
):
    """
    Get permutation scores for a given classifier and data.
    cloning ensures that the classifier is not previously fitted.
    """
    # first get the true balanced accuracy scores from original data
    classifier_original = clone(classifier_model)
    classifier_original.fit(X_train, y_train)
    y_train_pred = classifier_original.predict(X_train)
    y_test_pred = classifier_original.predict(X_test)

    # next calculate the balanced accuracy scores for permuted data
    permutation_train_scores = []
    permutation_test_scores = []
    for _ in range(n_permutations):
        if groups_train is not None:
            # permute the labels while keeping the group structure intact
            y_train_permuted = group_permutation(y_train, groups_train)
        else:
            y_train_permuted = np.random.permutation(y_train)
        model_permuted = clone(classifier_model)
        model_permuted.fit(X_train, y_train_permuted)

        y_train_permuted_pred = model_permuted.predict(X_train)
        y_test_permuted_pred = model_permuted.predict(X_test)
        permutation_train_scores.append(
            balanced_accuracy_score(y_train_permuted, y_train_permuted_pred)
        )
        permutation_test_scores.append(
            balanced_accuracy_score(y_test, y_test_permuted_pred)
        )
    p_value_train = (
        np.sum(
            np.array(permutation_train_scores)
            >= balanced_accuracy_score(y_train, y_train_pred)
        )
        + 1
    ) / (len(permutation_train_scores) + 1)
    p_value_test = (
        np.sum(
            np.array(permutation_test_scores)
            >= balanced_accuracy_score(y_test, y_test_pred)
        )
        + 1
    ) / (len(permutation_test_scores) + 1)

    return permutation_train_scores, permutation_test_scores, p_value_train, p_value_test


def softmax(x, tau=1.0, axis=1):
    z = (x - np.max(x, axis=axis, keepdims=True)) / float(tau)
    np.exp(z, out=z)
    z_sum = np.sum(z, axis=axis, keepdims=True)
    z /= z_sum
    return z


def clip_and_renorm(P, eps=1e-6, axis=1):
    P = np.asarray(P, float)
    P = np.clip(P, eps, None)
    P /= P.sum(axis=axis, keepdims=True)
    return P


# ---- log-ratio transforms ----
def clr_transform(P, eps=1e-6):
    """Centered log-ratio: log(p) - mean(log(p)) row-wise."""
    P = clip_and_renorm(P, eps=eps)
    L = np.log(P)
    return L - L.mean(axis=1, keepdims=True)  # each row sums to 0


def ilr_transform(P, eps=1e-6):
    """Pivot ILR using an orthonormal basis; returns (n, K-1)."""
    P = clip_and_renorm(P, eps=eps)
    L = np.log(P)
    clr = L - L.mean(axis=1, keepdims=True)
    K = P.shape[1]
    V = np.zeros((K, K - 1))
    # Pivot coordinates basis (orthonormal in Aitchison geometry)
    for j in range(1, K):
        V[:j, j - 1] = 1 / j
        V[j, j - 1] = -1
        V[:, j - 1] *= np.sqrt(j / (j + 1))
    return clr @ V  # (n, K-1)


def process_SB_features(X, measure_name):
    """
    Process state-based features for a given measure.

    The process involves applying a softmax function followed by an ILR transform.
    This is to ensure that the features are properly normalized and transformed for subsequent analysis.

    State-based feature vectors are compositional (non-negative and sum-to-one). We therefore analyze
    them in the Aitchison geometry and apply the isometric log-ratio (ILR) transformation (K−1 coordinates).
    The output has K−1 dimensions.
    """
    tau = 1.0  # temperature; 0.5–2.0 is typical

    X_transformed = None
    if measure_name in ["CAP", "Clustering"]:
        X_transformed = softmax(-X, tau=tau)
        # 2) ILR transform
        X_transformed = ilr_transform(X_transformed)
    elif measure_name in ["ContinuousHMM", "DiscreteHMM", "Windowless"]:
        X_transformed = ilr_transform(X)
    return X_transformed


def get_classification_scores(
    target,
    pred,
):
    """
    Get classification scores for a given target and predicted labels.
    Returns a dictionary with these metrics:
    - accuracy
    - balanced accuracy
    - recall
    - precision
    - f1 score
    - fp, fn, tp, tn
    - average precision
    """
    tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
    scores = {
        "accuracy": accuracy_score(target, pred),
        "balanced accuracy": balanced_accuracy_score(target, pred),
        "recall": recall_score(target, pred),
        "precision": precision_score(target, pred),
        "f1": f1_score(target, pred),
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "average precision": average_precision_score(target, pred),
    }
    return scores


def task_presence_classification(
    task,
    dFC_id,
    roi_root,
    dFC_root,
    run=None,
    session=None,
    dynamic_pred="no",
    normalize_dFC=False,
    train_test_ratio=0.8,
):
    """
    perform task presence classification using logistic regression, SVM, KNN, Random Forest, Gradient Boosting
    for a given task and dFC method and run.
    """
    if run is None:
        print(f"=============== {task} ===============")
    else:
        print(f"=============== {task} {run} ===============")

    if task == "task-restingstate":
        return

    SUBJECTS = find_available_subjects(
        dFC_root=dFC_root, task=task, run=run, session=session, dFC_id=dFC_id
    )

    # randomly select train_test_ratio of the subjects for training
    # and rest for testing using numpy.random.choice
    train_subjects = np.random.choice(
        SUBJECTS, int(train_test_ratio * len(SUBJECTS)), replace=False
    )
    test_subjects = np.setdiff1d(SUBJECTS, train_subjects)
    print(
        f"Number of train subjects: {len(train_subjects)} and test subjects: {len(test_subjects)}"
    )

    X_train, X_test, y_train, y_test, subj_label_train, subj_label_test, measure_name = (
        dFC_feature_extraction(
            task=task,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
            dFC_id=dFC_id,
            roi_root=roi_root,
            dFC_root=dFC_root,
            run=run,
            session=session,
            dynamic_pred=dynamic_pred,
            normalize_dFC=normalize_dFC,
            FCS_proba_for_SB=True,  # for state-based dFC features, we use FCS_proba
        )
    )
    measure_is_state_based = None
    if measure_name in ["SlidingWindow", "Time-Freq"]:
        measure_is_state_based = False
    elif measure_name in [
        "CAP",
        "Clustering",
        "ContinuousHMM",
        "DiscreteHMM",
        "Windowless",
    ]:
        measure_is_state_based = True
    else:
        # raise error
        raise ValueError(f"Unknown measure name: {measure_name}")

    if measure_is_state_based:
        X_train = process_SB_features(X=X_train, measure_name=measure_name)
        X_test = process_SB_features(X=X_test, measure_name=measure_name)

    # center the data by subject before embedding to remove subject effects
    # separately for train and test sets to avoid data leakage
    # for both state-based and state-free methods
    X_train = subject_center(X_train, subj_label_train, mode="demean")
    X_test = subject_center(X_test, subj_label_test, mode="demean")

    ML_scores = {
        "group_lvl": {
            "task": list(),
            "run": list(),
            "dFC method": list(),
            "embedding": list(),
            "group": list(),
            "SI": list(),
        },
        "subj_lvl": {
            "subj_id": list(),
            "group": list(),
            "SI": list(),
            "task": list(),
            "run": list(),
            "dFC method": list(),
            "embedding": list(),
        },
    }

    EMBEDDINGS = ["PCA", "PLS"]
    check_count = len(EMBEDDINGS)
    num_excluded_subjects = 0
    for embedding in EMBEDDINGS:
        if measure_is_state_based:
            embedding_to_use = None
        else:
            embedding_to_use = embedding

        # check if both classes are present in train and test sets
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(
                f"Only one class present in train or test sets for {embedding}. Skipping..."
            )
            check_count -= 1
            continue

        # task presence classification

        print("task presence classification ...")

        # logistic regression
        log_reg_RESULT = logistic_regression_classify(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            subj_label_train=subj_label_train,
            embedding_method=embedding_to_use,
        )

        # SVM
        SVM_RESULT = SVM_classify(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            subj_label_train=subj_label_train,
            embedding_method=embedding_to_use,
        )

        ML_models = {"Logistic regression": log_reg_RESULT, "SVM": SVM_RESULT}

        # Silhouette score
        # SI does not need to be separated for train and test sets
        # we will use the same SI for both train and test sets
        # using all samples from train and test sets
        # use the embedding and scaler trained in SVM_RESULT["model"]
        # so the results are comparable to the classification scores
        scaler = SVM_RESULT["model"].named_steps["scaler"]
        embedding_model = SVM_RESULT["model"].named_steps.get("emb", None)
        if embedding_model is not None:
            X_train_embedded = embedding_model.transform(scaler.transform(X_train))
            X_test_embedded = embedding_model.transform(scaler.transform(X_test))
        else:
            X_train_embedded = scaler.transform(X_train)
            X_test_embedded = scaler.transform(X_test)

        X_combined = np.concatenate((X_train_embedded, X_test_embedded), axis=0)
        y_combined = np.concatenate((y_train, y_test), axis=0)

        SI = {
            "train": silhouette_score(X_combined, y_combined),
            "test": silhouette_score(X_combined, y_combined),
        }

        # # permutation tests
        # permutation_scores = {
        #     "train": {},
        #     "test": {},
        # }
        # for model_name in ML_models:
        #     (
        #         permutation_train_scores,
        #         permutation_test_scores,
        #         p_value_train,
        #         p_value_test,
        #     ) = get_permutation_scores(
        #         X_train=X_train_embedded,
        #         y_train=y_train,
        #         X_test=X_test_embedded,
        #         y_test=y_test,
        #         classifier_model=ML_models[model_name]["model"],
        #         groups_train=subj_label_train,
        #         n_permutations=100,
        #     )
        #     permutation_scores["train"][
        #         f"{model_name} permutation p_value"
        #     ] = p_value_train
        #     permutation_scores["train"][f"{model_name} permutation score mean"] = np.mean(
        #         permutation_train_scores
        #     )
        #     permutation_scores["train"][f"{model_name} permutation score std"] = np.std(
        #         permutation_train_scores
        #     )
        #     permutation_scores["test"][f"{model_name} permutation p_value"] = p_value_test
        #     permutation_scores["test"][f"{model_name} permutation score mean"] = np.mean(
        #         permutation_test_scores
        #     )
        #     permutation_scores["test"][f"{model_name} permutation score std"] = np.std(
        #         permutation_test_scores
        #     )

        # group level scores
        for group in ["train", "test"]:

            ML_scores["group_lvl"]["group"].append(group)
            ML_scores["group_lvl"]["embedding"].append(embedding)
            ML_scores["group_lvl"]["task"].append(task)
            ML_scores["group_lvl"]["run"].append(run)
            ML_scores["group_lvl"]["dFC method"].append(measure_name)
            # SI
            ML_scores["group_lvl"]["SI"].append(SI[group])

            for model_name in ML_models:
                # accuracy score
                for metric in ML_models[model_name][group]:
                    if not f"{model_name} {metric}" in ML_scores["group_lvl"]:
                        ML_scores["group_lvl"][f"{model_name} {metric}"] = list()
                    ML_scores["group_lvl"][f"{model_name} {metric}"].append(
                        ML_models[model_name][group][metric]
                    )

            # # permutation test results
            # for key in permutation_scores[group]:
            #     if not key in ML_scores["group_lvl"]:
            #         ML_scores["group_lvl"][key] = list()
            #     ML_scores["group_lvl"][key].append(permutation_scores[group][key])

        # subject level scores
        for subj in SUBJECTS:
            if subj in train_subjects:
                subj_group = "train"
                features = X_train[subj_label_train == subj, :]
                target = y_train[subj_label_train == subj]
            elif subj in test_subjects:
                subj_group = "test"
                features = X_test[subj_label_test == subj, :]
                target = y_test[subj_label_test == subj]
            # check if only one class is present, skip the subject
            if len(np.unique(target)) < 2:
                num_excluded_subjects += 1
                continue
            ML_scores["subj_lvl"]["group"].append(subj_group)
            ML_scores["subj_lvl"]["subj_id"].append(subj)

            # Silhouette score
            ML_scores["subj_lvl"]["SI"].append(silhouette_score(features, target))
            # measure pred score using different metrics on each subj
            for model_name in ML_models:
                model = ML_models[model_name]["model"]
                pred = model.predict(features)
                scores = get_classification_scores(target=target, pred=pred)

                for metric in scores:
                    if not f"{model_name} {metric}" in ML_scores["subj_lvl"]:
                        ML_scores["subj_lvl"][f"{model_name} {metric}"] = list()
                    ML_scores["subj_lvl"][f"{model_name} {metric}"].append(scores[metric])

            ML_scores["subj_lvl"]["task"].append(task)
            ML_scores["subj_lvl"]["run"].append(run)
            ML_scores["subj_lvl"]["dFC method"].append(measure_name)
            ML_scores["subj_lvl"]["embedding"].append(embedding)

    # sanity check of the ML_scores
    L = None
    for key in ML_scores["group_lvl"]:
        if L is None:
            L = len(ML_scores["group_lvl"][key])
        else:
            assert (
                len(ML_scores["group_lvl"][key]) == L
            ), f"Length of {key} is not equal to others."

    # L is supposed to be equal to 3 embeddings (PCA, PLS, and LE) * 2 groups (train and test)
    assert (
        L == check_count * 2
    ), f"Length of group_lvl is not equal to {check_count * 2}, but {L}."

    L = None
    for key in ML_scores["subj_lvl"]:
        if L is None:
            L = len(ML_scores["subj_lvl"][key])
        else:
            assert (
                len(ML_scores["subj_lvl"][key]) == L
            ), f"Length of {key} is not equal to others."

    # L is supposed to be equal to number of subjects * 3 embeddings (PCA, PLS, and LE)
    assert (
        L == len(SUBJECTS) * check_count - num_excluded_subjects
    ), f"Length of subj_lvl is not equal to {len(SUBJECTS) * check_count - num_excluded_subjects}, but {L}."

    return ML_scores
