"""
Test Cases for dFC Validation

This module implements concrete test cases that validate dFC methods
against synthetic data with known ground truth connectivity structure.

Created for dFC validation framework
@author: Copilot
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Result of a single test case evaluation."""

    test_name: str
    method_name: str
    passed: bool
    score: float  # scalar in [-1, 1] or [0, 1]
    per_subject_scores: List[float]  # one score per subject for diagnostics
    details: str  # human-readable explanation of failure or pass


class TestCase(ABC):
    """
    Abstract base class for dFC validation test cases.

    Each test case evaluates whether a dFC method produces outputs consistent
    with known ground truth connectivity structure.
    """

    def __init__(self, name: str, description: str, pass_threshold: float = 0.9):
        """
        Initialize a test case.

        Parameters
        ----------
        name : str
            Short name for the test case
        description : str
            Detailed description of what the test validates
        pass_threshold : float, default=0.9
            Score threshold above which the test is considered passed
        """
        self.name = name
        self.description = description
        self.pass_threshold = pass_threshold

    @abstractmethod
    def evaluate(self, dfc_output, ground_truth) -> TestResult:
        """
        Evaluate dFC output against ground truth.

        Parameters
        ----------
        dfc_output : variable
            Output from a dFC method (shape/format depends on method)
        ground_truth : SyntheticGroundTruth
            Ground truth metadata with block assignments and expected correlations

        Returns
        -------
        TestResult
            Result object containing pass/fail decision and detailed scores
        """
        raise NotImplementedError


def _average_segment_connectivity(
    dfc_subject: np.ndarray, start: int, end: int
) -> np.ndarray:
    """Average a subject's dFC matrices over a segment, inferring the time axis."""
    if dfc_subject.ndim != 3:
        raise ValueError(
            f"Unexpected dFC shape: {dfc_subject.shape}. Expected a 3D array for one subject."
        )

    if dfc_subject.shape[0] >= end and dfc_subject.shape[1] == dfc_subject.shape[2]:
        segment = dfc_subject[start:end, :, :]
        return np.nanmean(np.abs(segment), axis=0)

    if dfc_subject.shape[2] >= end and dfc_subject.shape[0] == dfc_subject.shape[1]:
        segment = dfc_subject[:, :, start:end]
        return np.nanmean(np.abs(segment), axis=2)

    raise ValueError(
        f"Cannot infer time axis from dFC shape {dfc_subject.shape}. "
        "Expected either [time, regions, regions] or [regions, regions, time]."
    )


class ZeroAndPerfectCorrTest(TestCase):
    """
    Test that dFC correctly separates zero-corr from perfect-corr pairs.

    This test uses rank-biserial correlation between binary labels (zero-corr vs
    perfect-corr pairs) and the ranked absolute connectivity values. It is
    agnostic to the absolute range of dFC values (e.g., Fisher-z, coherence, etc).
    """

    def __init__(
        self,
        name: str = "ZeroAndPerfectCorr",
        description: str = "Rank-based separation of zero and perfect correlation pairs",
        pass_threshold: float = 0.9,
    ):
        super().__init__(name, description, pass_threshold)

    def _rank_biserial_correlation(
        self, binary_labels: np.ndarray, values: np.ndarray
    ) -> float:
        """
        Compute rank-biserial correlation between binary labels and ranked values.

        Parameters
        ----------
        binary_labels : np.ndarray
            Binary labels (0 or 1) indicating group membership
        values : np.ndarray
            Continuous values to rank

        Returns
        -------
        float
            Rank-biserial correlation in range [-1, 1]
        """
        # Use Mann-Whitney U test to compute rank-biserial
        group_0 = values[binary_labels == 0]
        group_1 = values[binary_labels == 1]

        if len(group_0) == 0 or len(group_1) == 0:
            return 0.0

        # Compute Mann-Whitney U statistic
        U, _ = stats.mannwhitneyu(group_1, group_0, alternative="two-sided")

        # Convert U to rank-biserial correlation. Since U is computed as
        # mannwhitneyu(group_1, group_0), higher group_1 values should yield
        # positive correlation.
        n0 = len(group_0)
        n1 = len(group_1)
        r = (2.0 * U) / (n0 * n1) - 1.0

        return float(r)

    def evaluate(self, dfc_output, ground_truth) -> TestResult:
        """
        Evaluate zero and perfect correlation separation.

        Parameters
        ----------
        dfc_output : dict or np.ndarray
            Output from dFC method. Expected to be dict with keys per subject,
            or np.ndarray of shape [n_subjects, n_timepoints, n_regions, n_regions]
        ground_truth : SyntheticGroundTruth
            Ground truth with connectivity masks

        Returns
        -------
        TestResult
            Test result with per-subject and aggregate scores
        """
        n_subjects = ground_truth.n_subjects
        per_subject_scores = []

        # Collect dFC matrices for all subjects
        if isinstance(dfc_output, dict):
            # Extract matrices by subject ID
            dfc_matrices = []
            for subj_idx in range(n_subjects):
                key = (
                    f"sub_{subj_idx:03d}"
                    if f"sub_{subj_idx:03d}" in dfc_output
                    else subj_idx
                )
                if key in dfc_output:
                    dfc_matrices.append(dfc_output[key])
        elif isinstance(dfc_output, np.ndarray):
            dfc_matrices = dfc_output
        else:
            raise ValueError(
                f"Unexpected dfc_output type: {type(dfc_output)}. "
                "Expected dict or np.ndarray"
            )

        # Ensure dfc_matrices has correct shape
        if len(dfc_matrices) != n_subjects:
            raise ValueError(
                f"Number of subjects in output ({len(dfc_matrices)}) "
                f"does not match ground truth ({n_subjects})"
            )

        segment_scores = []

        # Evaluate each segment
        for seg_idx, segment_gt in enumerate(ground_truth.segments):
            start = segment_gt.eval_start
            end = segment_gt.eval_end

            # Collect scores for this segment across all subjects
            for subj_idx in range(n_subjects):
                # Extract dFC for this subject in this segment
                if isinstance(dfc_output, dict):
                    key = (
                        f"sub_{subj_idx:03d}"
                        if f"sub_{subj_idx:03d}" in dfc_output
                        else subj_idx
                    )
                    dfc_subject = dfc_output[key]
                else:
                    dfc_subject = dfc_output[subj_idx]

                avg_conn = _average_segment_connectivity(dfc_subject, start, end)

                # Get upper triangle (to avoid redundancy and self-correlation)
                upper_triangle_indices = np.triu_indices(avg_conn.shape[0], k=1)
                connectivity_values = avg_conn[upper_triangle_indices]

                # Create binary labels: 1 for perfect-corr pairs, 0 for zero-corr
                perfect_pairs = segment_gt.perfect_corr_mask[upper_triangle_indices]
                zero_pairs = segment_gt.zero_corr_mask[upper_triangle_indices]

                # Only use pairs that are either perfect-corr or zero-corr
                valid_mask = perfect_pairs | zero_pairs
                binary_labels = perfect_pairs[valid_mask].astype(int)
                connectivity_subset = connectivity_values[valid_mask]
                finite_mask = np.isfinite(connectivity_subset)
                connectivity_subset = connectivity_subset[finite_mask]
                binary_labels = binary_labels[finite_mask]

                # Compute rank-biserial correlation
                if len(binary_labels) > 0 and len(np.unique(binary_labels)) > 1:
                    score = self._rank_biserial_correlation(
                        binary_labels, connectivity_subset
                    )
                else:
                    score = 0.0

                segment_scores.append(score)

        # Aggregate score: average across all subjects and segments
        if len(segment_scores) > 0:
            avg_score = float(np.mean(segment_scores))
        else:
            avg_score = 0.0

        per_subject_scores = segment_scores

        # Determine pass/fail
        passed = avg_score > self.pass_threshold
        details = (
            f"Rank-biserial correlation scores across {n_subjects} subjects and "
            f"{len(ground_truth.segments)} segments. "
            f"Mean score: {avg_score:.3f}, threshold: {self.pass_threshold}. "
            f"{'PASS' if passed else 'FAIL'}"
        )

        return TestResult(
            test_name=self.name,
            method_name="",  # to be filled by runner
            passed=passed,
            score=avg_score,
            per_subject_scores=per_subject_scores,
            details=details,
        )


class StepChangeTest(TestCase):
    """
    Test that dFC connectivity patterns change appropriately between segments.

    This test checks that pairs which are perfect-corr in one segment and
    zero-corr in another show appropriate rank changes between segments.
    """

    def __init__(
        self,
        name: str = "StepChange",
        description: str = "Connectivity changes between segments with different block structures",
        pass_threshold: float = 0.9,
    ):
        super().__init__(name, description, pass_threshold)

    def _rank_biserial_correlation(
        self, binary_labels: np.ndarray, values: np.ndarray
    ) -> float:
        """
        Compute rank-biserial correlation between binary labels and ranked values.

        Parameters
        ----------
        binary_labels : np.ndarray
            Binary labels (0 or 1) indicating group membership
        values : np.ndarray
            Continuous values to rank

        Returns
        -------
        float
            Rank-biserial correlation in range [-1, 1]
        """
        group_0 = values[binary_labels == 0]
        group_1 = values[binary_labels == 1]

        if len(group_0) == 0 or len(group_1) == 0:
            return 0.0

        U, _ = stats.mannwhitneyu(group_1, group_0, alternative="two-sided")
        n0 = len(group_0)
        n1 = len(group_1)
        r = (2.0 * U) / (n0 * n1) - 1.0

        return float(r)

    def evaluate(self, dfc_output, ground_truth) -> TestResult:
        """
        Evaluate connectivity changes between segments.

        Parameters
        ----------
        dfc_output : dict or np.ndarray
            Output from dFC method (same format as ZeroAndPerfectCorrTest)
        ground_truth : SyntheticGroundTruth
            Ground truth with connectivity masks

        Returns
        -------
        TestResult
            Test result with per-subject scores
        """
        n_subjects = ground_truth.n_subjects
        n_segments = len(ground_truth.segments)

        if n_segments < 2:
            return TestResult(
                test_name=self.name,
                method_name="",
                passed=False,
                score=0.0,
                per_subject_scores=[],
                details="Insufficient segments for step change test (need ≥2)",
            )

        # Collect dFC matrices for all subjects
        if isinstance(dfc_output, dict):
            dfc_matrices = {}
            for subj_idx in range(n_subjects):
                key = (
                    f"sub_{subj_idx:03d}"
                    if f"sub_{subj_idx:03d}" in dfc_output
                    else subj_idx
                )
                if key in dfc_output:
                    dfc_matrices[key] = dfc_output[key]
        elif isinstance(dfc_output, np.ndarray):
            dfc_matrices = dfc_output
        else:
            raise ValueError(
                f"Unexpected dfc_output type: {type(dfc_output)}. "
                "Expected dict or np.ndarray"
            )

        segment_scores = []

        # Compare adjacent segment pairs
        for seg_pair_idx in range(n_segments - 1):
            seg_a = ground_truth.segments[seg_pair_idx]
            seg_b = ground_truth.segments[seg_pair_idx + 1]

            for subj_idx in range(n_subjects):
                # Get subject's dFC data
                if isinstance(dfc_output, dict):
                    key = (
                        f"sub_{subj_idx:03d}"
                        if f"sub_{subj_idx:03d}" in dfc_output
                        else subj_idx
                    )
                    dfc_subject = dfc_output[key]
                else:
                    dfc_subject = dfc_output[subj_idx]

                avg_a = _average_segment_connectivity(
                    dfc_subject, seg_a.eval_start, seg_a.eval_end
                )
                avg_b = _average_segment_connectivity(
                    dfc_subject, seg_b.eval_start, seg_b.eval_end
                )

                # Get upper triangle
                upper_triangle_indices = np.triu_indices(avg_a.shape[0], k=1)

                # Identify transition pairs
                pairs_perfect_a_zero_b = (seg_a.perfect_corr_mask & seg_b.zero_corr_mask)[
                    upper_triangle_indices
                ]
                pairs_zero_a_perfect_b = (seg_a.zero_corr_mask & seg_b.perfect_corr_mask)[
                    upper_triangle_indices
                ]

                # For pairs that should drop in rank (perfect→zero)
                if np.any(pairs_perfect_a_zero_b):
                    conn_a_drop = avg_a[upper_triangle_indices][pairs_perfect_a_zero_b]
                    conn_b_drop = avg_b[upper_triangle_indices][pairs_perfect_a_zero_b]
                    finite_mask_drop = np.isfinite(conn_a_drop) & np.isfinite(conn_b_drop)
                    conn_a_drop = conn_a_drop[finite_mask_drop]
                    conn_b_drop = conn_b_drop[finite_mask_drop]
                    # Expect conn_a > conn_b, so score based on rank in segment A
                    values_drop = np.concatenate([conn_a_drop, conn_b_drop])
                    labels_drop_all = np.concatenate(
                        [np.ones(len(conn_a_drop)), np.zeros(len(conn_b_drop))]
                    )
                    score_drop = self._rank_biserial_correlation(
                        labels_drop_all, values_drop
                    )
                else:
                    score_drop = 0.0

                # For pairs that should rise in rank (zero→perfect)
                if np.any(pairs_zero_a_perfect_b):
                    conn_a_rise = avg_a[upper_triangle_indices][pairs_zero_a_perfect_b]
                    conn_b_rise = avg_b[upper_triangle_indices][pairs_zero_a_perfect_b]
                    finite_mask_rise = np.isfinite(conn_a_rise) & np.isfinite(conn_b_rise)
                    conn_a_rise = conn_a_rise[finite_mask_rise]
                    conn_b_rise = conn_b_rise[finite_mask_rise]
                    # Expect conn_b > conn_a, so score based on rank in segment B
                    values_rise = np.concatenate([conn_b_rise, conn_a_rise])
                    labels_rise_all = np.concatenate(
                        [np.ones(len(conn_b_rise)), np.zeros(len(conn_a_rise))]
                    )
                    score_rise = self._rank_biserial_correlation(
                        labels_rise_all, values_rise
                    )
                else:
                    score_rise = 0.0

                # Average the two directional scores
                if np.any(pairs_perfect_a_zero_b) and np.any(pairs_zero_a_perfect_b):
                    avg_score = (score_drop + score_rise) / 2.0
                elif np.any(pairs_perfect_a_zero_b):
                    avg_score = score_drop
                elif np.any(pairs_zero_a_perfect_b):
                    avg_score = score_rise
                else:
                    avg_score = 0.0

                segment_scores.append(avg_score)

        # Aggregate score
        if len(segment_scores) > 0:
            avg_score = float(np.mean(segment_scores))
        else:
            avg_score = 0.0

        passed = avg_score > self.pass_threshold
        details = (
            f"Rank-based step change scores across {n_subjects} subjects and "
            f"{n_segments - 1} segment transitions. "
            f"Mean score: {avg_score:.3f}, threshold: {self.pass_threshold}. "
            f"{'PASS' if passed else 'FAIL'}"
        )

        return TestResult(
            test_name=self.name,
            method_name="",
            passed=passed,
            score=avg_score,
            per_subject_scores=segment_scores,
            details=details,
        )
