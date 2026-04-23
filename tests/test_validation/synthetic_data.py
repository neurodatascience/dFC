"""
Synthetic Data Generator for dFC Validation

This module generates synthetic fMRI time series with known connectivity structure
for validating dFC methods. The synthetic data is designed with ground truth block
structure where within-block regions have perfect correlation and between-block
regions have zero correlation.

Created for dFC validation framework
@author: Copilot
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SegmentGroundTruth:
    """Ground truth information for a single segment."""

    start: int
    end: int
    eval_start: int
    eval_end: int
    n_blocks: int
    region_block_ids: np.ndarray  # [n_regions] array of block IDs
    perfect_corr_mask: np.ndarray  # [n_regions × n_regions] boolean
    zero_corr_mask: np.ndarray  # [n_regions × n_regions] boolean


@dataclass
class SyntheticGroundTruth:
    """Complete ground truth metadata for synthetic dataset."""

    segments: List[SegmentGroundTruth]
    n_subjects: int
    n_regions: int
    n_timepoints: int
    TR: float
    noise_floor: float = 0.01
    random_seed: int = 42

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "segments": [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "eval_start": seg.eval_start,
                    "eval_end": seg.eval_end,
                    "n_blocks": seg.n_blocks,
                    "region_block_ids": seg.region_block_ids.tolist(),
                    "perfect_corr_pairs": np.argwhere(seg.perfect_corr_mask).tolist(),
                    "zero_corr_pairs": np.argwhere(seg.zero_corr_mask).tolist(),
                }
                for seg in self.segments
            ],
            "n_subjects": self.n_subjects,
            "n_regions": self.n_regions,
            "n_timepoints": self.n_timepoints,
            "TR": self.TR,
            "noise_floor": self.noise_floor,
            "random_seed": self.random_seed,
        }


class SyntheticDataGenerator:
    """
    Generates synthetic fMRI time series with known block structure.

    The synthetic data is organized into segments, each with a different block
    structure. Within-block regions are perfectly correlated, while between-block
    regions have zero correlation.

    Parameters
    ----------
    n_subjects : int, default=50
        Number of subjects in the synthetic dataset
    n_regions : int, default=100
        Number of brain regions
    n_timepoints : int, default=600
        Total number of timepoints
    TR : float, default=1.0
        Repetition time in seconds
    segment_n_blocks : List[int], optional
        Number of blocks for each segment. If None, defaults to [5, 7, 10]
    noise_floor : float, default=0.01
        Small noise floor for numerical realism
    random_seed : int, default=42
        Random seed for reproducibility
    signal_type : str, default="gaussian_smooth"
        Type of signal to generate: "gaussian_smooth" or "white_noise"
    """

    def __init__(
        self,
        n_subjects: int = 50,
        n_regions: int = 100,
        n_timepoints: int = 600,
        TR: float = 1.0,
        segment_n_blocks: List[int] = None,
        noise_floor: float = 0.01,
        random_seed: int = 42,
        signal_type: str = "gaussian_smooth",
    ):
        self.n_subjects = n_subjects
        self.n_regions = n_regions
        self.n_timepoints = n_timepoints
        self.TR = TR
        self.noise_floor = noise_floor
        self.random_seed = random_seed
        self.signal_type = signal_type

        if segment_n_blocks is None:
            segment_n_blocks = [5, 7, 10]
        self.segment_n_blocks = segment_n_blocks
        self.n_segments = len(segment_n_blocks)

        # Calculate segment boundaries
        segment_length = n_timepoints // self.n_segments
        self.segment_boundaries = [
            (
                (i * segment_length, (i + 1) * segment_length)
                if i < self.n_segments - 1
                else (i * segment_length, n_timepoints)
            )
            for i in range(self.n_segments)
        ]

    def _generate_block_assignments(
        self, n_regions: int, n_blocks: int, random_state: np.random.RandomState
    ) -> np.ndarray:
        """
        Generate random block assignments for regions.

        Uses a Dirichlet distribution to generate unequal block sizes, then
        shuffles regions randomly into these blocks.

        Parameters
        ----------
        n_regions : int
            Number of regions
        n_blocks : int
            Number of blocks
        random_state : np.random.RandomState
            Random state for reproducibility

        Returns
        -------
        region_block_ids : np.ndarray
            Array of shape [n_regions] where each element is the block ID
            for that region.
        """
        # Generate unequal block sizes using Dirichlet distribution
        block_sizes = random_state.dirichlet(np.ones(n_blocks))
        block_sizes = np.round(block_sizes * n_regions).astype(int)

        # Adjust for rounding errors
        diff = n_regions - block_sizes.sum()
        if diff > 0:
            block_sizes[0] += diff
        elif diff < 0:
            # Remove from largest blocks
            for _ in range(-diff):
                largest_idx = np.argmax(block_sizes)
                if block_sizes[largest_idx] > 0:
                    block_sizes[largest_idx] -= 1

        # Create block assignments
        region_block_ids = np.repeat(np.arange(n_blocks), block_sizes)
        assert len(region_block_ids) == n_regions

        # Shuffle the assignment
        random_state.shuffle(region_block_ids)

        return region_block_ids

    def _create_correlation_masks(
        self, region_block_ids: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create perfect and zero correlation masks from block assignments.

        Parameters
        ----------
        region_block_ids : np.ndarray
            Array of shape [n_regions] with block ID for each region

        Returns
        -------
        perfect_corr_mask : np.ndarray
            Boolean array [n_regions × n_regions], True for within-block pairs
        zero_corr_mask : np.ndarray
            Boolean array [n_regions × n_regions], True for between-block pairs
        """
        n_regions = len(region_block_ids)
        perfect_corr_mask = np.zeros((n_regions, n_regions), dtype=bool)
        zero_corr_mask = np.zeros((n_regions, n_regions), dtype=bool)

        for i in range(n_regions):
            for j in range(n_regions):
                if region_block_ids[i] == region_block_ids[j]:
                    perfect_corr_mask[i, j] = True
                else:
                    zero_corr_mask[i, j] = True

        return perfect_corr_mask, zero_corr_mask

    def _generate_block_signals(
        self, segment_length: int, n_blocks: int, random_state: np.random.RandomState
    ) -> Dict[int, np.ndarray]:
        """
        Generate random signals for each block.

        Parameters
        ----------
        segment_length : int
            Length of the segment in timepoints
        n_blocks : int
            Number of blocks
        random_state : np.random.RandomState
            Random state for reproducibility

        Returns
        -------
        block_signals : Dict[int, np.ndarray]
            Dictionary mapping block ID to signal array of shape [segment_length]
        """
        block_signals = {}

        for block_id in range(n_blocks):
            if self.signal_type == "gaussian_smooth":
                # Generate smooth signal using Gaussian filtering
                raw_signal = random_state.randn(segment_length)
                # Apply simple smoothing with a Gaussian kernel
                from scipy.ndimage import gaussian_filter1d

                smooth_signal = gaussian_filter1d(raw_signal, sigma=2.0)
                block_signals[block_id] = smooth_signal
            elif self.signal_type == "white_noise":
                block_signals[block_id] = random_state.randn(segment_length)
            else:
                raise ValueError(f"Unknown signal_type: {self.signal_type}")

        return block_signals

    def generate(self) -> Tuple[np.ndarray, SyntheticGroundTruth]:
        """
        Generate synthetic fMRI time series with known block structure.

        Returns
        -------
        timeseries : np.ndarray
            Shape [n_subjects, n_timepoints, n_regions]
            The synthetic fMRI time series
        ground_truth : SyntheticGroundTruth
            Ground truth metadata including block assignments and correlation masks
        """
        rng = np.random.RandomState(self.random_seed)

        # Initialize output array
        timeseries = np.zeros(
            (self.n_subjects, self.n_timepoints, self.n_regions), dtype=np.float32
        )

        segments_gt = []

        for seg_idx, (start, end) in enumerate(self.segment_boundaries):
            segment_length = end - start
            n_blocks = self.segment_n_blocks[seg_idx]

            # Generate unique block assignments for this segment
            region_block_ids = self._generate_block_assignments(
                self.n_regions, n_blocks, rng
            )
            perfect_corr_mask, zero_corr_mask = self._create_correlation_masks(
                region_block_ids
            )

            # Define evaluation window (middle 100 timepoints)
            eval_window = segment_length // 2 - 50
            eval_start = start + eval_window
            eval_end = eval_start + 100

            # Generate block signals once per segment (will be reused across subjects
            # with different noise realizations)
            block_signals = self._generate_block_signals(segment_length, n_blocks, rng)

            # Fill timeseries for each subject
            for subj_idx in range(self.n_subjects):
                for region_idx in range(self.n_regions):
                    block_id = region_block_ids[region_idx]
                    # Get the block signal
                    block_signal = block_signals[block_id].copy()
                    # Add independent noise per subject
                    noise = self.noise_floor * rng.randn(segment_length)
                    timeseries[subj_idx, start:end, region_idx] = block_signal + noise

            # Store ground truth for this segment
            segments_gt.append(
                SegmentGroundTruth(
                    start=start,
                    end=end,
                    eval_start=eval_start,
                    eval_end=eval_end,
                    n_blocks=n_blocks,
                    region_block_ids=region_block_ids.copy(),
                    perfect_corr_mask=perfect_corr_mask,
                    zero_corr_mask=zero_corr_mask,
                )
            )

        ground_truth = SyntheticGroundTruth(
            segments=segments_gt,
            n_subjects=self.n_subjects,
            n_regions=self.n_regions,
            n_timepoints=self.n_timepoints,
            TR=self.TR,
            noise_floor=self.noise_floor,
            random_seed=self.random_seed,
        )

        return timeseries, ground_truth
