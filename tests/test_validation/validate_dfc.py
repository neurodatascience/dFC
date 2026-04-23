"""
Main Validation Script for dFC Methods

This script orchestrates the complete dFC validation pipeline:
1. Generate synthetic data with known ground truth
2. Run dFC methods on the synthetic data
3. Evaluate methods against test cases
4. Generate and display results

Usage
-----
python validate_dfc.py [--n-subjects N] [--n-regions R] [--n-timepoints T] [--methods METHOD1 METHOD2 ...]

Example
-------
python validate_dfc.py --n-subjects 50 --n-regions 100 --n-timepoints 600

Created for dFC validation framework
@author: Copilot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from .dfc_method_wrappers import (
    get_available_methods,
    get_method_availability,
    get_method_catalog,
    list_registered_methods,
    resolve_method_requests,
)
from .runner_reporter import Reporter, ValidationRunner
from .synthetic_data import SyntheticDataGenerator, SyntheticGroundTruth
from .test_cases import StepChangeTest, TestCase, ZeroAndPerfectCorrTest


def create_test_suite(pass_threshold: float = 0.9) -> List[TestCase]:
    """
    Create the standard test suite for dFC validation.

    Parameters
    ----------
    pass_threshold : float, default=0.9
        Pass threshold for all tests

    Returns
    -------
    List[TestCase]
        List of test cases
    """
    return [
        ZeroAndPerfectCorrTest(pass_threshold=pass_threshold),
        StepChangeTest(pass_threshold=pass_threshold),
    ]


def create_synthetic_dataset(
    n_subjects: int = 50,
    n_regions: int = 100,
    n_timepoints: int = 600,
    noise_floor: float = 0.01,
    random_seed: int = 42,
) -> tuple:
    """
    Generate synthetic dFC validation dataset.

    Parameters
    ----------
    n_subjects : int, default=50
        Number of subjects
    n_regions : int, default=100
        Number of brain regions
    n_timepoints : int, default=600
        Total number of timepoints
    noise_floor : float, default=0.01
        Noise floor for numerical realism
    random_seed : int, default=42
        Random seed

    Returns
    -------
    timeseries : np.ndarray
        Synthetic timeseries [n_subjects, n_timepoints, n_regions]
    ground_truth : SyntheticGroundTruth
        Ground truth metadata
    """
    if n_regions <= 20:
        seg_n_blocks = [3, 4, 2]
    else:
        seg_n_blocks = [5, 7, 10]
    generator = SyntheticDataGenerator(
        n_subjects=n_subjects,
        n_regions=n_regions,
        n_timepoints=n_timepoints,
        TR=1.0,
        segment_n_blocks=seg_n_blocks,
        noise_floor=noise_floor,
        random_seed=random_seed,
        signal_type="gaussian_smooth",
    )

    print("Generating synthetic dataset...")
    timeseries, ground_truth = generator.generate()

    print(f"  Generated: {timeseries.shape}")
    print(f"  Subjects: {ground_truth.n_subjects}")
    print(f"  Regions: {ground_truth.n_regions}")
    print(f"  Timepoints: {ground_truth.n_timepoints}")
    print(f"  Segments: {len(ground_truth.segments)}")
    for i, seg in enumerate(ground_truth.segments):
        print(
            f"    Segment {i}: {seg.n_blocks} blocks, eval window [{seg.eval_start}:{seg.eval_end}]"
        )

    return timeseries, ground_truth


def main(args=None):
    """
    Main entry point for validation pipeline.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Command-line arguments. If None, uses sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Validate dFC methods using synthetic data"
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=50,
        help="Number of subjects in synthetic dataset",
    )
    parser.add_argument(
        "--n-regions",
        type=int,
        default=100,
        help="Number of brain regions",
    )
    parser.add_argument(
        "--n-timepoints",
        type=int,
        default=600,
        help="Total number of timepoints",
    )
    parser.add_argument(
        "--noise-floor",
        type=float,
        default=0.01,
        help="Noise floor for synthetic data",
    )
    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=0.9,
        help="Pass threshold for tests",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Specific methods to test (default: all available)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List registered methods with availability and exit",
    )

    parsed_args = parser.parse_args(args)

    print("=" * 80)
    print("DFC VALIDATION FRAMEWORK")
    print("=" * 80)

    if parsed_args.list_methods:
        print("\nRegistered methods:")
        for entry in get_method_catalog():
            status = "AVAILABLE" if entry["available"] else "UNAVAILABLE"
            aliases = ", ".join(entry["aliases"]) if entry["aliases"] else "-"
            print(f"  [{entry['id']}] {entry['key']} -> {status}")
            print(f"      aliases: {aliases}")
            if not entry["available"]:
                print(f"      reason: {entry['reason']}")
        return 0

    # Step 1: Generate synthetic dataset
    print("\n[1/4] Generating synthetic dataset...")
    timeseries, ground_truth = create_synthetic_dataset(
        n_subjects=parsed_args.n_subjects,
        n_regions=parsed_args.n_regions,
        n_timepoints=parsed_args.n_timepoints,
        noise_floor=parsed_args.noise_floor,
        random_seed=parsed_args.seed,
    )

    # Step 2: Create test suite
    print("\n[2/4] Creating test suite...")
    test_cases = create_test_suite(pass_threshold=parsed_args.pass_threshold)
    print(f"  Tests: {[t.name for t in test_cases]}")

    # Step 3: Get methods to test
    print("\n[3/4] Loading dFC methods...")
    all_methods, unavailable_methods = get_method_availability()

    if unavailable_methods:
        print("  Some registered methods are unavailable in this environment:")
        for method_name, reason in unavailable_methods.items():
            print(f"    - {method_name}: {reason}")

    if parsed_args.methods:
        methods_to_test, missing, unavailable_selected = resolve_method_requests(
            parsed_args.methods
        )
        if missing:
            print(f"  WARNING: Methods not found: {missing}")
            print(f"  Registered methods: {list_registered_methods()}")
        if unavailable_selected:
            print("  WARNING: Requested methods unavailable in this environment:")
            for method_name, reason in unavailable_selected.items():
                print(f"    - {method_name}: {reason}")
    else:
        print("  Default selection uses runnable methods only in this environment.")
        methods_to_test = all_methods

    print(f"  Methods to test: {list(methods_to_test.keys())}")

    if len(methods_to_test) == 0:
        print("  ERROR: No runnable methods selected.")
        return 2

    # Step 4: Run validation
    print("\n[4/4] Running validation tests...")
    runner = ValidationRunner(verbose=parsed_args.verbose)
    results = runner.run(
        methods=methods_to_test,
        test_cases=test_cases,
        timeseries=timeseries,
        ground_truth=ground_truth,
    )

    # Report results
    print("\n" + "=" * 80)
    reporter = Reporter(output_dir=parsed_args.output_dir)
    reporter.generate_report(results, save_json=True)

    # Return exit code based on results
    if len(results) == 0:
        return 2

    all_passed = all(r.passed for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
