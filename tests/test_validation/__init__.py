"""
dFC Validation Framework

A comprehensive testing framework for dynamic functional connectivity methods
using synthetic data with known ground truth connectivity structure.

This module provides:
- SyntheticDataGenerator: Generate synthetic fMRI data with known block structure
- TestCase classes: Validate dFC output against ground truth
- DFCMethodWrapper: Standardized interface for dFC methods
- ValidationRunner: Execute tests on multiple methods
- Reporter: Summarize and display results

Example Usage
-------------
from test_validation import SyntheticDataGenerator, ZeroAndPerfectCorrTest, StepChangeTest
from test_validation.runner_reporter import ValidationRunner, Reporter

# Generate synthetic data
generator = SyntheticDataGenerator(n_subjects=50, n_regions=100, n_timepoints=600)
timeseries, ground_truth = generator.generate()

# Define tests
tests = [ZeroAndPerfectCorrTest(), StepChangeTest()]

# Run tests on a method (method must return dFC output with shape [n_subjects, n_timepoints, n_regions, n_regions])
runner = ValidationRunner()
results = runner.run(methods={"my_method": my_method_wrapper}, test_cases=tests,
                     timeseries=timeseries, ground_truth=ground_truth)

# Report results
reporter = Reporter()
reporter.generate_report(results)
"""

from .dfc_method_wrappers import (
    CAPWrapper,
    ContinuousHMMWrapper,
    DFCMethodWrapper,
    DiscreteHMMWrapper,
    DummyMethod,
    PydfcMethodWrapper,
    SlidingWindowClustrWrapper,
    SlidingWindowWrapper,
    TimeFreqWrapper,
    WindowlessWrapper,
    get_available_methods,
    get_method_availability,
    list_registered_methods,
    resolve_method_requests,
)
from .runner_reporter import Reporter, ValidationRunner
from .synthetic_data import (
    SegmentGroundTruth,
    SyntheticDataGenerator,
    SyntheticGroundTruth,
)
from .test_cases import StepChangeTest, TestCase, TestResult, ZeroAndPerfectCorrTest

__all__ = [
    # Synthetic data
    "SyntheticDataGenerator",
    "SyntheticGroundTruth",
    "SegmentGroundTruth",
    # Test cases
    "TestCase",
    "TestResult",
    "ZeroAndPerfectCorrTest",
    "StepChangeTest",
    # Method wrappers
    "DFCMethodWrapper",
    "PydfcMethodWrapper",
    "SlidingWindowWrapper",
    "TimeFreqWrapper",
    "CAPWrapper",
    "ContinuousHMMWrapper",
    "DiscreteHMMWrapper",
    "WindowlessWrapper",
    "SlidingWindowClustrWrapper",
    "DummyMethod",
    "get_available_methods",
    "get_method_availability",
    "resolve_method_requests",
    "list_registered_methods",
    # Runner and reporter
    "ValidationRunner",
    "Reporter",
]
