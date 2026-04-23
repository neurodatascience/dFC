"""
Runner and Reporter for dFC Validation

This module implements the main runner that executes tests and reporters
that summarize and display results.

Created for dFC validation framework
@author: Copilot
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .dfc_method_wrappers import DFCMethodWrapper
from .synthetic_data import SyntheticGroundTruth
from .test_cases import TestCase, TestResult


class ValidationRunner:
    """
    Run dFC validation tests on multiple methods.

    Coordinates the execution of test cases on dFC methods using synthetic data.
    """

    def __init__(self, verbose: int = 1):
        """
        Initialize the runner.

        Parameters
        ----------
        verbose : int, default=1
            Verbosity level (0=silent, 1=normal, 2=detailed)
        """
        self.verbose = verbose
        self.results: List[TestResult] = []

    def run(
        self,
        methods: Dict[str, DFCMethodWrapper],
        test_cases: List[TestCase],
        timeseries: np.ndarray,
        ground_truth: SyntheticGroundTruth,
    ) -> List[TestResult]:
        """
        Run all test cases on all methods.

        Parameters
        ----------
        methods : Dict[str, DFCMethodWrapper]
            Dictionary of method name -> method wrapper
        test_cases : List[TestCase]
            List of test cases to run
        timeseries : np.ndarray
            Synthetic timeseries of shape [n_subjects, n_timepoints, n_regions]
        ground_truth : SyntheticGroundTruth
            Ground truth metadata

        Returns
        -------
        List[TestResult]
            Results from all method × test combinations
        """
        self.results = []

        # Run each method
        for method_name, method in methods.items():
            if self.verbose >= 1:
                print(f"\n{'='*60}")
                print(f"Running method: {method_name}")
                print(f"{'='*60}")

            # Run the method to get dFC output
            try:
                if self.verbose >= 2:
                    print("  Executing dFC estimation...")
                dfc_output = method.run(timeseries)
                if self.verbose >= 2:
                    print(
                        f"  Output shape: {dfc_output.shape if hasattr(dfc_output, 'shape') else 'dict'}"
                    )
            except Exception as e:
                print(f"  ERROR: Failed to run method: {e}")
                continue

            # Run each test case
            for test_case in test_cases:
                if self.verbose >= 2:
                    print(f"  Running test: {test_case.name}...")

                try:
                    result = test_case.evaluate(dfc_output, ground_truth)
                    result.method_name = method_name
                    self.results.append(result)

                    status = "PASS" if result.passed else "FAIL"
                    print(f"    {test_case.name}: {status} (score={result.score:.3f})")

                except Exception as e:
                    print(f"    ERROR in {test_case.name}: {e}")
                    self.results.append(
                        TestResult(
                            test_name=test_case.name,
                            method_name=method_name,
                            passed=False,
                            score=0.0,
                            per_subject_scores=[],
                            details=f"Test raised an exception: {e}",
                        )
                    )

        return self.results


class Reporter:
    """
    Generate reports of validation results.

    Produces summary tables and detailed logs of validation test results.
    """

    def __init__(self, output_dir: str = "./validation_results"):
        """
        Initialize the reporter.

        Parameters
        ----------
        output_dir : str
            Directory to save report files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _format_table(self, headers: List[str], rows: List[List[str]]) -> str:
        widths = [len(str(header)) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(str(cell)))

        def format_row(row: List[str]) -> str:
            return " | ".join(
                str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)
            )

        separator = "-+-".join("-" * width for width in widths)
        lines = [format_row(headers), separator]
        lines.extend(format_row(row) for row in rows)
        return "\n".join(lines)

    def print_summary(self, results: List[TestResult]) -> None:
        """
        Print a summary table of results.

        Parameters
        ----------
        results : List[TestResult]
            List of test results
        """
        if len(results) == 0:
            print("No results to report.")
            return

        # Organize results by method
        methods = sorted(set(r.method_name for r in results))
        test_names = sorted(set(r.test_name for r in results))

        # Build summary table
        table_data = []
        for method in methods:
            row = [method]
            n_pass = 0
            total = 0

            for test_name in test_names:
                result = next(
                    (
                        r
                        for r in results
                        if r.method_name == method and r.test_name == test_name
                    ),
                    None,
                )
                if result:
                    status = "PASS" if result.passed else "FAIL"
                    score_str = f"({result.score:.3f})"
                    row.append(f"{status} {score_str}")
                    if result.passed:
                        n_pass += 1
                    total += 1
                else:
                    row.append("N/A")

            row.append(f"{n_pass}/{total}")
            table_data.append(row)

        headers = ["Method"] + test_names + ["TOTAL"]

        print("\n" + "=" * 80)
        print("VALIDATION RESULTS SUMMARY")
        print("=" * 80)
        print(self._format_table(headers, table_data))
        print()

    def print_details(self, results: List[TestResult]) -> None:
        """
        Print detailed information for failed tests.

        Parameters
        ----------
        results : List[TestResult]
            List of test results
        """
        failed_results = [r for r in results if not r.passed]

        if len(failed_results) == 0:
            print("All tests passed! ✓")
            return

        print("\n" + "=" * 80)
        print(f"FAILURE DETAILS ({len(failed_results)} failures)")
        print("=" * 80)

        for result in failed_results:
            print(f"\nMethod: {result.method_name}")
            print(f"Test: {result.test_name}")
            print(f"Score: {result.score:.3f}")
            print(f"Details: {result.details}")

            if result.per_subject_scores:
                print("Per-subject scores:")
                per_subj_array = np.array(result.per_subject_scores)
                print(f"  Mean: {np.mean(per_subj_array):.3f}")
                print(f"  Std: {np.std(per_subj_array):.3f}")
                print(f"  Min: {np.min(per_subj_array):.3f}")
                print(f"  Max: {np.max(per_subj_array):.3f}")
                print(f"  Median: {np.median(per_subj_array):.3f}")

            print("-" * 80)

    def save_json(self, results: List[TestResult]) -> str:
        """
        Save results as JSON.

        Parameters
        ----------
        results : List[TestResult]
            List of test results

        Returns
        -------
        str
            Path to saved JSON file
        """
        # Convert results to dictionaries
        results_dict = {
            "timestamp": self.timestamp,
            "results": [
                {
                    "test_name": r.test_name,
                    "method_name": r.method_name,
                    "passed": r.passed,
                    "score": float(r.score),
                    "per_subject_scores": [float(s) for s in r.per_subject_scores],
                    "details": r.details,
                }
                for r in results
            ],
        }

        output_file = self.output_dir / f"validation_results_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        print(f"Results saved to: {output_file}")
        return str(output_file)

    def generate_report(self, results: List[TestResult], save_json: bool = True) -> None:
        """
        Generate complete report.

        Parameters
        ----------
        results : List[TestResult]
            List of test results
        save_json : bool, default=True
            Whether to save results as JSON
        """
        self.print_summary(results)
        self.print_details(results)

        if save_json:
            self.save_json(results)
