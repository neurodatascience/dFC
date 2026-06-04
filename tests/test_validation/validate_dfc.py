"""
dFC Methods — API Conformance Checks

Runs the 6-sub-check API conformance suite on registered dFC methods.
Each check is isolated: a failure in one does not prevent the rest from running.

Usage
-----
python -m tests.test_validation.validate_dfc [--methods METHOD1 METHOD2 ...]
python -m tests.test_validation.validate_dfc --list-methods

Example
-------
python -W ignore -m tests.test_validation.validate_dfc --methods SlidingWindow aec
python -W ignore -m tests.test_validation.validate_dfc > results.txt 2>&1
"""

from __future__ import annotations

import argparse
import sys

from .api_checks import APIConformanceCheck
from .dfc_method_wrappers import (
    get_method_availability,
    get_method_catalog,
    list_registered_methods,
    resolve_method_requests,
)
from .runner_reporter import Reporter


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Run API conformance checks on dFC methods"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to check by name or alias (default: all available)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./validation_results",
        help="Directory for JSON output",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="0=silent, 1=per-method status, 2=per-sub-check detail",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List registered methods with aliases and exit",
    )

    parsed_args = parser.parse_args(args)

    print("=" * 80)
    print("DFC VALIDATION — API CONFORMANCE CHECKS")
    print("=" * 80)

    if parsed_args.list_methods:
        print("\nRegistered methods:")
        for entry in get_method_catalog():
            aliases = ", ".join(entry["aliases"]) if entry["aliases"] else "-"
            print(f"  [{entry['id']}] {entry['key']}")
            print(f"      aliases: {aliases}")
        return 0

    all_methods, _ = get_method_availability()

    if parsed_args.methods:
        methods_to_test, missing, _ = resolve_method_requests(parsed_args.methods)
        if missing:
            print(f"\nWARNING: Methods not found: {missing}")
            print(f"  Registered methods: {list_registered_methods()}")
    else:
        methods_to_test = all_methods

    print(f"\nMethods to test: {methods_to_test}")

    if not methods_to_test:
        print("ERROR: No runnable methods selected.")
        return 2

    # Run API conformance checks
    print("\nRunning API conformance checks...")
    api_checker = APIConformanceCheck()
    api_conformance_results = {}

    for measure_name in methods_to_test:
        if parsed_args.verbose >= 2:
            print(f"  checking {measure_name}...")
        api_conformance_results[measure_name] = api_checker.run(measure_name)

    # Report
    reporter = Reporter(output_dir=parsed_args.output_dir)
    reporter.generate_report(api_conformance_results, save_json=True)

    all_passed = all(
        all(r.passed for r in sub_results)
        for sub_results in api_conformance_results.values()
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
