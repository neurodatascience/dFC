"""
Example: Using the dFC Validation Framework

This script demonstrates how to use the dFC validation framework with a simple
example using the DummyMethod (for testing the framework itself).

To run:
    python example_validation.py

To use with a real dFC method, modify the methods dict and ensure the wrapper
is correctly implemented.
"""

from pathlib import Path

import numpy as np

# Import validation framework components
from test_validation import (
    DummyMethod,
    Reporter,
    StepChangeTest,
    SyntheticDataGenerator,
    ValidationRunner,
    ZeroAndPerfectCorrTest,
)


def example_basic_usage():
    """Basic example: Generate data, run dummy method, validate."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 80)

    # Step 1: Generate synthetic dataset
    print("\n1. Generating synthetic dataset...")
    generator = SyntheticDataGenerator(
        n_subjects=10,  # Small dataset for quick demo
        n_regions=50,
        n_timepoints=300,
        segment_n_blocks=[3, 4, 5],
        noise_floor=0.01,
        random_seed=42,
    )

    timeseries, ground_truth = generator.generate()
    print(f"   Timeseries shape: {timeseries.shape}")
    print(f"   Segments: {len(ground_truth.segments)}")

    # Step 2: Create tests
    print("\n2. Creating test cases...")
    tests = [
        ZeroAndPerfectCorrTest(pass_threshold=0.85),
        StepChangeTest(pass_threshold=0.85),
    ]
    print(f"   Tests: {[t.name for t in tests]}")

    # Step 3: Create method (using DummyMethod for demonstration)
    print("\n3. Setting up dFC method...")
    methods = {
        "DummyMethod": DummyMethod(),
    }
    print(f"   Methods: {list(methods.keys())}")

    # Step 4: Run validation
    print("\n4. Running validation...")
    runner = ValidationRunner(verbose=1)
    results = runner.run(
        methods=methods,
        test_cases=tests,
        timeseries=timeseries,
        ground_truth=ground_truth,
    )

    # Step 5: Report results
    print("\n5. Generating report...")
    reporter = Reporter(output_dir="./example_results")
    reporter.print_summary(results)
    reporter.print_details(results)

    return results


def example_custom_configuration():
    """Example: Using custom parameters and configurations."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Configuration")
    print("=" * 80)

    # Generate larger dataset with custom settings
    print("\n1. Generating custom dataset...")
    generator = SyntheticDataGenerator(
        n_subjects=20,
        n_regions=100,
        n_timepoints=600,
        segment_n_blocks=[6, 8, 10],  # More blocks
        noise_floor=0.02,  # Higher noise
        random_seed=123,
        signal_type="gaussian_smooth",
    )

    timeseries, ground_truth = generator.generate()

    # Inspect ground truth
    print("\n   Dataset details:")
    print(f"   - Shape: {timeseries.shape}")
    print(f"   - Noise floor: {ground_truth.noise_floor}")
    print("   - Segments:")
    for i, seg in enumerate(ground_truth.segments):
        print(
            f"       Segment {i}: blocks={seg.n_blocks}, "
            f"eval_window=[{seg.eval_start}:{seg.eval_end}]"
        )

    # Create tests with custom thresholds
    print("\n2. Creating tests with custom thresholds...")
    tests = [
        ZeroAndPerfectCorrTest(pass_threshold=0.80),  # Relaxed
        StepChangeTest(pass_threshold=0.75),  # More relaxed
    ]

    # Run with dummy method
    methods = {"DummyMethod": DummyMethod()}

    print("\n3. Running validation...")
    runner = ValidationRunner(verbose=1)
    results = runner.run(
        methods=methods,
        test_cases=tests,
        timeseries=timeseries,
        ground_truth=ground_truth,
    )

    # Report with JSON output
    print("\n4. Generating report with JSON...")
    reporter = Reporter(output_dir="./example_results_custom")
    reporter.print_summary(results)
    json_file = reporter.save_json(results)
    print(f"   Results saved: {json_file}")

    return results


def example_inspect_synthetic_data():
    """Example: Inspect properties of synthetic data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Inspecting Synthetic Data")
    print("=" * 80)

    # Generate small dataset for inspection
    generator = SyntheticDataGenerator(
        n_subjects=5,
        n_regions=20,
        n_timepoints=200,
        segment_n_blocks=[2, 3],
        random_seed=42,
    )

    timeseries, ground_truth = generator.generate()

    print(
        f"\nDataset: {timeseries.shape[0]} subjects, "
        f"{timeseries.shape[1]} timepoints, {timeseries.shape[2]} regions"
    )

    # Analyze first segment
    seg0 = ground_truth.segments[0]
    print("\nSegment 0 Analysis:")
    print(f"  Timepoint range: [{seg0.start}:{seg0.end}]")
    print(f"  Evaluation window: [{seg0.eval_start}:{seg0.eval_end}]")
    print(f"  Number of blocks: {seg0.n_blocks}")
    print(f"  Block assignments: {seg0.region_block_ids}")

    # Check block sizes
    unique_blocks, counts = np.unique(seg0.region_block_ids, return_counts=True)
    print(f"  Block sizes: {dict(zip(unique_blocks, counts))}")

    # Verify correlation structure
    n_perfect = np.sum(seg0.perfect_corr_mask) // 2  # Divide by 2 (symmetric)
    n_zero = np.sum(seg0.zero_corr_mask) // 2
    print(f"  Perfect-corr region pairs: {n_perfect}")
    print(f"  Zero-corr region pairs: {n_zero}")

    # Check actual correlations in synthetic data
    print("\nActual correlations in first subject, first segment:")
    subject_segment_ts = timeseries[0, seg0.start : seg0.eval_end, :]
    actual_corr = np.corrcoef(subject_segment_ts.T)

    # Within-block correlation
    within_block_indices = np.where(
        seg0.perfect_corr_mask
        & (
            np.arange(seg0.perfect_corr_mask.shape[0])[:, None]
            != np.arange(seg0.perfect_corr_mask.shape[0])[None, :]
        )
    )
    within_block_corrs = actual_corr[within_block_indices]
    print(
        f"  Within-block corr: mean={np.mean(within_block_corrs):.3f}, "
        f"std={np.std(within_block_corrs):.3f}"
    )

    # Between-block correlation
    between_block_indices = np.where(seg0.zero_corr_mask)
    between_block_corrs = actual_corr[between_block_indices]
    print(
        f"  Between-block corr: mean={np.mean(between_block_corrs):.3f}, "
        f"std={np.std(between_block_corrs):.3f}"
    )


def example_comparing_methods():
    """Example: Compare multiple methods (extended)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Comparing Multiple Methods")
    print("=" * 80)

    # Generate dataset
    print("\n1. Generating dataset...")
    generator = SyntheticDataGenerator(
        n_subjects=15,
        n_regions=80,
        n_timepoints=400,
        segment_n_blocks=[4, 5, 6],
    )
    timeseries, ground_truth = generator.generate()

    # Create tests
    tests = [
        ZeroAndPerfectCorrTest(),
        StepChangeTest(),
    ]

    # Compare multiple method configurations
    methods = {
        "DummyMethod_v1": DummyMethod(),
        "DummyMethod_v2": DummyMethod(),  # Could be different variant
    }

    print(f"\n2. Testing {len(methods)} methods...")
    runner = ValidationRunner(verbose=1)
    results = runner.run(
        methods=methods,
        test_cases=tests,
        timeseries=timeseries,
        ground_truth=ground_truth,
    )

    # Generate comparison report
    print("\n3. Comparison Results:")
    reporter = Reporter()
    reporter.print_summary(results)

    # Analyze which method performed better
    print("\nDetailed Comparison:")
    for method_name in methods.keys():
        method_results = [r for r in results if r.method_name == method_name]
        avg_score = np.mean([r.score for r in method_results])
        n_pass = sum(1 for r in method_results if r.passed)
        print(f"  {method_name}:")
        print(f"    - Average score: {avg_score:.3f}")
        print(f"    - Tests passed: {n_pass}/{len(tests)}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DFC VALIDATION FRAMEWORK - EXAMPLES")
    print("=" * 80)

    # Run all examples
    print("\nRunning examples...\n")

    # Example 1
    try:
        example_basic_usage()
    except Exception as e:
        print(f"ERROR in example 1: {e}")

    # Example 2
    try:
        example_custom_configuration()
    except Exception as e:
        print(f"ERROR in example 2: {e}")

    # Example 3
    try:
        example_inspect_synthetic_data()
    except Exception as e:
        print(f"ERROR in example 3: {e}")

    # Example 4
    try:
        example_comparing_methods()
    except Exception as e:
        print(f"ERROR in example 4: {e}")

    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
