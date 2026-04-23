# DFC Validation Framework

A comprehensive testing framework for dynamic functional connectivity (dFC) methods using synthetic data with known ground truth structure.

## Overview

Since there is no ground truth for real dFC data, this framework generates synthetic time series with **known and controllable connectivity structure** to enable automatic validation of dFC methods. The synthetic data is designed with multiple segments, each containing a different block structure where:

- **Within-block regions**: perfectly correlated (correlation = 1.0)
- **Between-block regions**: uncorrelated (correlation = 0.0)
- **Block structure varies across segments**: allows testing temporal sensitivity

## Architecture

### 1. Synthetic Data Generator (`synthetic_data.py`)

Generates synthetic fMRI time series with known block connectivity structure.

**Key Features:**
- Configurable number of subjects, regions, and timepoints
- Three segments with different block structures (default: 5, 7, 10 blocks)
- Ground truth metadata including correlation masks and block assignments
- Deterministic generation (configurable random seed)
- Numerical realism with optional noise floor

**Example:**
```python
from test_validation import SyntheticDataGenerator

generator = SyntheticDataGenerator(
    n_subjects=50,
    n_regions=100,
    n_timepoints=600,
    segment_n_blocks=[5, 7, 10],
    noise_floor=0.01,
    random_seed=42
)

timeseries, ground_truth = generator.generate()
# timeseries shape: [50, 600, 100]
# ground_truth contains block assignments and correlation masks
```

### 2. Test Cases (`test_cases.py`)

Define what constitutes a "pass" for a dFC method.

#### **ZeroAndPerfectCorrTest**

Tests whether a dFC method correctly **separates zero-correlation pairs from perfect-correlation pairs** using rank-based statistics.

- **Method:** Rank-biserial correlation between binary labels (zero-corr vs perfect-corr) and ranked absolute connectivity values
- **Agnostic to:** Absolute connectivity scale (works with correlation, Fisher-z, coherence, etc.)
- **Evaluation window:** Middle 100 timepoints of each segment
- **Score range:** [-1, 1] (1 = perfect separation, -1 = reversed)
- **Pass threshold:** 0.9 (configurable)

#### **StepChangeTest**

Tests whether connectivity **changes appropriately between segments** when the same pair switches connectivity class.

- **Principle:** If a pair is perfect-corr in segment A and zero-corr in segment B, its connectivity should rank higher in A than B (and vice versa)
- **Method:** Rank-biserial correlation of rank changes across segment transitions
- **Evaluation:** Compares adjacent segment pairs
- **Score range:** [-1, 1]
- **Pass threshold:** 0.9 (configurable)

**Adding New Tests:**

```python
from test_validation import TestCase, TestResult

class MyCustomTest(TestCase):
    def __init__(self, pass_threshold=0.9):
        super().__init__(
            name="MyTest",
            description="Test description",
            pass_threshold=pass_threshold
        )

    def evaluate(self, dfc_output, ground_truth) -> TestResult:
        # Your test logic here
        score = compute_score(dfc_output, ground_truth)
        passed = score > self.pass_threshold
        return TestResult(
            test_name=self.name,
            method_name="",  # filled by runner
            passed=passed,
            score=score,
            per_subject_scores=[...],
            details=f"Description of results"
        )
```

### 3. Method Wrappers (`dfc_method_wrappers.py`)

Standardize the interface for dFC methods to work with the validation framework.

**DFCMethodWrapper Base Class:**
- Inherits to wrap existing dFC methods
- Standardizes input: `[n_subjects, n_timepoints, n_regions]`
- Standardizes output: `[n_subjects, n_timepoints, n_regions, n_regions]` or dict

**Existing Wrappers:**
- `SlidingWindowWrapper`: Wraps pydfc.dfc_methods.SLIDING_WINDOW
- `DummyMethod`: Test wrapper that produces synthetic dFC output

**Creating a New Wrapper:**

```python
from test_validation import DFCMethodWrapper

class MyMethodWrapper(DFCMethodWrapper):
    def __init__(self, **params):
        super().__init__(name="MyMethod", **params)
        # Initialize your method here
        self.method = MyDFCImplementation(**params)

    def run(self, timeseries):
        """
        Parameters
        ----------
        timeseries : np.ndarray
            Shape [n_subjects, n_timepoints, n_regions]

        Returns
        -------
        dfc_output : np.ndarray
            Shape [n_subjects, n_timepoints, n_regions, n_regions]
        """
        # Implementation
        return dfc_output
```

### 4. Runner and Reporter (`runner_reporter.py`)

Execute tests and generate results reports.

**ValidationRunner:**
- Runs all test cases on all methods
- Handles errors gracefully
- Supports verbose output

**Reporter:**
- Prints summary table
- Shows detailed failure information
- Saves results to JSON

## Usage

### Command Line

```bash
# List registered methods (with availability and reasons) without running tests
python -m test_validation.validate_dfc --list-methods

# Run with default settings
python -m test_validation.validate_dfc

# Customize dataset and methods
python -m test_validation.validate_dfc \
    --n-subjects 100 \
    --n-regions 200 \
    --n-timepoints 1200 \
    --methods SlidingWindow_W30 MyCustomMethod \
    --pass-threshold 0.85 \
    --output-dir ./my_results \
    --seed 123 \
    --verbose 2
```

Default behavior note:
- When `--methods` is not provided, the validator runs only methods that are currently runnable in the active environment.
- Methods missing optional dependencies are listed as unavailable and skipped.

### Python Script

```python
from test_validation import (
    SyntheticDataGenerator,
    ZeroAndPerfectCorrTest,
    StepChangeTest,
    ValidationRunner,
    Reporter,
    get_available_methods,
)

# 1. Generate synthetic data
generator = SyntheticDataGenerator(
    n_subjects=50,
    n_regions=100,
    n_timepoints=600,
    noise_floor=0.01
)
timeseries, ground_truth = generator.generate()

# 2. Create test suite
tests = [
    ZeroAndPerfectCorrTest(pass_threshold=0.9),
    StepChangeTest(pass_threshold=0.9),
]

# 3. Get methods to test
methods = get_available_methods()

# 4. Run validation
runner = ValidationRunner(verbose=2)
results = runner.run(
    methods=methods,
    test_cases=tests,
    timeseries=timeseries,
    ground_truth=ground_truth,
)

# 5. Generate report
reporter = Reporter(output_dir="./results")
reporter.generate_report(results)
```

## Output Format

### Summary Table

```
Method                  │ ZeroAndPerfectCorr    │ StepChange         │ TOTAL
────────────────────────┼───────────────────────┼────────────────────┼──────
SlidingWindow_W30       │ PASS (0.97)           │ PASS (0.94)        │ 2/2
DummyMethod             │ PASS (0.92)           │ PASS (0.91)        │ 2/2
```

### Detailed Results (JSON)

```json
{
  "timestamp": "20240422_143022",
  "results": [
    {
      "test_name": "ZeroAndPerfectCorr",
      "method_name": "SlidingWindow_W30",
      "passed": true,
      "score": 0.972,
      "per_subject_scores": [0.95, 0.98, ...],
      "details": "Rank-biserial correlation scores across 50 subjects..."
    }
  ]
}
```

## Data Format Details

### Input Timeseries

**Shape:** `[n_subjects, n_timepoints, n_regions]`

```python
# Example with synthetic data
timeseries.shape  # (50, 600, 100)
timeseries[0, :, :]  # First subject: [600 timepoints, 100 regions]
```

### Output dFC (from methods)

**Expected Shape:** `[n_subjects, n_timepoints, n_regions, n_regions]`

Where `[i, t, r1, r2]` is the connectivity between regions r1 and r2 at timepoint t for subject i.

### Ground Truth Structure

```python
ground_truth.n_subjects      # 50
ground_truth.n_regions       # 100
ground_truth.n_timepoints    # 600
ground_truth.TR              # 1.0 (seconds)
ground_truth.noise_floor     # 0.01

# Access segment information
for segment in ground_truth.segments:
    segment.start                # segment start timepoint
    segment.end                  # segment end timepoint
    segment.eval_start           # evaluation window start
    segment.eval_end             # evaluation window end
    segment.n_blocks             # number of blocks in this segment
    segment.region_block_ids     # [n_regions] array of block assignments
    segment.perfect_corr_mask    # [n_regions, n_regions] boolean mask
    segment.zero_corr_mask       # [n_regions, n_regions] boolean mask
```

## Advanced Features

### Custom Noise Characteristics

```python
generator = SyntheticDataGenerator(
    noise_floor=0.05,  # Higher noise for challenging validation
    signal_type="white_noise",  # or "gaussian_smooth"
)
```

### Custom Pass Thresholds

```python
test = ZeroAndPerfectCorrTest(pass_threshold=0.85)  # Relaxed threshold
```

### HPC Integration

Each method can be run independently, facilitating parallelization:

```python
# Could be submitted as separate SLURM jobs
for method_name in method_names:
    method = methods[method_name]
    dfc_output = method.run(timeseries)
    results = [tc.evaluate(dfc_output, gt) for tc in test_cases]
    reporter.save_json(results)
```

## Interpreting Results

### High Scores (> 0.90)

The method correctly captures the synthetic connectivity structure:
- Separates zero-correlation from perfect-correlation pairs
- Shows appropriate temporal dynamics as block structure changes

### Low Scores (< 0.70)

Possible issues:
- Method produces spurious correlations or false zeros
- Temporal resolution too coarse (missing transitions)
- Sensitivity to noise settings
- Implementation bug in wrapper

### Troubleshooting

1. **Check per-subject scores:** High variance suggests subject-specific issues
2. **Visualize segment-level results:** Determine which segments fail
3. **Verify data flow:** Ensure dFC output shape matches expected format
4. **Test with relaxed threshold:** Confirm test infrastructure is working

## References

This framework is designed for validating dFC methods as described in:

- Torabi et al., 2024. "On the variability of dynamic functional connectivity assessment methods." *GigaScience*.

## Directory Structure

```
test_validation/
├── __init__.py                  # Package initialization
├── synthetic_data.py            # Synthetic data generator
├── test_cases.py                # Test case implementations
├── dfc_method_wrappers.py       # Method wrapper classes
├── runner_reporter.py           # Test execution and reporting
├── validate_dfc.py              # Main entry point
└── README.md                    # This file
```

## Contributing

To add a new test case:

1. Create a subclass of `TestCase` in `test_cases.py`
2. Implement the `evaluate()` method
3. Add to the test suite in `validate_dfc.py`

To add a new dFC method wrapper:

1. Create a subclass of `DFCMethodWrapper` in `dfc_method_wrappers.py`
2. Implement the `run()` method
3. Register in `get_available_methods()`
