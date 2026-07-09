# DFC Validation Framework - Quick Start Guide

## 5-Minute Setup

### Installation

The validation framework requires only standard dependencies already in the pydfc environment:

```bash
# Required (usually already installed):
pip install numpy scipy scikit-learn joblib
```

No additional installation needed! The framework is self-contained in `/tests/test_validation/`.

## Quick Examples

### 1. Run Default Validation (30 seconds)

```bash
cd /path/to/dfc/repo/tests/test_validation
python validate_dfc.py
```

This generates synthetic data and tests all available methods with default settings.

### 1b. List Methods Without Running

```bash
python -m tests.test_validation.validate_dfc --list-methods
```

This prints registered method IDs, aliases, and availability reasons.

### 2. Run with Custom Settings (1 minute)

```bash
python validate_dfc.py \
    --n-subjects 50 \
    --n-regions 100 \
    --n-timepoints 600 \
    --pass-threshold 0.90 \
    --output-dir ./my_results
```

### 3. Test Specific Methods Only

```bash
python validate_dfc.py --methods SlidingWindow_W30 DummyMethod
```

Default selection note:
- If `--methods` is omitted, only runnable methods in the current environment are selected.

### 4. Run from Python

```python
from test_validation import (
    SyntheticDataGenerator,
    ZeroAndPerfectCorrTest,
    StepChangeTest,
    ValidationRunner,
    Reporter,
    DummyMethod,
)

# Generate synthetic data
gen = SyntheticDataGenerator(n_subjects=50, n_regions=100, n_timepoints=600)
timeseries, ground_truth = gen.generate()

# Create tests
tests = [ZeroAndPerfectCorrTest(), StepChangeTest()]

# Run validation
runner = ValidationRunner()
results = runner.run(
    methods={"DummyMethod": DummyMethod()},
    test_cases=tests,
    timeseries=timeseries,
    ground_truth=ground_truth,
)

# Report results
reporter = Reporter()
reporter.generate_report(results)
```

## Adding Your Method

### Step 1: Create a Wrapper

```python
from test_validation import DFCMethodWrapper

class MyMethodWrapper(DFCMethodWrapper):
    def __init__(self, param1=value1):
        super().__init__(name="MyMethod_param1")
        self.method = MyDFCClass(param1=param1)

    def run(self, timeseries):
        """
        Input: timeseries [n_subjects, n_timepoints, n_regions]
        Output: dfc_output [n_subjects, n_timepoints, n_regions, n_regions]
        """
        n_subjects, n_timepoints, n_regions = timeseries.shape
        dfc_output = np.zeros((n_subjects, n_timepoints, n_regions, n_regions))

        for subj in range(n_subjects):
            # Run your method on subject data
            result = self.method.run(timeseries[subj])
            # Store result (ensure shape is [n_timepoints, n_regions, n_regions])
            dfc_output[subj] = result

        return dfc_output
```

### Step 2: Register Your Method

Add to `dfc_method_wrappers.py` in the `get_available_methods()` function:

```python
def get_available_methods():
    methods = {
        "DummyMethod": DummyMethod(),
        "SlidingWindow_W30": SlidingWindowWrapper(W=30),
        "MyMethod_param1": MyMethodWrapper(param1=value1),  # Add this
    }
    return methods
```

### Step 3: Test Your Method

```bash
python validate_dfc.py --methods MyMethod_param1
```

## Understanding Results

### Summary Table

```
Method          | ZeroAndPerfectCorr | StepChange | TOTAL
────────────────┼────────────────────┼────────────┼──────
MyMethod_param1 | PASS (0.92)        | PASS (0.89)| 2/2
```

- **PASS**: Score > pass_threshold (default 0.90)
- **FAIL**: Score ≤ pass_threshold
- **Score**: Rank-biserial correlation [-1, 1], higher is better

### Failure Analysis

If your method fails:

1. **Check per-subject scores:** High variance suggests numerical instability
2. **Verify output shape:** Must be `[n_subjects, n_timepoints, n_regions, n_regions]`
3. **Use absolute values:** Tests compare absolute connectivity (take `np.abs()`)
4. **Check temporal resolution:** Method must have sufficient timepoint resolution

## File Structure

```
test_validation/
├── README.md                    # Full documentation
├── QUICKSTART.md               # This file
├── synthetic_data.py           # Generate synthetic fMRI
├── test_cases.py               # Test definitions (ZeroAndPerfectCorr, StepChange)
├── dfc_method_wrappers.py      # Method wrappers
├── runner_reporter.py          # Test execution
├── validate_dfc.py             # Entry point (CLI)
└── example_validation.py       # Examples
```

## Troubleshooting

The validation reporter now uses only the Python standard library. If you see a
formatting issue in the summary table, check the local validation code rather
than an external package install.

### Method Run Fails

1. Check that `run()` returns shape `[n_subjects, n_timepoints, n_regions, n_regions]`
2. Ensure time axis is index 1 (not 0 or 2)
3. Verify connectivity values are numeric (not NaN or inf)

### All Tests Fail

- Your method might genuinely not match the synthetic structure
- Try with a relaxed `--pass-threshold 0.70`
- Check if method needs different parameter tuning

### Results Saved Where?

Default: `./validation_results/validation_results_YYYYMMDD_HHMMSS.json`

Specify with: `--output-dir /path/to/results`

## Next Steps

1. **Read full docs:** See `README.md` for detailed information
2. **Run examples:** `python example_validation.py`
3. **Add your method:** Follow "Adding Your Method" above
4. **Explore outputs:** Check saved JSON for detailed per-subject scores
5. **Customize tests:** Extend `test_cases.py` for domain-specific validation

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review examples in `example_validation.py`
3. Inspect validation output and per-subject scores
4. Check method wrapper implementation
