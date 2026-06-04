# dFC Validation — API Conformance Checks

Runs a 6-sub-check structural smoke test on any registered dFC method to verify it satisfies the PydFC base-class contract. Each sub-check is isolated: a failure in one does not prevent the rest from running.

## Sub-checks

| # | Name | What it verifies |
|---|------|-----------------|
| 1 | `registry_instantiation` | Class resolves from its `MEASURE_NAME` via `create_measure_obj`; constructor does not raise |
| 2 | `estimate_FCS_returns_self` | `estimate_FCS(group_ts)` returns the method object itself |
| 3 | `estimate_dFC_returns_DFC` | `estimate_dFC(subj_ts)` returns a `pydfc.dfc.DFC` instance |
| 4 | `dfc_mat_shape` | `dfc.get_dFC_mat()` returns a 3-D array of shape `[n_time, R, R]` |
| 5 | `symmetry` | Each FC matrix equals its own transpose (up to 1e-10) |
| 6 | `finite_values` | No NaN or Inf in any element of `get_dFC_mat()` |

**Failure meanings:**

- *Sub-check 1 fails*: method cannot be instantiated at all — check imports, `MEASURE_NAME`, and constructor.
- *Sub-check 2 fails*: `estimate_FCS` does not return `self` — state-based chaining will silently break.
- *Sub-check 3 fails*: no valid `DFC` object returned — sub-checks 4–6 are skipped.
- *Sub-checks 4–6 fail*: output is malformed (wrong shape / asymmetric / non-finite).

## Command-line usage

```bash
# Check all registered methods
python -W ignore -m tests.test_validation.validate_dfc

# Check specific methods (by name or alias)
python -W ignore -m tests.test_validation.validate_dfc --methods SlidingWindow aec led

# List all registered methods and aliases
python -W ignore -m tests.test_validation.validate_dfc --list-methods

# Save output to a file
python -W ignore -m tests.test_validation.validate_dfc > results.txt 2>&1

# Verbose sub-check detail
python -W ignore -m tests.test_validation.validate_dfc --verbose 2
```

## Python API

```python
from tests.test_validation import APIConformanceCheck, Reporter

checker = APIConformanceCheck(n_regions=12, n_timepoints=80, n_subjects=3, Fs=1.0)
sub_results = checker.run("SlidingWindow")
for r in sub_results:
    print(r.name, "PASS" if r.passed else f"FAIL: {r.error}")
```

## JSON output

Results are saved to `./validation_results/validation_results_<timestamp>.json`:

```json
{
  "timestamp": "20260603_120000",
  "api_conformance": {
    "SlidingWindow": [
      {"name": "registry_instantiation",    "passed": true,  "error": ""},
      {"name": "estimate_FCS_returns_self",  "passed": true,  "error": ""},
      {"name": "estimate_dFC_returns_DFC",   "passed": true,  "error": ""},
      {"name": "dfc_mat_shape",              "passed": true,  "error": ""},
      {"name": "symmetry",                   "passed": true,  "error": ""},
      {"name": "finite_values",              "passed": true,  "error": ""}
    ]
  }
}
```

## Directory structure

```
tests/test_validation/
├── __init__.py              # Package exports
├── api_checks.py            # 6-sub-check conformance suite
├── dfc_method_wrappers.py   # CLI alias map; discovery delegates to pydfc auto-scan
├── runner_reporter.py       # Reporter (print + save JSON)
├── validate_dfc.py          # CLI entry point
└── README.md                # This file
```
