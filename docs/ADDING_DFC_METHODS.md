# Adding New dFC Methods to PydFC

This guide summarizes the conventions for adding a dynamic functional
connectivity (dFC) method to PydFC. It is based on the current codebase patterns
and the validation workflow in `tests/test_validation`.

PydFC methods should make their assumptions explicit. Based on the repository
context and Torabi et al., 2024, different dFC methods can produce substantially
different temporal estimates, so new methods should be treated as
assumption-dependent estimators rather than ground truth.

## File Layout

Use one Python file per dFC method:

```text
pydfc/dfc_methods/my_new_method.py
```

Critical rule: method scripts must be self-sufficient.

- Do not make a new method rely on an additional helper script in `pydfc/dfc_methods/` (for example, a shared `*_core.py` that contains required logic).
- Keep the method's full executable logic in its own method file so that each method remains portable and independently readable.
- Shared repository infrastructure imports (for example `BaseDFCMethod`, `DFC`, `TIME_SERIES`) are still expected.

Do not put multiple concrete dFC methods in one module unless they are tightly
coupled variants that must share a public implementation. The established
package style is one method class per file, for example:

```text
pydfc/dfc_methods/sliding_window.py
pydfc/dfc_methods/time_freq.py
pydfc/dfc_methods/cap.py
```

Each concrete method should inherit directly from:

```python
from .base_dfc_method import BaseDFCMethod
```

Do not modify `base_dfc_method.py` just to add a method.

## Required Class Shape

A method class should define:

- `__init__(self, **params)`
- `measure_name` property
- `dFC(...)` or method-specific computation helpers
- `estimate_FCS(...)`
- `estimate_dFC(...)`

State-free methods usually return `self` from `estimate_FCS`, because there are
no group-level functional connectivity states to fit.

State-based methods should implement `estimate_FCS` when they require fitting
states, clusters, dictionaries, or transition models before subject-level dFC
estimation.

## Required Attributes

Initialize these attributes in `__init__`:

```python
self.logs_ = ""
self.TPM = []
self.FCS_ = []
self.FCS_fit_time_ = None
self.dFC_assess_time_ = None
```

Define `params_name_lst` explicitly. Include every parameter used by the method
and every shared preprocessing parameter needed by `BaseDFCMethod`:

```python
self.params_name_lst = [
    "measure_name",
    "is_state_based",
    # method-specific parameters here
    "normalization",
    "num_select_nodes",
    "num_time_point",
    "Fs_ratio",
    "noise_ratio",
    "num_realization",
    "session",
]
```

Then populate `self.params` from `params`:

```python
self.params = {}
for params_name in self.params_name_lst:
    self.params[params_name] = params.get(params_name, None)
```

Always set:

```python
self.params["measure_name"] = "MyNewMethod"
self.params["is_state_based"] = False  # or True for state-based methods
```

Set defaults for method-specific parameters after `self.params` is created:

```python
if self.params["min_periods"] is None:
    self.params["min_periods"] = 10
```

## State-Free Method Template

This is a minimal single-subject state-free method template:

```python
"""
My new dFC method.
"""

import time

import numpy as np

from ..dfc import DFC
from ..time_series import TIME_SERIES
from .base_dfc_method import BaseDFCMethod


class MY_NEW_METHOD(BaseDFCMethod):
    """Short description of the method assumption."""

    def __init__(self, **params):
        self.logs_ = ""
        self.TPM = []
        self.FCS_ = []
        self.FCS_fit_time_ = None
        self.dFC_assess_time_ = None
        self.params_name_lst = [
            "measure_name",
            "is_state_based",
            "min_periods",
            "normalization",
            "num_select_nodes",
            "num_time_point",
            "Fs_ratio",
            "noise_ratio",
            "num_realization",
            "session",
        ]
        self.params = {}
        for params_name in self.params_name_lst:
            self.params[params_name] = params.get(params_name, None)

        self.params["measure_name"] = "MyNewMethod"
        self.params["is_state_based"] = False

        if self.params["min_periods"] is None:
            self.params["min_periods"] = 10

    @property
    def measure_name(self):
        return self.params["measure_name"]

    def dFC(self, time_series, Fs):
        min_periods = int(self.params["min_periods"])
        FCSs = []
        TR_array = []

        for tr in range(min_periods - 1, time_series.shape[1]):
            matrix = np.corrcoef(time_series[:, : tr + 1])
            matrix[np.isnan(matrix)] = 0
            matrix[np.diag_indices_from(matrix)] = 1
            FCSs.append(matrix)
            TR_array.append(tr)

        return np.array(FCSs), np.array(TR_array)

    def estimate_FCS(self, time_series):
        return self

    def estimate_dFC(self, time_series):
        assert (
            len(time_series.subj_id_lst) == 1
        ), "this function takes only one subject as input."
        assert type(time_series) is TIME_SERIES, "time_series must be of TIME_SERIES class."

        time_series = self.manipulate_time_series4dFC(time_series)

        tic = time.time()
        FCSs, TR_array = self.dFC(time_series=time_series.data, Fs=time_series.Fs)
        self.set_dFC_assess_time(time.time() - tic)

        dFC = DFC(measure=self)
        dFC.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.info_dict)
        return dFC
```

## Output Contract

`estimate_dFC` must return a `pydfc.dfc.DFC` object.

For state-free methods, call:

```python
dFC = DFC(measure=self)
dFC.set_dFC(FCSs=FCSs, TR_array=TR_array, TS_info=time_series.info_dict)
```

where:

- `FCSs` has shape `[n_time_samples, n_regions, n_regions]`
- `TR_array` has one integer TR index for each matrix in `FCSs`
- `TR_array` is sorted in ascending order
- every FC matrix is square

Use the same orientation as existing methods:

```text
time_series.data shape = [n_regions, n_timepoints]
```

If the method cannot estimate all TRs, returning a subset is valid as long as
`TR_array` identifies the corresponding time points.

## Preprocessing Contract

Use:

```python
time_series = self.manipulate_time_series4dFC(time_series)
```

inside `estimate_dFC`.

This applies shared options such as:

- `num_select_nodes`
- `Fs_ratio`
- `normalization`
- `noise_ratio`
- `num_time_point`

Include these keys in `params_name_lst`; otherwise `BaseDFCMethod` may fail when
it tries to access them.

For state-based `estimate_FCS`, use:

```python
time_series = self.manipulate_time_series4FCS(time_series)
```

and include any FCS-only parameters such as `num_subj` if the method uses them.

## Package Export

After adding a method file, update:

```text
pydfc/dfc_methods/__init__.py
```

Example:

```python
from .my_new_method import MY_NEW_METHOD

__all__ = [
    ...
    "MY_NEW_METHOD",
]
```

Be aware that package-level imports can fail if a method imports optional
dependencies that are not installed. If a method needs an optional package, keep
the dependency localized and make the failure message clear.

## Validation Wrapper Registration

To use the method in the validation framework, add a wrapper or registry entry
in:

```text
tests/test_validation/dfc_method_wrappers.py
```

For a state-free PydFC method, the generic `PydfcMethodWrapper` pattern is:

```python
class MyNewMethodWrapper(PydfcMethodWrapper):
    def __init__(self, **kwargs):
        MY_NEW_METHOD = _load_pydfc_class(
            "pydfc.dfc_methods.my_new_method", "MY_NEW_METHOD"
        )

        params = {
            "min_periods": kwargs.get("min_periods", 10),
            "normalization": kwargs.get("normalization", True),
            "num_select_nodes": kwargs.get("num_select_nodes", None),
        }
        super().__init__(
            name="MyNewMethod",
            method_factory=MY_NEW_METHOD,
            fit_on_dataset=False,
            **params,
        )
```

Then add it to `_method_registry()`:

```python
"MyNewMethod": {
    "factory": lambda: MyNewMethodWrapper(),
    "aliases": ["mynew", "mnm"],
},
```

Use `fit_on_dataset=True` only for methods that need group-level fitting through
`estimate_FCS`.

## Visualization Registration

To include a method in the visual comparison script, update:

```text
tests/test_validation/visualize_dfc.py
```

Add the method class and parameters to `method_specs()`.

The visualization script checks that every parameter in the config is supported
by the method’s `params_name_lst`. This is intentional: if a parameter is
misspelled or unsupported, visualization should fail early instead of silently
ignoring it.

## Recommended Validation Commands

Run syntax checks:

```bash
python -m py_compile pydfc/dfc_methods/my_new_method.py
python -m py_compile tests/test_validation/dfc_method_wrappers.py
```

List methods and availability:

```bash
python -m tests.test_validation.validate_dfc --list-methods
```

Run a focused validation:

```bash
python -m tests.test_validation.validate_dfc \
  --n-subjects 2 \
  --n-regions 12 \
  --n-timepoints 600 \
  --methods mynew \
  --verbose 0 \
  --pass-threshold 0.5
```

For serious evaluation, use larger synthetic datasets and stricter thresholds.
Passing synthetic tests means the method can recover the validation structure; it
does not prove neurobiological validity.

## Scientific Reporting Checklist

When adding or describing a method, document:

- Whether it is state-free or state-based
- Whether it assumes temporal smoothness, recurrence, event synchrony, phase
  synchrony, latent states, or another dependency model
- What each major hyperparameter controls
- Whether outputs are correlation-like, partial-correlation-like, phase-locking,
  event coactivation, kernel similarity, or another dependency score
- What range of values is expected
- Whether negative values are meaningful
- How the method differs from Sliding Window, CAP, HMM, Windowless, or
  Time-Frequency methods

Based on the repository context and Torabi et al., 2024, method choice can
substantially affect dFC results. New methods should therefore be compared
against established methods rather than interpreted in isolation.

## Common Mistakes

- Putting several unrelated method classes in one file.
- Forgetting to include shared preprocessing keys in `params_name_lst`.
- Returning a raw NumPy array from `estimate_dFC` instead of a `DFC` object.
- Returning matrices without setting `TR_array`.
- Passing unsupported parameters from wrappers or visualization scripts.
- Modifying `base_dfc_method.py` when the method can be implemented as a normal
  subclass.
- Importing optional dependencies at package level in a way that breaks unrelated
  methods.
