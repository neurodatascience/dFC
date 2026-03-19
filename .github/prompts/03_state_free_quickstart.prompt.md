# State-Free Quickstart (Single Subject)

Guide the user through the fastest way to run PydFC.

## Context Sources

Refer to:
- `docs/DFC_METHODS_CONTEXT.md` for assumptions, interpretation, and comparison principles
- `docs/PAPER_KNOWLEDGE_BASE.md` for paper-grounded implementation details and tradeoffs
- `docs/CHOOSING_A_METHOD.md` for a human-readable decision guide with copy-paste snippets

Always ground method explanations in these documents.

## Deep Mode

When user asks about methods:
- Explain assumptions
- Explain expected behavior
- Avoid oversimplified answers

## Steps

1. Confirm PydFC is installed.
2. Provide demo data download commands.

### Jupyter notebook cells
```python
!curl --create-dirs \
  "https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz?versionId=UfCs4xtwIEPDgmb32qFbtMokl_jxLUKr" \
  -o sample_data/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz

!curl --create-dirs \
  "https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_desc-confounds_regressors.tsv?versionId=biaIJGNQ22P1l1xEsajVzUW6cnu1_8lD" \
  -o sample_data/sub-0001_task-restingstate_acq-mb3_desc-confounds_regressors.tsv
```

*(Remove the leading `!` when running from a terminal.)*

3. Show minimal loading code for `BOLD`:

```python
from pydfc import data_loader
import numpy as np
import warnings

warnings.simplefilter("ignore")

BOLD = data_loader.nifti2timeseries(
    nifti_file=(
        "sample_data/sub-0001_task-restingstate_acq-mb3_"
        "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    ),
    n_rois=100,
    Fs=1 / 0.75,
    subj_id="sub-0001",
    confound_strategy="no_motion",
    standardize=False,
    TS_name=None,
    session=None,
)

BOLD.visualize(start_time=0, end_time=1000, nodes_lst=range(10))
```

4. Ask whether they want SW or TF.
   - If unsure, recommend SW first (simplest, fewest parameters).
   - If they ask SW vs TF, explain assumptions and expected behavior from the context sources.

5. Provide the matching snippet.

### SW (Sliding Window)
```python
from pydfc.dfc_methods import SLIDING_WINDOW

params_methods = {
    "W": 44,
    "n_overlap": 0.5,
    "sw_method": "pear_corr",
    "tapered_window": True,
    "normalization": True,
    "num_select_nodes": None,
}

measure = SLIDING_WINDOW(**params_methods)
dFC = measure.estimate_dFC(time_series=BOLD)
dFC.visualize_dFC(TRs=dFC.TR_array[:], normalize=False, fix_lim=False)
```

### TF (Time-Frequency)
```python
from pydfc.dfc_methods import TIME_FREQ

params_methods = {
    "TF_method": "WTC",
    "n_jobs": 2,
    "verbose": 0,
    "backend": "loky",
    "normalization": True,
    "num_select_nodes": None,
}

measure = TIME_FREQ(**params_methods)
dFC = measure.estimate_dFC(time_series=BOLD)
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)
```

6. After showing results, ask: "Are there any other methods you are curious about?"
7. Cite Torabi et al., 2024 when discussing paper-derived assumptions or tradeoffs.
