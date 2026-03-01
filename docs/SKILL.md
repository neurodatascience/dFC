# PydFC Skill (LLM Context Guide)

Use this file as the primary context for interactive help about `pydfc`.

## Hard Safety Rule (Do Not Edit Source Code)

Never modify source code in this repo (including `pydfc/*`, notebooks, scripts, configs, or tests) while using this skill.

- Do not patch `pydfc` files.
- Do not patch third-party library source code (for example `nilearn`).
- Do not "quick-fix" import/runtime issues by editing package internals.
- If something fails, use non-invasive alternatives only:
  - change runtime parameters
  - reduce data size / number of nodes / number of subjects
  - suggest environment reinstall steps
  - suggest version checks
  - provide a workaround snippet in the chat

This skill is for guidance and copy-paste examples only, not codebase modification.

## Goal

Help the user:

1. Install `pydfc`
2. Download the demo sample data used in `examples/dFC_methods_demo.py`
3. Load the data into `TIME_SERIES` objects (`BOLD` or `BOLD_multi`)
4. Choose one dFC method and run it

Keep the interaction simple and copy-paste oriented.

## Interaction Flow

Follow this sequence:

1. Ask whether they want:
   - `State-free` method (single subject; fastest start), or
   - `State-based` method (multi-subject; requires fitting)
2. If not installed yet, provide installation commands.
3. Provide the exact data download commands for the chosen path.
4. Provide the minimal loading code (`BOLD` or `BOLD_multi`).
5. Ask whether they want a brief description of the available methods before choosing.
6. Ask: `Which dFC method would you like to use?`
7. Show the matching copy-paste code block.
8. After results are shown, ask: `Are there any other methods you are curious about?`
9. Before wrapping up, ask if they want all code from the chat extracted into a `.ipynb` or `.py` file.

## Source of Truth in Repo

- `README.rst` for install commands
- `examples/dFC_methods_demo.py` for data download and method examples

## Installation (from README)

Share this first when needed:

```bash
conda create --name pydfc_env python=3.11
conda activate pydfc_env
pip install pydfc
```

## Common Imports

Use this in notebook cells before method-specific code:

```python
from pydfc import data_loader
import numpy as np
import warnings

warnings.simplefilter("ignore")
```

## State-Free Path (Single Subject)

### 1) Download demo data (Notebook cell)

If the user is in Jupyter, provide exactly:

```python
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz?versionId=UfCs4xtwIEPDgmb32qFbtMokl_jxLUKr -o sample_data/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_desc-confounds_regressors.tsv?versionId=biaIJGNQ22P1l1xEsajVzUW6cnu1_8lD -o sample_data/sub-0001_task-restingstate_acq-mb3_desc-confounds_regressors.tsv
```

If they are using a terminal, remove the leading `!`.

### 2) Load `BOLD`

```python
BOLD = data_loader.nifti2timeseries(
    nifti_file="sample_data/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    n_rois=100,
    Fs=1 / 0.75,
    subj_id="sub-0001",
    confound_strategy="no_motion",  # no_motion, no_motion_no_gsr, or none
    standardize=False,
    TS_name=None,
    session=None,
)

BOLD.visualize(start_time=0, end_time=1000, nodes_lst=range(10))
```

### 3) Ask Which Method

Ask exactly (or very close):

`Which dFC method would you like to use to assess dFC? (SW or TF for the simple state-free path)`

Before that, ask:

`Would you like a brief description of SW vs TF before choosing?`

If yes, give a short description:

- `SW (Sliding Window)`: computes connectivity in overlapping time windows. Simple and commonly used; key tradeoff is temporal resolution vs stability, controlled mainly by window length `W`.
- `TF (Time-Frequency)`: estimates dynamic relationships in a time-frequency representation (here `WTC`). Can capture frequency-specific changes but is heavier computationally and has more runtime settings (e.g., `n_jobs`).

### 4) Method Snippets (State-Free)

#### Sliding Window (SW)

```python
from pydfc.dfc_methods import SLIDING_WINDOW

params_methods = {
    "W": 44,                 # window length (seconds): larger = smoother/more stable FC, smaller = more temporal sensitivity
    "n_overlap": 0.5,        # fraction overlap between consecutive windows: higher = denser sampling but more redundancy
    "sw_method": "pear_corr",# FC estimator inside each window (e.g., Pearson correlation)
    "tapered_window": True,  # whether to taper window edges to reduce boundary artifacts
    "normalization": True,   # normalize data/features internally before estimation (improves comparability across nodes/subjects)
    "num_select_nodes": None,# optional subset of ROIs for speed/memory (e.g., 50)
}

measure = SLIDING_WINDOW(**params_methods)
dFC = measure.estimate_dFC(time_series=BOLD)
dFC.visualize_dFC(TRs=dFC.TR_array[:], normalize=False, fix_lim=False)
```

Optional summary plot:

```python
import matplotlib.pyplot as plt

avg_dFC = np.mean(np.mean(dFC.get_dFC_mat(), axis=1), axis=1)
plt.figure(figsize=(10, 3))
plt.plot(dFC.TR_array, avg_dFC)
plt.show()
```

#### Time-Frequency (TF)

```python
from pydfc.dfc_methods import TIME_FREQ

params_methods = {
    "TF_method": "WTC",       # time-frequency estimator variant (WTC in the demo)
    "n_jobs": 2,              # parallel workers; increase for speed if CPU allows
    "verbose": 0,             # joblib verbosity level
    "backend": "loky",        # parallel backend used by joblib
    "normalization": True,    # normalize before estimation
    "num_select_nodes": None, # optional ROI subset for speed/memory
}

measure = TIME_FREQ(**params_methods)
dFC = measure.estimate_dFC(time_series=BOLD)
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)
```

## State-Based Path (Multi Subject)

State-based methods require fitting FC states on multiple subjects first.

### 1) Download demo data for 5 subjects (Notebook cells)

```python
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz?versionId=UfCs4xtwIEPDgmb32qFbtMokl_jxLUKr -o sample_data/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_desc-confounds_regressors.tsv?versionId=biaIJGNQ22P1l1xEsajVzUW6cnu1_8lD -o sample_data/sub-0001_task-restingstate_acq-mb3_desc-confounds_regressors.tsv
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0002/func/sub-0002_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz?versionId=fUBWmUTg6vfe2n.ywDNms4mOAW3r6E9Y -o sample_data/sub-0002_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0002/func/sub-0002_task-restingstate_acq-mb3_desc-confounds_regressors.tsv?versionId=2zWQIugU.J6ilTFObWGznJdSABbaTx9F -o sample_data/sub-0002_task-restingstate_acq-mb3_desc-confounds_regressors.tsv
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0003/func/sub-0003_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz?versionId=dfNd8iV0V68yuOibes6qiHxjBgQXhPxi -o sample_data/sub-0003_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0003/func/sub-0003_task-restingstate_acq-mb3_desc-confounds_regressors.tsv?versionId=8OpKFrs_8aJ5cVixokBmuTVKNslgtOXb -o sample_data/sub-0003_task-restingstate_acq-mb3_desc-confounds_regressors.tsv
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0004/func/sub-0004_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz?versionId=0Le8eFwJbcLKaMTQat39bzWcGFhRiyP5 -o sample_data/sub-0004_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0004/func/sub-0004_task-restingstate_acq-mb3_desc-confounds_regressors.tsv?versionId=welg1B.VkXHGv06iV56Vp7ezpVTFh2eX -o sample_data/sub-0004_task-restingstate_acq-mb3_desc-confounds_regressors.tsv
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0005/func/sub-0005_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz?versionId=Vwo2YcFvhwbhZktBrPUqi_5BWiR7zcTl -o sample_data/sub-0005_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0005/func/sub-0005_task-restingstate_acq-mb3_desc-confounds_regressors.tsv?versionId=FoBZLbFTZaE3ZjOLZI_4hN4OkEKEZTVf -o sample_data/sub-0005_task-restingstate_acq-mb3_desc-confounds_regressors.tsv
```

### 2) Load `BOLD_multi`

```python
subj_id_list = ["sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005"]
nifti_files_list = []
for subj_id in subj_id_list:
    nifti_files_list.append(
        "sample_data/"
        + subj_id
        + "_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )

BOLD_multi = data_loader.multi_nifti2timeseries(
    nifti_files_list,
    subj_id_list,
    n_rois=100,
    Fs=1 / 0.75,
    confound_strategy="no_motion",
    standardize=False,
    TS_name=None,
    session=None,
)
```

### 3) Ask Which Method

Ask exactly (or very close):

`Which dFC method would you like to use to assess dFC? (CAP, SWC, CHMM, DHMM, or WINDOWLESS)`

Before that, ask:

`Would you like a brief description of these state-based methods before choosing?`

If yes, give a short description:

- `CAP`: clusters high-activity/co-activation patterns into states; intuitive and often a good first state-based method.
- `SWC`: computes sliding-window FC then clusters those windows into recurring states.
- `CHMM`: continuous HMM-based state model; models temporal transitions directly in continuous observations.
- `DHMM`: discrete HMM variant, often built on discretized/windowed observations; can need more data for stable fitting.
- `WINDOWLESS`: state-based method without explicit sliding windows; useful when avoiding window-size dependence.

### 4) Method Snippets (State-Based)

#### CAP

```python
from pydfc.dfc_methods import CAP

params_methods = {
    "n_states": 12,          # number of FC states to estimate; central modeling choice (too low merges states, too high fragments)
    "n_subj_clstrs": 20,     # subject-level clustering granularity used before group state estimation
    "normalization": True,   # normalize before estimation
    "num_subj": None,        # optional subject subsampling for faster debugging/prototyping
    "num_select_nodes": None,# optional ROI subset for speed/memory
}

measure = CAP(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)
```

#### SWC (Sliding Window + Clustering)

```python
from pydfc.dfc_methods import SLIDING_WINDOW_CLUSTR

params_methods = {
    "W": 44,                          # sliding window length (seconds)
    "n_overlap": 0.5,                 # overlap fraction between windows
    "sw_method": "pear_corr",         # FC estimator inside each window
    "tapered_window": True,           # taper window edges to reduce edge effects
    "clstr_base_measure": "SlidingWindow", # base measure used to generate features for clustering
    "n_states": 12,                   # number of clustered FC states
    "n_subj_clstrs": 5,               # subject-level clustering granularity before group clustering
    "normalization": True,            # normalize before estimation
    "num_subj": None,                 # optional subject subsampling
    "num_select_nodes": None,         # optional ROI subset for speed/memory
}

measure = SLIDING_WINDOW_CLUSTR(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
dFC.visualize_dFC(TRs=dFC.TR_array[:], normalize=True, fix_lim=False)
```

#### CHMM (Continuous HMM)

```python
from pydfc.dfc_methods import HMM_CONT

params_methods = {
    "hmm_iter": 20,         # number of HMM training iterations; more can improve convergence but costs time
    "n_states": 12,         # number of hidden states
    "normalization": True,  # normalize before estimation
    "num_subj": None,       # optional subject subsampling
    "num_select_nodes": None,# optional ROI subset for speed/memory
}

measure = HMM_CONT(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)
```

#### DHMM (Discrete HMM)

Note: the demo notebook warns that 5 subjects is too small to fit DHMM well; a warning is expected.

```python
from pydfc.dfc_methods import HMM_DISC

params_methods = {
    "W": 44,                       # sliding window length (seconds) used to create observations
    "n_overlap": 0.5,              # overlap fraction for sliding windows
    "sw_method": "pear_corr",      # FC estimator per window
    "tapered_window": True,        # taper window edges
    "clstr_base_measure": "SlidingWindow", # base measure for discretization pipeline
    "hmm_iter": 20,                # HMM training iterations
    "dhmm_obs_state_ratio": 16 / 24, # ratio controlling observation-state discretization relative to hidden states
    "n_states": 12,                # number of hidden states
    "n_subj_clstrs": 5,            # subject-level clustering granularity
    "normalization": True,         # normalize before estimation
    "num_subj": None,              # optional subject subsampling
    "num_select_nodes": 50,        # ROI subset (demo uses 50 here to reduce cost)
}

measure = HMM_DISC(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
dFC.visualize_dFC(TRs=dFC.TR_array[:], normalize=True, fix_lim=False)
```

#### WINDOWLESS

```python
from pydfc.dfc_methods import WINDOWLESS

params_methods = {
    "n_states": 12,          # number of states to estimate
    "normalization": True,   # normalize before estimation
    "num_subj": None,        # optional subject subsampling
    "num_select_nodes": None,# optional ROI subset for speed/memory
}

measure = WINDOWLESS(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)
```

## Response Style Rules

- Keep replies short and practical.
- Prefer one code block at a time (do not dump all methods unless the user asks).
- Reuse the exact demo parameters first; optimize later only if requested.
- If the user is unsure, recommend `SW` first (state-free, simplest).
- Offer a brief method overview before asking them to choose, if they want it.
- After each method snippet, ask: `Are there any other methods you are curious about?`
- Before ending, ask: `Would you like me to extract all code from this chat into a Jupyter notebook (.ipynb) or a Python script (.py)?`

## Failure Handling (Non-Invasive Only)

If the user reports an error:

1. Do not edit repo source files or third-party library source.
2. Ask for the traceback / exact error text.
3. Prefer fixes in this order:
   - environment check (`python --version`, package versions)
   - reinstall steps (`pip install -U pydfc`, dependency install)
   - smaller compute settings (`num_select_nodes`, `num_subj`, `n_jobs`)
   - simpler method (`SW` before state-based methods)
   - parameter adjustments
4. If a package-level bug is suspected, explain the workaround in chat and explicitly avoid source edits.
