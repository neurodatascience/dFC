# Choosing a dFC Method and Getting Started with PydFC

> **Citation:** Content in this guide is grounded in:
> Torabi et al., 2024 — *On the variability of dynamic functional connectivity
> assessment methods* — GigaScience —
> <https://doi.org/10.1093/gigascience/giae009>

---

## Quick Start: "Which method should I use?"

There is **no universally best** dFC method — each captures a different aspect
of brain dynamics (Torabi et al., 2024).  Use the decision tree below to pick a
sensible starting point, then validate with at least one method from a different
family.

```
Do you have a single subject, or multiple subjects?
│
├── Single subject  ──► use a STATE-FREE method
│       │
│       ├── Want simplest / fastest?         ──► SW  (Sliding Window)
│       └── Want frequency-specific detail?  ──► TF  (Time-Frequency)
│
└── Multiple subjects ──► use a STATE-BASED method
        │
        ├── Want intuitive co-activation states?  ──► CAP
        ├── Want windowed states?                  ──► SWC
        ├── Want temporal transitions (HMM)?
        │       ├── Continuous observations?       ──► CHMM
        │       └── Discretized observations?      ──► DHMM  (needs ≥10 subjects)
        └── Want to avoid window-size dependence?  ──► WINDOWLESS
```

**If you are completely new → start with SW** (state-free, single subject, fewest
parameters, fastest to run).

---

## Method Summaries

### State-Free Methods (single subject, no fitting required)

| Method | Class | Key parameter | Best for |
|--------|-------|---------------|----------|
| **SW** — Sliding Window | `SLIDING_WINDOW` | `W` (window length, s) | General-purpose continuous FC tracking |
| **TF** — Time-Frequency | `TIME_FREQ` | `TF_method` (e.g. `"WTC"`) | Frequency-specific dynamic FC |

### State-Based Methods (multi-subject, fitting required)

| Method | Class | Key parameter | Best for |
|--------|-------|---------------|----------|
| **CAP** — Co-activation Patterns | `CAP` | `n_states` | Intuitive, instantaneous states |
| **SWC** — Sliding Window + Clustering | `SLIDING_WINDOW_CLUSTR` | `n_states`, `W` | Windowed recurring states |
| **CHMM** — Continuous HMM | `HMM_CONT` | `n_states`, `hmm_iter` | Smooth temporal state transitions |
| **DHMM** — Discrete HMM | `HMM_DISC` | `n_states`, `hmm_iter` | Discretized state sequences (needs more subjects) |
| **WINDOWLESS** | `WINDOWLESS` | `n_states` | State estimation without a fixed window |

---

## Installation

```bash
conda create --name pydfc_env python=3.11
conda activate pydfc_env
pip install pydfc
```

---

## Path A — State-Free Quickstart (single subject)

### 1. Download demo data

**Jupyter notebook cell:**

```python
!curl --create-dirs \
  "https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz?versionId=UfCs4xtwIEPDgmb32qFbtMokl_jxLUKr" \
  -o sample_data/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz

!curl --create-dirs \
  "https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_desc-confounds_regressors.tsv?versionId=biaIJGNQ22P1l1xEsajVzUW6cnu1_8lD" \
  -o sample_data/sub-0001_task-restingstate_acq-mb3_desc-confounds_regressors.tsv
```

*(Remove the leading `!` when running from a terminal.)*

### 2. Load data

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
    confound_strategy="no_motion",  # no_motion, no_motion_no_gsr, or none
    standardize=False,
    TS_name=None,
    session=None,
)

BOLD.visualize(start_time=0, end_time=1000, nodes_lst=range(10))
```

### 3a. Run Sliding Window (SW)

```python
from pydfc.dfc_methods import SLIDING_WINDOW

params_methods = {
    "W": 44,                  # window length (seconds)
    "n_overlap": 0.5,         # overlap fraction between consecutive windows
    "sw_method": "pear_corr", # FC estimator inside each window
    "tapered_window": True,   # taper window edges to reduce boundary artefacts
    "normalization": True,    # normalize before estimation
    "num_select_nodes": None, # optional ROI subset for speed (e.g. 50)
}

measure = SLIDING_WINDOW(**params_methods)
dFC = measure.estimate_dFC(time_series=BOLD)
dFC.visualize_dFC(TRs=dFC.TR_array[:], normalize=False, fix_lim=False)
```

### 3b. Run Time-Frequency (TF)

```python
from pydfc.dfc_methods import TIME_FREQ

params_methods = {
    "TF_method": "WTC",       # time-frequency estimator
    "n_jobs": 2,              # parallel workers
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

---

## Path B — State-Based Quickstart (multiple subjects)

### 1. Download demo data for 5 subjects

```python
import subprocess

base_url = "https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep"
preproc_suffix = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
confound_suffix = "desc-confounds_regressors.tsv"

downloads = [
    (f"sub-0001/func/sub-0001_task-restingstate_acq-mb3_{preproc_suffix}"
     "?versionId=UfCs4xtwIEPDgmb32qFbtMokl_jxLUKr",
     f"sample_data/sub-0001_task-restingstate_acq-mb3_{preproc_suffix}"),
    (f"sub-0001/func/sub-0001_task-restingstate_acq-mb3_{confound_suffix}"
     "?versionId=biaIJGNQ22P1l1xEsajVzUW6cnu1_8lD",
     f"sample_data/sub-0001_task-restingstate_acq-mb3_{confound_suffix}"),
    (f"sub-0002/func/sub-0002_task-restingstate_acq-mb3_{preproc_suffix}"
     "?versionId=fUBWmUTg6vfe2n.ywDNms4mOAW3r6E9Y",
     f"sample_data/sub-0002_task-restingstate_acq-mb3_{preproc_suffix}"),
    (f"sub-0002/func/sub-0002_task-restingstate_acq-mb3_{confound_suffix}"
     "?versionId=2zWQIugU.J6ilTFObWGznJdSABbaTx9F",
     f"sample_data/sub-0002_task-restingstate_acq-mb3_{confound_suffix}"),
    (f"sub-0003/func/sub-0003_task-restingstate_acq-mb3_{preproc_suffix}"
     "?versionId=dfNd8iV0V68yuOibes6qiHxjBgQXhPxi",
     f"sample_data/sub-0003_task-restingstate_acq-mb3_{preproc_suffix}"),
    (f"sub-0003/func/sub-0003_task-restingstate_acq-mb3_{confound_suffix}"
     "?versionId=8OpKFrs_8aJ5cVixokBmuTVKNslgtOXb",
     f"sample_data/sub-0003_task-restingstate_acq-mb3_{confound_suffix}"),
    (f"sub-0004/func/sub-0004_task-restingstate_acq-mb3_{preproc_suffix}"
     "?versionId=0Le8eFwJbcLKaMTQat39bzWcGFhRiyP5",
     f"sample_data/sub-0004_task-restingstate_acq-mb3_{preproc_suffix}"),
    (f"sub-0004/func/sub-0004_task-restingstate_acq-mb3_{confound_suffix}"
     "?versionId=welg1B.VkXHGv06iV56Vp7ezpVTFh2eX",
     f"sample_data/sub-0004_task-restingstate_acq-mb3_{confound_suffix}"),
    (f"sub-0005/func/sub-0005_task-restingstate_acq-mb3_{preproc_suffix}"
     "?versionId=Vwo2YcFvhwbhZktBrPUqi_5BWiR7zcTl",
     f"sample_data/sub-0005_task-restingstate_acq-mb3_{preproc_suffix}"),
    (f"sub-0005/func/sub-0005_task-restingstate_acq-mb3_{confound_suffix}"
     "?versionId=FoBZLbFTZaE3ZjOLZI_4hN4OkEKEZTVf",
     f"sample_data/sub-0005_task-restingstate_acq-mb3_{confound_suffix}"),
]

for url_path, out_path in downloads:
    subprocess.run(
        f'curl --create-dirs "{base_url}/{url_path}" -o {out_path}',
        shell=True,
    )
```

### 2. Load multi-subject data

```python
from pydfc import data_loader
import numpy as np
import warnings

warnings.simplefilter("ignore")

subj_id_list = ["sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005"]
nifti_files_list = [
    f"sample_data/{s}_task-restingstate_acq-mb3_"
    "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    for s in subj_id_list
]

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

### 3. Run a state-based method

Pick one of the snippets below.

#### CAP (recommended first choice)

```python
from pydfc.dfc_methods import CAP

params_methods = {
    "n_states": 12,
    "n_subj_clstrs": 20,
    "normalization": True,
    "num_subj": None,
    "num_select_nodes": None,
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
    "W": 44,
    "n_overlap": 0.5,
    "sw_method": "pear_corr",
    "tapered_window": True,
    "clstr_base_measure": "SlidingWindow",
    "n_states": 12,
    "n_subj_clstrs": 5,
    "normalization": True,
    "num_subj": None,
    "num_select_nodes": None,
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
    "hmm_iter": 20,
    "n_states": 12,
    "normalization": True,
    "num_subj": None,
    "num_select_nodes": None,
}

measure = HMM_CONT(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)
```

#### DHMM (Discrete HMM)

> **Note:** DHMM requires more subjects for stable fitting (10 or more recommended).
> With only 5 subjects a warning is expected.

```python
from pydfc.dfc_methods import HMM_DISC

params_methods = {
    "W": 44,
    "n_overlap": 0.5,
    "sw_method": "pear_corr",
    "tapered_window": True,
    "clstr_base_measure": "SlidingWindow",
    "hmm_iter": 20,
    "dhmm_obs_state_ratio": 16 / 24,
    "n_states": 12,
    "n_subj_clstrs": 5,
    "normalization": True,
    "num_subj": None,
    "num_select_nodes": 50,   # reduced for demo speed
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
    "n_states": 12,
    "normalization": True,
    "num_subj": None,
    "num_select_nodes": None,
}

measure = WINDOWLESS(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)
```

---

## Method Comparison Principles

Following Torabi et al., 2024:

1. **No single best method** — each captures a different aspect of dFC.
2. **Variability across methods** is comparable to variability across subjects or
   time — method choice can influence results as much as the biological signal.
3. Methods cluster into three families:
   - *Group A*: Clustering, CHMM, DHMM — produce similar results
   - *Group B*: CAP, WINDOWLESS — produce similar results
   - *Group C*: SW, TF — produce similar results
4. Methods agree on *what* networks look like (spatial), but not *when* they
   occur (temporal).
5. **Recommendation:** use at least one method from each family to cross-validate
   your findings.

---

## AI-Assisted Guidance

If you use GitHub Copilot, Codex, Claude, or another AI assistant you can get
interactive, copy-paste guidance:

- **GitHub Copilot (VS Code):** run the `/02_choose_method` or
  `/03_state_free_quickstart` prompt from the Copilot Chat panel.
- **Codex / Claude / other:** ask your assistant to follow `docs/SKILL.md`.
- **Any LLM chat:** paste the contents of `docs/SKILL.md` and ask
  "Guide me through a minimal PydFC workflow."

---

## See Also

- `examples/dFC_methods_demo.py` — full demo script
- `examples/multi_analysis_demo.py` — multi-method comparison demo
- `docs/DFC_METHODS_CONTEXT.md` — detailed method assumptions and interpretation
- `docs/PAPER_KNOWLEDGE_BASE.md` — paper-grounded implementation notes
- `docs/SKILL.md` — AI-agent tutorial flow
