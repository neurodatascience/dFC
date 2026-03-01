# %% [markdown]
# ## Assessing dFC using individual methodologies
#
# In this demo we will illustrate how to use each of the dFC assessment methods
# implemented in pydfc toolbox.
#
# We use sample data from an openneuro dataset by: Lukas Snoek and Maite van der
# Miesen and Andries van der Leij and Tinka Beemsterboer and Annemarie Eigenhuis
# and Steven Scholte (2020). AOMIC-PIOP1. OpenNeuro. [Dataset]
# doi: 10.18112/openneuro.ds002785.v2.0.0

import warnings

import matplotlib.pyplot as plt
import numpy as np

# %%
# Importing necessary libraries
from pydfc import data_loader

warnings.simplefilter("ignore")

# %% [markdown]
# ## STATE-FREE METHODS
# ----------------------------
# These methods do not require any model fitting or brain states estimation.
# Therefore, they can be simply applied on each single subject's data.

# %% [markdown]
# First let's download the fmriprep processed functional data of sub-0001
# from openneuro website.

# %%
# loading data of one subject from nifti file

import subprocess

base_url = "https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep"
preproc_suffix = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
confound_suffix = "desc-confounds_regressors.tsv"

# Download commands for sub-0001
download_cmds = [
    f"{base_url}/sub-0001/func/sub-0001_task-restingstate_acq-mb3_{preproc_suffix}"
    f"?versionId=UfCs4xtwIEPDgmb32qFbtMokl_jxLUKr -o "
    f"sample_data/sub-0001_task-restingstate_acq-mb3_{preproc_suffix}",
    f"{base_url}/sub-0001/func/sub-0001_task-restingstate_acq-mb3_{confound_suffix}"
    f"?versionId=biaIJGNQ22P1l1xEsajVzUW6cnu1_8lD -o "
    f"sample_data/sub-0001_task-restingstate_acq-mb3_{confound_suffix}",
]

for cmd in download_cmds:
    subprocess.run(f"curl --create-dirs {cmd}", shell=True)

# Load sub-0001 data from nifti file

# %% [markdown]
# Next we load the downloaded nifti files as a TIME_SERIES object and name
# it BOLD.

# %%
# load sub-0001 data from nifti file
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

# %% [markdown]
# ### Sliding Window (SW)
# ----------------------------
# %%
# estimating dFC using Sliding Window method
from pydfc.dfc_methods import SLIDING_WINDOW

params_methods = {
    # W is window length in sec
    "W": 44,
    "n_overlap": 0.5,
    "sw_method": "pear_corr",
    "tapered_window": True,
    # data Parameters
    "normalization": True,
    "num_select_nodes": None,  # set to e.g. 50 for faster computation
}

measure = SLIDING_WINDOW(**params_methods)
dFC = measure.estimate_dFC(time_series=BOLD)
dFC.visualize_dFC(
    TRs=dFC.TR_array[:], normalize=False, fix_lim=False
)  # TRs are time indices

# %%
# Example: Plotting FC averaged across all connections over time using the
# obtained dFC
# ----------------------------
avg_dFC = np.mean(np.mean(dFC.get_dFC_mat(), axis=1), axis=1)
plt.figure(figsize=(10, 3))
plt.plot(dFC.TR_array, avg_dFC)
plt.show()

# %% [markdown]
# ### Time-Frequency (TF)
# ----------------------------
# %%
# estimating dFC using Time-Frequency method
from pydfc.dfc_methods import TIME_FREQ

params_methods = {
    "TF_method": "WTC",
    # Parallelization Parameters
    "n_jobs": 2,
    "verbose": 0,
    "backend": "loky",
    # Data Parameters
    "normalization": True,
    "num_select_nodes": None,  # set to e.g. 50 for faster computation
}

measure = TIME_FREQ(**params_methods)
dFC = measure.estimate_dFC(time_series=BOLD)
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)

# %% [markdown]
# ## STATE-BASED METHODS
# ----------------------------
# The state-based methods require an initial model fitting and functional
# connectivity states (FCS) estimation on multiple or all subjects.
# You can specify the assumed number of brain states by setting "n_states"

# %% [markdown]
# First let's download the func data of 5 subjects.

# %%
# loading data of multiple subjects from nifti files

base_url = "https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep"
preproc_suffix = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
confound_suffix = "desc-confounds_regressors.tsv"

# Subject 1
download_cmds = [
    f"{base_url}/sub-0001/func/sub-0001_task-restingstate_acq-mb3_{preproc_suffix}"
    f"?versionId=UfCs4xtwIEPDgmb32qFbtMokl_jxLUKr -o "
    f"sample_data/sub-0001_task-restingstate_acq-mb3_{preproc_suffix}",
    f"{base_url}/sub-0001/func/sub-0001_task-restingstate_acq-mb3_{confound_suffix}"
    f"?versionId=biaIJGNQ22P1l1xEsajVzUW6cnu1_8lD -o "
    f"sample_data/sub-0001_task-restingstate_acq-mb3_{confound_suffix}",
]

# Subject 2
download_cmds.extend(
    [
        f"{base_url}/sub-0002/func/sub-0002_task-restingstate_acq-mb3_{preproc_suffix}"
        f"?versionId=fUBWmUTg6vfe2n.ywDNms4mOAW3r6E9Y -o "
        f"sample_data/sub-0002_task-restingstate_acq-mb3_{preproc_suffix}",
        f"{base_url}/sub-0002/func/sub-0002_task-restingstate_acq-mb3_{confound_suffix}"
        f"?versionId=2zWQIugU.J6ilTFObWGznJdSABbaTx9F -o "
        f"sample_data/sub-0002_task-restingstate_acq-mb3_{confound_suffix}",
    ]
)

# Subject 3
download_cmds.extend(
    [
        f"{base_url}/sub-0003/func/sub-0003_task-restingstate_acq-mb3_{preproc_suffix}"
        f"?versionId=dfNd8iV0V68yuOibes6qiHxjBgQXhPxi -o "
        f"sample_data/sub-0003_task-restingstate_acq-mb3_{preproc_suffix}",
        f"{base_url}/sub-0003/func/sub-0003_task-restingstate_acq-mb3_{confound_suffix}"
        f"?versionId=8OpKFrs_8aJ5cVixokBmuTVKNslgtOXb -o "
        f"sample_data/sub-0003_task-restingstate_acq-mb3_{confound_suffix}",
    ]
)

# Subject 4
download_cmds.extend(
    [
        f"{base_url}/sub-0004/func/sub-0004_task-restingstate_acq-mb3_{preproc_suffix}"
        f"?versionId=0Le8eFwJbcLKaMTQat39bzWcGFhRiyP5 -o "
        f"sample_data/sub-0004_task-restingstate_acq-mb3_{preproc_suffix}",
        f"{base_url}/sub-0004/func/sub-0004_task-restingstate_acq-mb3_{confound_suffix}"
        f"?versionId=welg1B.VkXHGv06iV56Vp7ezpVTFh2eX -o "
        f"sample_data/sub-0004_task-restingstate_acq-mb3_{confound_suffix}",
    ]
)

# Subject 5
download_cmds.extend(
    [
        f"{base_url}/sub-0005/func/sub-0005_task-restingstate_acq-mb3_{preproc_suffix}"
        f"?versionId=Vwo2YcFvhwbhZktBrPUqi_5BWiR7zcTl -o "
        f"sample_data/sub-0005_task-restingstate_acq-mb3_{preproc_suffix}",
        f"{base_url}/sub-0005/func/sub-0005_task-restingstate_acq-mb3_{confound_suffix}"
        f"?versionId=FoBZLbFTZaE3ZjOLZI_4hN4OkEKEZTVf -o "
        f"sample_data/sub-0005_task-restingstate_acq-mb3_{confound_suffix}",
    ]
)

for cmd in download_cmds:
    subprocess.run(f"curl --create-dirs {cmd}", shell=True)

# %% [markdown]
# Next we load the downloaded nifti files as TIME_SERIES object containing
# multiple subjects' BOLD signals and name it BOLD_multi.

# %%
# loading data of multiple subjects from their niifti files
subj_id_list = ["sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005"]
nifti_files_list = []
for subj_id in subj_id_list:
    nifti_files_list.append(
        f"sample_data/{subj_id}_task-restingstate_acq-mb3_"
        "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
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

# %% [markdown]
# ### Co-Activation Patterns (CAP)
# ----------------------------
# %%
# estimating dFC using Co-Activation Patterns method
from pydfc.dfc_methods import CAP

params_methods = {
    # State Parameters
    "n_states": 12,
    "n_subj_clstrs": 20,
    # Data Parameters
    "normalization": True,
    "num_subj": None,  # set to e.g. 2 for faster computation
    "num_select_nodes": None,  # set to e.g. 50 for faster computation
}

measure = CAP(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)

# %% [markdown]
# ### Sliding Window + Clustering (SWC)
# ----------------------------
# %%
# estimating dFC using Sliding Window + Clustering method
from pydfc.dfc_methods import SLIDING_WINDOW_CLUSTR

params_methods = {
    # Sliding Parameters
    # W is window length in sec
    "W": 44,
    "n_overlap": 0.5,
    "sw_method": "pear_corr",
    "tapered_window": True,
    # CLUSTERING
    "clstr_base_measure": "SlidingWindow",
    # State Parameters
    "n_states": 12,
    "n_subj_clstrs": 5,
    # Data Parameters
    "normalization": True,
    "num_subj": None,  # set to e.g. 2 for faster computation
    "num_select_nodes": None,  # set to e.g. 50 for faster computation
}

measure = SLIDING_WINDOW_CLUSTR(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
dFC.visualize_dFC(TRs=dFC.TR_array[:], normalize=True, fix_lim=False)

# %% [markdown]
# ### Continuous Hidden Markov Model (CHMM)
# ----------------------------
# Note: 5 subjects is too small for CHMM. Set num_select_nodes to 20.
# %%
# estimating dFC using Continuous Hidden Markov Model method
from pydfc.dfc_methods import HMM_CONT

params_methods = {
    # HMM
    "hmm_iter": 20,
    # State Parameters
    "n_states": 12,
    # Data Parameters
    "normalization": True,
    "num_subj": None,  # set to e.g. 2 for faster computation
    "num_select_nodes": 20,  # set to 20 for faster computation
}

measure = HMM_CONT(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)

# %% [markdown]
# ### Discrete Hidden Markov Model (DHMM)
# ----------------------------

# %%
# estimating dFC using Discrete Hidden Markov Model method
from pydfc.dfc_methods import HMM_DISC

params_methods = {
    # Sliding Parameters
    # W is window length in sec
    "W": 44,
    "n_overlap": 1.0,
    "sw_method": "pear_corr",
    "tapered_window": True,
    # CLUSTERING AND DHMM
    "clstr_base_measure": "SlidingWindow",
    # HMM
    "hmm_iter": 20,
    "dhmm_obs_state_ratio": 16 / 24,
    # State Parameters
    "n_states": 12,
    "n_subj_clstrs": 5,
    # Data Parameters
    "normalization": True,
    "num_subj": None,  # set to e.g. 2 for faster computation
    "num_select_nodes": 20,  # set to 20 for faster computation
}

measure = HMM_DISC(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
dFC.visualize_dFC(TRs=dFC.TR_array[::29], normalize=True, fix_lim=False)

# %% [markdown]
# ### Window-less (WL)
# ----------------------------
# %%
# estimating dFC using Window-less method
from pydfc.dfc_methods import WINDOWLESS

params_methods = {
    # State Parameters
    "n_states": 12,
    # Data Parameters
    "normalization": True,
    "num_subj": None,  # set to e.g. 2 for faster computation
    "num_select_nodes": None,  # set to e.g. 50 for faster computation
}

measure = WINDOWLESS(**params_methods)
measure.estimate_FCS(time_series=BOLD_multi)
dFC = measure.estimate_dFC(time_series=BOLD_multi.get_subj_ts(subjs_id="sub-0001"))
TRs = dFC.TR_array[np.arange(29, 480 - 29, 29)]
dFC.visualize_dFC(TRs=TRs, normalize=True, fix_lim=False)
