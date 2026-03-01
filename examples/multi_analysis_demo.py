# %% [markdown]
# ## Assessing dynamic Functional Connectivity (dFC) using multiple methodologies
#
# In this demo we will illustrate how to use the tools implemented in `pydfc` toolbox,
# specifically the ones in `multi_analysis_utils` module,
# to assess dFC using multiple methodologies simultaneously.
#
# We use sample data from an openneuro dataset by:
#
# Lukas Snoek and Maite van der Miesen and Andries van der Leij and
# Tinka Beemsterboer and Annemarie Eigenhuis and Steven Scholte (2020).
# AOMIC-PIOP1. OpenNeuro. [Dataset]
# doi: 10.18112/openneuro.ds002785.v2.0.0
#
# Note that in this demo since we are using data from only two subjects, the
# similarity values obtained in the end are not accurate.

import time
import warnings

# %%
# importing necessary libraries
import numpy as np

warnings.filterwarnings("ignore")

# %% [markdown]
# ## Parameters

# %% [markdown]
# 1. Set the default hyperparameter values for different dFC assessment methods

# %%
# setting the default hyperparameter values for different dFC assessment methods
params_methods = {
    # Sliding Parameters
    # W is in sec
    "W": 44,
    "n_overlap": 0.5,
    "sw_method": "pear_corr",
    "tapered_window": True,
    # TIME_FREQ
    "TF_method": "WTC",
    # CLUSTERING AND DHMM
    "clstr_base_measure": "SlidingWindow",
    # HMM
    "hmm_iter": 30,
    "dhmm_obs_state_ratio": 16 / 24,
    # State Parameters
    "n_states": 12,
    "n_subj_clstrs": 13,
    # Parallelization Parameters
    "n_jobs": 2,
    "verbose": 0,
    "backend": "loky",
    # SESSION
    "session": "rest",
    # data parameters
    "normalization": True,
}

# %% [markdown]
# 2. Specify the list of methods you want to include in the multi-analysis.
# You will see how easily you can add or remove methods
# from the multi-analysis by changing this list.

# %%
# selecting methods to be included in the multi-analysis
MEASURES_name_lst = [
    # state-free methods:
    "SlidingWindow",
    # "Time-Freq",
    # state-based methods:
    "CAP",
    "ContinuousHMM",
    "Windowless",
    "Clustering",
    # "DiscreteHMM"
]

# %% [markdown]
# 3. You may set a list of alternative values for each hyperparameter.
# Here for example, we want to run the analysis also with 6 number of states.

# %%
# setting the alternative hyperparameter value for n_states for state-based methods
alter_hparams = {
    "n_states": [6],
}

# %% [markdown]
# ## LOAD DATA

# %% [markdown]
# First, we download resting state fMRI data of 2 subjects from OpenNeuro website.

# %%
# downloading resting state fMRI data of 2 subjects from OpenNeuro website

import subprocess

# Base URL for OpenNeuro dataset
base_url = "https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep"
preproc_suffix = "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
confound_suffix = "desc-confounds_regressors.tsv"

# Download URLs for both subjects
download_cmds = [
    f"{base_url}/sub-0001/func/"
    f"sub-0001_task-restingstate_acq-mb3_{preproc_suffix}?versionId="
    f"UfCs4xtwIEPDgmb32qFbtMokl_jxLUKr -o "
    f"sample_data/sub-0001_task-restingstate_acq-mb3_{preproc_suffix}",
    f"{base_url}/sub-0001/func/"
    f"sub-0001_task-restingstate_acq-mb3_{confound_suffix}?versionId="
    f"biaIJGNQ22P1l1xEsajVzUW6cnu1_8lD -o "
    f"sample_data/sub-0001_task-restingstate_acq-mb3_{confound_suffix}",
    f"{base_url}/sub-0002/func/"
    f"sub-0002_task-restingstate_acq-mb3_{preproc_suffix}?versionId="
    f"fUBWmUTg6vfe2n.ywDNms4mOAW3r6E9Y -o "
    f"sample_data/sub-0002_task-restingstate_acq-mb3_{preproc_suffix}",
    f"{base_url}/sub-0002/func/"
    f"sub-0002_task-restingstate_acq-mb3_{confound_suffix}?versionId="
    f"2zWQIugU.J6ilTFObWGznJdSABbaTx9F -o "
    f"sample_data/sub-0002_task-restingstate_acq-mb3_{confound_suffix}",
]

for cmd in download_cmds:
    subprocess.run(f"curl --create-dirs {cmd}", shell=True)

# %% [markdown]
# Then we load the downloaded nifti files of both subjects as a `TimeSeries` object.

# %%
# loading the downloaded nifti files of both subjects as a `TimeSeries` object
from pydfc import data_loader

subj_id_list = ["sub-0001", "sub-0002"]
nifti_files_list = []
for subj_id in subj_id_list:
    nifti_files_list.append(
        "sample_data/"
        + subj_id
        + "_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )

BOLD_multi = {
    "rest": data_loader.multi_nifti2timeseries(
        nifti_files_list,
        subj_id_list,
        n_rois=100,
        Fs=1 / 0.75,
        confound_strategy="no_motion",
        standardize=False,
        TS_name=None,
        session=None,
    )
}

# %%
# This is how the info dictionary of the loaded TimeSeries look like:
BOLD_multi["rest"].info

# %% [markdown]
# This is how the BOLD signals look like:

# %%
# visualizing the BOLD signals of the first 10 nodes for
# the first 1000 seconds for one subject
BOLD_multi["rest"].get_subj_ts(subjs_id="sub-0001").visualize(
    start_time=0,
    end_time=1000,
    nodes_lst=range(10),
)

# %% [markdown]
# ## Measures of dFC

# %% [markdown]
# Initialize the list of measures with the given default and
# alternative hyperparameter values
#
# We use the `measures_initializer` function from `multi_analysis_utils`
# which initializes the measure objects for the list of all the methods
# passed to it simultaneously.

# %%
# Initialize measures with default and alternative hyperparameter values
from pydfc import multi_analysis_utils

MEASURES_lst, hyper_param_info = multi_analysis_utils.measures_initializer(
    MEASURES_name_lst, params_methods, alter_hparams
)

# %% [markdown]
# There are one instance of each state-free method, and two instances
# of each state-based method,
# one with 12 states and the other one with 6 states.

# %%
# list of measures initialized with the given default and alternative
# hyperparameter values:
print(MEASURES_lst)

# %% [markdown]
# ## Estimate FC states

# %% [markdown]
# Run FCS estimation and collect fitted measures in `MEASURES_fit_lst`.
#
# We use the `estimate_group_FCS` function from `multi_analysis_utils`
# which estimates FCS for all the measures passed to it simultaneously
# for a group of subjects.
#
# You can also estimate FCS for each measure separately using the
# `estimate_FCS` method of each measure object.
#
# Note that state-free methods do not have FCS estimation step,
# so they will be returned in `MEASURES_fit_lst` without any change.

# %%
#
tic = time.time()
print("Measurement Started ...")

MEASURES_fit_lst = multi_analysis_utils.estimate_group_FCS(
    time_series=BOLD_multi["rest"],
    MEASURES_lst=MEASURES_lst,
    n_jobs=None,
    verbose=0,
    backend=None,
)

print(f"Measurement required {time.time() - tic:0.3f} seconds.")

# %% [markdown]
# How info dictionary of a measure looks like (e.g. for the `SlidingWindow` object):

# %%
# printing the info dictionary of the first measure object in `MEASURES_fit_lst`,
# which is the fitted `SlidingWindow` object
MEASURES_fit_lst[0].info

# %% [markdown]
# ## DFC assessment

# %% [markdown]
# Assess each subject's dFC using the fitted measure objects and collect them
# in `SUBJ_dFC_dict`.
#
# We use the `subj_lvl_dFC_assess` function from `multi_analysis_utils` which
# estimates dFC for one subject at a time, and all the measures passed to it
# simultaneously.
#
# You can also estimate dFC for each measure separately using the `estimate_dFC`
# method of each fitted measure object.

# %%
# assessing each subject's dFC using the fitted measure objects and collecting
# them in SUBJ_dFC_dict
tic = time.time()
print("dFC estimation started...")

SUBJ_dFC_dict = {}
for subj in ["sub-0001", "sub-0002"]:
    # get the time series for this subject
    BOLD = {"rest": BOLD_multi["rest"].get_subj_ts(subjs_id=subj)}
    # estimate the dFC for this subject using the fitted measure objects in
    # MEASURES_fit_lst, and the subject_lvl_dFC_assess function from
    # multi_analysis_utils which estimates dFC for one subject at a time,
    # and all the measures passed to it simultaneously
    dFC_dict = multi_analysis_utils.subj_lvl_dFC_assess(
        time_series=BOLD["rest"],
        MEASURES_fit_lst=MEASURES_fit_lst,
        n_jobs=None,
        verbose=0,
        backend=None,
    )
    SUBJ_dFC_dict[subj] = dFC_dict

print("dFC estimation done.")
print(f"Measurement required {time.time() - tic:0.3f} seconds.")

# %% [markdown]
# Visualize dFC obtained by different methods for one of the subjects

# %%
# visualizing dFC obtained by different methods for one of the subjects
from pydfc.dfc_utils import TR_intersection

dFC_lst = SUBJ_dFC_dict["sub-0001"]["dFC_lst"]

TRs = TR_intersection(dFC_lst)
chosen_TRs = TRs[:]

for dFC in dFC_lst:
    if dFC.measure.is_state_based:
        print(
            f"measure: {dFC.measure.measure_name}, num_states: {dFC.measure.params['n_states']}"
        )
    else:
        print(f"measure: {dFC.measure.measure_name}")
    dFC.visualize_dFC(TRs=chosen_TRs, normalize=False, fix_lim=False)

# %% [markdown]
# ## Similarity assessment

# %% [markdown]
# Assess the similarity of dFC obtained by different methods (those with default
# hyperparameter values) for one of the subjects using the `comparison` module

# %%
# assessing the similarity of dFC obtained by different methods
# for one of the subjects using the comparison module
from pydfc.comparison import SimilarityAssessment

dFC_lst = SUBJ_dFC_dict["sub-0001"]["dFC_lst"]
similarity_assessment = SimilarityAssessment(dFC_lst=dFC_lst)

tic = time.time()

print("Similarity measurement started...")
SUBJ_output = similarity_assessment.run(
    FILTERS=hyper_param_info, downsampling_method="default"
)
print("Similarity measurement done.")

print(f"Measurement required {time.time() - tic:0.3f} seconds.")

# %% [markdown]
# This is what similarity results of each subject (`SUBJ_output`) contain:

# %%
#
print(SUBJ_output.keys())
print(SUBJ_output["default_values"].keys())

# %% [markdown]
# ## Results

# %% [markdown]
# First, create a dictionary containing RESULTS using `SUBJ_output` to be visualized

# %%
# creating a dictionary containing RESULTS using SUBJ_output to be visualized
# Similarity metric to use
metric = "spearman"
# Filter for methods with default hyperparameter values
filter = "default_values"

similarity_mat = np.squeeze(SUBJ_output[filter]["all"][metric])
methods_names = [measure.measure_name for measure in SUBJ_output[filter]["measure_lst"]]
RESULTS = {
    "rest": {
        "similarity_mat": similarity_mat,
        "methods_names": methods_names,
    }
}

# %% [markdown]
# Next, visualize Similarity results

# %%
# visualizing the similarity results using the visualize_sim_mat function in
# the comparison module
from pydfc.comparison.plotting import visualize_sim_mat

visualize_sim_mat(
    RESULTS,
    mat_key="similarity_mat",
    name_lst_key="methods_names",
    cmap="viridis",
)
