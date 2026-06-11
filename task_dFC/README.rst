.. raw:: html

   <a href="https://github.com/neurodatascience/dFC"><img src="https://img.shields.io/badge/GitHub-neurodatascience%2FdFC-blue.svg" alt="GitHub Repository"></a>

=======================================================
PydFC: task_dFC Module Documentation
=======================================================

The ``task_dFC`` module provides a scalable, open-source Python solution for the **large-scale benchmarking and application of dynamic functional connectivity (dFC) methods**.

Its core purpose is to apply end-to-end analytical workflows to fMRI data to assess the efficacy of various dFC methodologies in **predicting moment-to-moment cognitive states** — specifically, distinguishing between moments of task engagement versus rest at the single repetition time (TR) resolution.

Methods Implemented
-------------------

The module supports a diverse selection of seven well-established dFC methodologies implemented within the PydFC toolbox:

*   **State-free Methods:** Designed to capture continuous fluctuations in connectivity.

    *   Sliding Window (SW)
    *   Time-Frequency (TF)

*   **State-based Methods:** Designed to identify recurring, discrete connectivity patterns or states.

    *   Co-Activation Patterns (CAP)
    *   Clustering (SWC)
    *   Continuous Hidden Markov Models (CHMM)
    *   Discrete Hidden Markov Models (DHMM)
    *   Windowless (WL)

Prerequisites
-------------

*   Preprocessed fMRI data in BIDS format with ``events.tsv`` files (e.g., via fMRIPrep).
*   PydFC installed (see the root ``README.rst``).
*   For fMRIPrep preprocessing: ``nipoppy`` installed and configured.

Configuration Files
-------------------

Before running the pipeline, fill in the following JSON configuration files located in ``run_scripts_slurm/`` or ``run_scripts_sge/``:

*   ``dataset_info.json`` — dataset name, root paths, sessions, tasks, runs, and BOLD suffix.
*   ``methods_config.json`` — dFC method parameters, method list, and parallelism settings.
*   ``multi_dataset_info.json`` — paths and dataset lists for cross-dataset analysis.
*   ``global_config.json`` — nipoppy configuration for fMRIPrep (containers, FreeSurfer license, TemplateFlow).

Analysis Pipeline: Script-Based Workflow
-----------------------------------------

The ``task_dFC`` workflow starts assuming that fMRI data (in BIDS format with ``events.tsv``) has undergone standard preprocessing (via fMRIPrep). The subsequent analysis is executed sequentially through the following scripts:

1. ``nifti_to_roi_signal.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Function:** Runs denoising and extracts regional BOLD time series from preprocessed NIfTI data.

**Details:** Voxel-wise BOLD signals are parcellated, typically using an atlas such as the Schaefer 100-region atlas, yielding regional time series that serve as the input for dFC assessment.

2. ``FCS_estimate.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Function:** Estimates Functional Connectivity States (FCS).

**Details:** This script fits the dFC model required by **state-based methodologies** (CAP, HMM, Clustering) that rely on identifying **group-level recurring patterns**. Must be run before ``dFC_assessment.py``.

3. ``dFC_assessment.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Function:** Computes individual-level dFC patterns.

**Details:** Applies the seven implemented dFC methodologies (SW, TF, CAP, etc.) to the BOLD signals of each run and subject to obtain the corresponding high-dimensional dFC patterns.

4. ``ML.py``
~~~~~~~~~~~~~~~~~~~~
**Function:** Implements the core machine learning pipeline, including cognitive state labeling, feature extraction, supervised classification, and separability analysis.

**A. Task Presence Labeling**

*   Initial stimulus timings from ``events.tsv`` are convolved with a canonical **Hemodynamic Response Function (HRF)** to account for hemodynamic delay.
*   The HRF-convolved signal is binarized using a **Gaussian Mixture Model (GMM)** to assign time points as "rest" or "task-present". This process critically identifies and removes ambiguous **"gray zone" time points** corresponding to transitions, improving classifier performance.

**B. Feature Extraction and Reduction**

*   **State-free Methods (SW, TF):** DFC matrices are vectorized (e.g., 4950 connections for Schaefer 100-region atlas). **Laplacian Eigenmaps (LE)** dimensionality reduction is applied to make the high-dimensional discriminative information accessible to classifiers.
*   **State-based Methods (CAP, HMM, etc.):** Features are derived from state probabilities, distances from states, or state weights. These resulting compositional features (shape ``(time, states)``) are transformed using an **isometric log-ratio (ILR) transformation**.

**C. Prediction and Evaluation**

*   A **Support Vector Machine (SVM) with an RBF kernel** is trained to predict the cognitive state (rest vs. task) at the single-TR level.
*   **Balanced Accuracy** is used as the primary metric, ensuring chance performance is 50%.
*   **Cognitive State Separability** is quantified using the **Silhouette Index (SI)** to evaluate whether task and rest samples are intrinsically distinguishable in the feature space without supervision.

5. ``generate_report.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Function:** Summarizes classification efficacy and separability results for individual datasets and paradigms.

**Details:** Generates figures, tables, and reports (e.g., heatmaps and boxplots) documenting Balanced Accuracy and SI scores across methods and paradigms.

6. ``multi_dataset_analysis/``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**Function:** Aggregates and compares results across multiple datasets and paradigms.

**Details:** Facilitates **large-scale benchmarking** by calculating aggregate performance statistics across datasets and task paradigms. Run via ``run_scripts_slurm/run_across_dataset_analysis.sh``.

Available scripts:

*   ``performance_predict.py`` — prediction accuracy across datasets and methods
*   ``performance_factor.py`` — factors driving performance differences
*   ``ml_results.py`` — ML pipeline result summaries
*   ``dfc_visualization.py`` — dFC pattern visualizations
*   ``embedding_visualization.py`` — low-dimensional embedding visualizations
*   ``sample_matrix_visualization.py`` — sample dFC matrix plots
*   ``task_presence_binarization.py`` — task label binarization diagnostics
*   ``task_timing_stats.py`` — task timing statistics
*   ``cohensd.py`` — effect size analysis

Running the Pipeline
--------------------

Cluster job scripts are provided in two formats:

*   ``run_scripts_slurm/`` — for SLURM-based clusters (e.g., Compute Canada / Alliance)
*   ``run_scripts_sge/`` — for SGE-based clusters

Each script contains a **"Cluster configuration"** block at the top where you set your virtual environment path and pydfc code directory before submitting. Array jobs (``run_dFC.sh``, ``run_nifti_to_roi.sh``, ``run_fmriprep.sh``) expect a ``subj_list.txt`` file listing one subject ID per line.

Typical submission order::

    sbatch run_fmriprep.sh         # (optional) if preprocessing not done
    sbatch run_nifti_to_roi.sh
    sbatch run_FCS.sh
    sbatch --array=1-N run_dFC.sh
    sbatch run_ML.sh
    sbatch run_report.sh
    sbatch run_across_dataset_analysis.sh <script_name>
