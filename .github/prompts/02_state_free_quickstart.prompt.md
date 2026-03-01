# State-Free Quickstart (Single Subject)

Guide the user through the fastest way to run PydFC.

## Steps

1. Confirm PydFC is installed.
2. Provide demo data download commands.

### Jupyter
```python
!curl --create-dirs https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz -o sample_data/sub-0001_bold.nii.gz
