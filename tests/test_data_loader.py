import pytest

import nibabel as nb
import numpy as np

from pydfc.data_loader import nifti2timeseries


# @pytest.fixture(scope="session")
# def rest_file(tmp_path_factory):
#     URL = "https://s3.amazonaws.com/openneuro.org/ds002785/derivatives/fmriprep/sub-0001/func/sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz?versionId=UfCs4xtwIEPDgmb32qFbtMokl_jxLUKr"
#     tmpdir = tmp_path_factory.mktemp("data")
#     file_path = tmpdir / "sub-0001_task-restingstate_acq-mb3_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
#     with httpx.stream("GET", URL) as response:
#         with file_path.open("wb") as f:
#             for chunk in response.iter_bytes():
#                 f.write(chunk)
#
#     return file_path


@pytest.fixture
def simulated_bold_data(tmp_path):
    img = nb.Nifti1Image(np.random.rand(10, 10, 10, 100), np.eye(4))
    img.to_filename(tmp_path / "simulated_bold.nii.gz")
    return tmp_path / "simulated_bold.nii.gz"


def test_load(simulated_bold_data):
    timeseries = nifti2timeseries(
        nifti_file=str(simulated_bold_data),
        n_rois=100,
        Fs=1 / 0.75,
        subj_id="sub-0001",
    )
