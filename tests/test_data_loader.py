import nibabel as nb
import numpy as np
import pytest

from pydfc.data_loader import nifti2timeseries


@pytest.fixture
def simulated_bold_and_label(tmp_path):
    # Simulated BOLD data
    bold_data = np.random.rand(10, 10, 10, 100)
    affine = np.eye(4)
    bold_img = nb.Nifti1Image(bold_data, affine)
    bold_file = tmp_path / "bold.nii.gz"
    bold_img.to_filename(bold_file)

    # Simulated label image with 3 ROIs (labels 1, 2, 3)
    labels = np.zeros((10, 10, 10), dtype=int)
    labels[1:4, 1:4, 1:4] = 1
    labels[5:7, 5:7, 5:7] = 2
    labels[7:9, 1:3, 1:3] = 3
    label_img = nb.Nifti1Image(labels, affine)
    label_file = tmp_path / "labels.nii.gz"
    label_img.to_filename(label_file)

    return str(bold_file), str(label_file)


def test_load(simulated_bold_and_label):
    bold_file, label_file = simulated_bold_and_label
    ts = nifti2timeseries(
        nifti_file=bold_file,
        labels_img=label_file,
        region_names=["1", "2", "3"],
        Fs=1 / 0.75,
        subj_id="sub-0001",
    )
    assert ts is not None
    assert ts.n_regions == 3
