#!/bin/sh
#
#SBATCH --job-name=extract_roi_job
#SBATCH --output=logs/roi_out.txt
#SBATCH --error=logs/roi_err.txt
#SBATCH --time=24:00:00
#SBATCH --mem=64G

# -----------------------------
# Inputs
# -----------------------------
SUBJECT_LIST="./subj_list.txt"
DATASET_INFO="./dataset_info.json"
DENOISING_STRATEGY=${1:-simple}

echo "Denoising strategy: $DENOISING_STRATEGY"
echo "Number of subjects: $(wc -l < "$SUBJECT_LIST")"

SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SUBJECT_LIST")
echo "Subject ID: $SUBJECT_ID"

# -----------------------------
# Environment
# -----------------------------
source "/home/mt00/venvs/pydfc/bin/activate"

python "/home/mt00/pydfc/dFC/task_dFC/nifti_to_roi_signal.py" \
    --dataset_info $DATASET_INFO \
    --participant_id $SUBJECT_ID \
    --denoising_strategy $DENOISING_STRATEGY

deactivate
