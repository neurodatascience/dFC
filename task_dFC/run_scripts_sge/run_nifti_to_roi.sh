#!/bin/sh
#
#$ -N extract_roi_job
#$ -o logs/roi_out.txt
#$ -e logs/roi_err.txt
#$ -l h_rt=24:00:00
#$ -l h_vmem=64g
#$ -t 1-NSUBJECTS
#$ -q YOUR_QUEUE

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# -----------------------------------------------------------

# -----------------------------
# Inputs
# -----------------------------
SUBJECT_LIST="./subj_list.txt"
DATASET_INFO="./dataset_info.json"
DENOISING_STRATEGY=${1:-simple}

echo "Denoising strategy: $DENOISING_STRATEGY"
echo "Number of subjects: $(wc -l < "$SUBJECT_LIST")"

SUBJECT_ID=$(sed -n "${SGE_TASK_ID}p" "$SUBJECT_LIST")
echo "Subject ID: $SUBJECT_ID"

# -----------------------------
# Environment
# -----------------------------
source "$VENV_PATH"

python "$PYDFC_CODE_DIR/task_dFC/nifti_to_roi_signal.py" \
    --dataset_info $DATASET_INFO \
    --participant_id $SUBJECT_ID \
    --denoising_strategy $DENOISING_STRATEGY

deactivate
