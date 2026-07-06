#!/bin/bash
#
#SBATCH --job-name=assess_dfc_job
#SBATCH --output=logs/dfc_out_%A_%a.txt
#SBATCH --error=logs/dfc_err_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --requeue

SUBJECT_LIST="./subj_list.txt"
DATASET_INFO="./dataset_info.json"
METHODS_CONFIG="./methods_config.json"

echo "Number subjects found: $(cat $SUBJECT_LIST | wc -l)"

SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo "Subject ID: $SUBJECT_ID"

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# -----------------------------------------------------------

# Activate virtual environment
source "$VENV_PATH"

python "$PYDFC_CODE_DIR/task_dFC/dFC_assessment.py" \
--dataset_info $DATASET_INFO \
--methods_config $METHODS_CONFIG \
--participant_id $SUBJECT_ID

deactivate
