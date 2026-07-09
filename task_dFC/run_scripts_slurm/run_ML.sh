#!/bin/bash
#
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ML_out_%A_%a.txt   # %A = array job ID, %a = task ID
#SBATCH --error=logs/ML_err_%A_%a.txt
#SBATCH --mem=128G
#SBATCH --requeue

DATASET_INFO="./dataset_info.json"

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# -----------------------------------------------------------

# Activate virtual environment
source "$VENV_PATH"

python "$PYDFC_CODE_DIR/task_dFC/ML.py" \
--dataset_info $DATASET_INFO

deactivate
