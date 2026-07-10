#!/bin/bash
#
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ML_out_%A_%a.txt   # %A = array job ID, %a = task ID
#SBATCH --error=logs/ML_err_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --requeue

DATASET_INFO="./dataset_info.json"

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/home/achillev/projects/def-jbpoline/achillev/pydfc_env/bin/activate"
PYDFC_CODE_DIR="/home/achillev/scratch/Git_repo"
# -----------------------------------------------------------

# Activate virtual environment
source "$VENV_PATH"

python "$PYDFC_CODE_DIR/task_dFC/ML.py" \
--dataset_info $DATASET_INFO

deactivate
