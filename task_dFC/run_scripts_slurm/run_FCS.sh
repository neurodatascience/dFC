#!/bin/bash
#
#SBATCH --job-name=fit_fcs_job
#SBATCH --output=logs/fcs_out_%A_%a.txt
#SBATCH --error=logs/fcs_err_%A_%a.txt
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue

DATASET_INFO="./dataset_info.json"
METHODS_CONFIG="./methods_config.json"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/home/achillev/projects/def-jbpoline/achillev/pydfc_env/bin/activate"
PYDFC_CODE_DIR="/home/achillev/scratch/Git_repo"
# -----------------------------------------------------------

# Activate virtual environment
source "$VENV_PATH"

python "$PYDFC_CODE_DIR/task_dFC/FCS_estimate.py" \
--dataset_info $DATASET_INFO \
--methods_config $METHODS_CONFIG

deactivate
