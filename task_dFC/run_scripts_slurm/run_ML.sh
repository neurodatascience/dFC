#!/bin/sh
#
#SBATCH --cpus-per-task=8  # Number of CPU cores per task
#SBATCH --output=logs/ML_out.txt  # Standard output log
#SBATCH --error=logs/ML_err.txt   # Standard error log
#SBATCH --mem=128G                     # Memory request per node

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
