#!/bin/sh
#
#$ -N ml_job
#$ -o logs/ML_out.txt
#$ -e logs/ML_err.txt
#$ -pe smp 8
#$ -l h_vmem=16g
#$ -q YOUR_QUEUE

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# -----------------------------------------------------------

DATASET_INFO="./dataset_info.json"

# Activate virtual environment
source "$VENV_PATH"

python "$PYDFC_CODE_DIR/task_dFC/ML.py" \
--dataset_info $DATASET_INFO

deactivate
