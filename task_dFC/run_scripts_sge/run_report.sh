#!/bin/sh
#
#$ -N report_job
#$ -o logs/report_out.txt
#$ -e logs/report_err.txt
#$ -l h_rt=24:00:00
#$ -l h_vmem=64g
#$ -q YOUR_QUEUE

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# -----------------------------------------------------------

DATASET_INFO="./dataset_info.json"
SUBJ_LIST="./subj_list.txt"

# Activate virtual environment
source "$VENV_PATH"

python "$PYDFC_CODE_DIR/task_dFC/generate_report.py" \
--dataset_info $DATASET_INFO \
--subj_list $SUBJ_LIST

deactivate
