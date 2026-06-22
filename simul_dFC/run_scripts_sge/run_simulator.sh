#!/bin/bash
#
#$ -N simul_dfc_job
#$ -o logs/simul_out.txt
#$ -e logs/simul_err.txt
#$ -l h_rt=24:00:00
#$ -l h_vmem=8g
#$ -t 1-200
#$ -q YOUR_QUEUE

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# For conda environments, replace the two lines above with:
#   CONDA_SH="/path/to/conda/etc/profile.d/conda.sh"
#   CONDA_ENV="pydfc"
# -----------------------------------------------------------

SUBJECT_LIST="./subj_list.txt"
DATASET_INFO="./dataset_info.json"
TASKS_INFO="./tasks_info.json"

SUBJECT_ID=`sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST`
echo "Subject ID: $SUBJECT_ID"

# Activate virtual environment
source "$VENV_PATH"
# For conda: source "$CONDA_SH" && conda activate "$CONDA_ENV"

# Run Python script
python "$PYDFC_CODE_DIR/simul_dFC/task_data_simulator.py" \
--dataset_info $DATASET_INFO \
--tasks_info $TASKS_INFO \
--participant_id $SUBJECT_ID

# Deactivate environment
deactivate
