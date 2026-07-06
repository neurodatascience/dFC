#!/bin/bash
#
#SBATCH --job-name=simul_dfc_job   # Optional: Name of your job
#SBATCH --output=logs/simul_out.txt  # Standard output log
#SBATCH --error=logs/simul_err.txt   # Standard error log
#SBATCH --account=YOUR_ACCOUNT           # Account
#SBATCH --time=24:00:00                # Walltime for each task (24 hours)
#SBATCH --mem=8G                     # Memory request per node
#SBATCH --array=1-200                # Task array specification

SUBJECT_LIST="./subj_list.txt"
DATASET_INFO="./dataset_info.json"
TASKS_INFO="./tasks_info.json"

SUBJECT_ID=`sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST`
echo "Subject ID: $SUBJECT_ID"

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# -----------------------------------------------------------

# Activate virtual environment
source "$VENV_PATH"

# Run Python script
python "$PYDFC_CODE_DIR/simul_dFC/task_data_simulator.py" \
--dataset_info $DATASET_INFO \
--tasks_info $TASKS_INFO \
--participant_id $SUBJECT_ID

# Deactivate environment
deactivate
