#!/bin/sh
#
#SBATCH --job-name=report_job   # Optional: Name of your job
#SBATCH --output=logs/report_out.txt  # Standard output log
#SBATCH --error=logs/report_err.txt   # Standard error log
#SBATCH --time=24:00:00                # Walltime for each task (24 hours)
#SBATCH --mem=64G                     # Memory request per node

DATASET_INFO="./dataset_info.json"
SUBJ_LIST="./subj_list.txt"

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/home/achillev/projects/def-jbpoline/achillev/pydfc_env/bin/activate"
PYDFC_CODE_DIR="/home/achillev/scratch/Git_repo"
# -----------------------------------------------------------

# Activate virtual environment
source "$VENV_PATH"

python "$PYDFC_CODE_DIR/task_dFC/generate_report.py" \
--dataset_info $DATASET_INFO \
--subj_list $SUBJ_LIST

deactivate
