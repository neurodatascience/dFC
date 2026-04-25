#!/bin/sh
#
#SBATCH --job-name=report_job   # Optional: Name of your job
#SBATCH --output=logs/report_out.txt  # Standard output log
#SBATCH --error=logs/report_err.txt   # Standard error log
#SBATCH --time=24:00:00                # Walltime for each task (24 hours)
#SBATCH --mem=64G                     # Memory request per node

DATASET_INFO="./dataset_info.json"
SUBJ_LIST="./subj_list.txt"

# Activate  virtual environment
source "/home/mt00/venvs/pydfc/bin/activate"

python "/home/mt00/pydfc/dFC/task_dFC/generate_report.py" \
--dataset_info $DATASET_INFO \
--subj_list $SUBJ_LIST

deactivate
