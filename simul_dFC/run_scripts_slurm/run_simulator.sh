#!/bin/bash
#
#SBATCH --job-name=simul_dfc_job   # Optional: Name of your job
#SBATCH --output=logs/simul_out.txt  # Standard output log
#SBATCH --error=logs/simul_err.txt   # Standard error log
#SBATCH --account=def-jbpoline           # Account
#SBATCH --time=24:00:00                # Walltime for each task (24 hours)
#SBATCH --mem=8G                     # Memory request per node
#SBATCH --array=1-200                # Task array specification

SUBJECT_LIST="./subj_list.txt"
DATASET_INFO="./dataset_info.json"
TASKS_INFO="./tasks_info.json"

SUBJECT_ID=`sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST`
echo "Subject ID: $SUBJECT_ID"

# Activate  virtual environment
source "/home/mt00/venvs/pydfc/bin/activate"

# Run Python script
python "/home/mt00/pydfc/dFC/simul_dFC/task_data_simulator.py" \
--dataset_info $DATASET_INFO \
--tasks_info $TASKS_INFO \
--participant_id $SUBJECT_ID

# Deactivate environment
deactivate
