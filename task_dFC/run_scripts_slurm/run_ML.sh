#!/bin/sh
#
#SBATCH --cpus-per-task=8  # Number of CPU cores per task
#SBATCH --output=logs/ML_out.txt  # Standard output log
#SBATCH --error=logs/ML_err.txt   # Standard error log
#SBATCH --mem=128G                     # Memory request per node

DATASET_INFO="./dataset_info.json"

# Activate  virtual environment
module load python/3.11.5
source "/home/mt00/venvs/pydfc/bin/activate"

/home/mt00/venvs/pydfc/bin/python "/home/mt00/pydfc/dFC/task_dFC/ML.py" \
--dataset_info $DATASET_INFO

deactivate
