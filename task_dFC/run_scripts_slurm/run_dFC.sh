#!/bin/sh
#
#SBATCH --job-name=assess_dfc_job   # Optional: Name of your job
#SBATCH --output=logs/dfc_out.txt  # Standard output log
#SBATCH --error=logs/dfc_err.txt   # Standard error log
#SBATCH --time=24:00:00                # Walltime for each task (24 hours)
#SBATCH --mem=32G                     # Memory request per node

SUBJECT_LIST="./subj_list.txt"
DATASET_INFO="./dataset_info.json"
METHODS_CONFIG="./methods_config.json"

echo "Number subjects found: `cat $SUBJECT_LIST | wc -l`"

SUBJECT_ID=`sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST`
echo "Subject ID: $SUBJECT_ID"

# Activate  virtual environment
source "/home/mt00/venvs/pydfc/bin/activate"

python "/home/mt00/pydfc/dFC/task_dFC/dFC_assessment.py" \
--dataset_info $DATASET_INFO \
--methods_config $METHODS_CONFIG \
--participant_id $SUBJECT_ID

deactivate
