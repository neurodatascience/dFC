#!/bin/bash
#
#SBATCH --job-name=fmriprep_job       # Name of the job
#SBATCH --output=logs/fmriprep_out.log  # Standard output log
#SBATCH --error=logs/fmriprep_err.log   # Standard error log
#SBATCH --mem-per-cpu=16G                # Memory (16 GB) per cpu
#SBATCH --cpus-per-task=8              # Number of CPU cores (increase based on availability)

module load apptainer

source "/home/mt00/venvs/nipoppy_env/bin/activate"

SUBJECT_LIST="./subj_list.txt"

echo "Number subjects found: $(wc -l < $SUBJECT_LIST)"

SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo "Subject ID: $SUBJECT_ID"

nipoppy run \
"$(dirname "$(pwd)")" \
--pipeline fmriprep \
--participant-id $SUBJECT_ID

deactivate
