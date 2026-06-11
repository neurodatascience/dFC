#!/bin/bash
#
#SBATCH --job-name=fmriprep_job       # Name of the job
#SBATCH --output=logs/fmriprep_out.log  # Standard output log
#SBATCH --error=logs/fmriprep_err.log   # Standard error log
#SBATCH --mem-per-cpu=16G                # Memory (16 GB) per cpu
#SBATCH --cpus-per-task=8              # Number of CPU cores (increase based on availability)

module load apptainer

# ---- Cluster configuration (set these for your system) ----
NIPOPPY_VENV_PATH="/path/to/your/nipoppy_venv/bin/activate"
# -----------------------------------------------------------

source "$NIPOPPY_VENV_PATH"

SUBJECT_LIST="./subj_list.txt"

echo "Number subjects found: $(wc -l < $SUBJECT_LIST)"

SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo "Subject ID: $SUBJECT_ID"

nipoppy run \
"$(dirname "$(pwd)")" \
--pipeline fmriprep \
--participant-id $SUBJECT_ID

deactivate
