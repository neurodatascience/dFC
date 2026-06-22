#!/bin/bash
#
#$ -N fmriprep_job
#$ -o logs/fmriprep_out.log
#$ -e logs/fmriprep_err.log
#$ -l h_vmem=16g
#$ -pe smp 8
#$ -t 1-NSUBJECTS
#$ -q YOUR_QUEUE

# ---- Cluster configuration (set these for your system) ----
NIPOPPY_VENV_PATH="/path/to/your/nipoppy_venv/bin/activate"
# -----------------------------------------------------------

module load apptainer

source "$NIPOPPY_VENV_PATH"

SUBJECT_LIST="./subj_list.txt"

echo "Number subjects found: $(wc -l < $SUBJECT_LIST)"

SUBJECT_ID=$(sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST)
echo "Subject ID: $SUBJECT_ID"

nipoppy run \
"$(dirname "$(pwd)")" \
--pipeline fmriprep \
--participant-id $SUBJECT_ID

deactivate
