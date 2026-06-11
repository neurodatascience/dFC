#!/bin/sh
#
#$ -N assess_dfc_job
#$ -o logs/dfc_out.txt
#$ -e logs/dfc_err.txt
#$ -l h_rt=24:00:00
#$ -l h_vmem=32g
#$ -t 1-NSUBJECTS
#$ -q YOUR_QUEUE

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# -----------------------------------------------------------

SUBJECT_LIST="./subj_list.txt"
DATASET_INFO="./dataset_info.json"
METHODS_CONFIG="./methods_config.json"

echo "Number subjects found: `cat $SUBJECT_LIST | wc -l`"

SUBJECT_ID=`sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST`
echo "Subject ID: $SUBJECT_ID"

# Activate virtual environment
source "$VENV_PATH"

python "$PYDFC_CODE_DIR/task_dFC/dFC_assessment.py" \
--dataset_info $DATASET_INFO \
--methods_config $METHODS_CONFIG \
--participant_id $SUBJECT_ID

deactivate
