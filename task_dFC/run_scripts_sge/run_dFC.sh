#!/bin/sh
#
#$ -cwd
#$ -o logs/dfc_out.txt
#$ -e logs/dfc_err.txt
#$ -l h_vmem=32G
#$ -q origami.q

SUBJECT_LIST="./subj_list.txt"
DATASET_INFO="./dataset_info.json"

echo "Number subjects found: `cat $SUBJECT_LIST | wc -l`"

SUBJECT_ID=`sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST`
echo "Subject ID: $SUBJECT_ID"

source /data/origami/dFC/anaconda3/etc/profile.d/conda.sh
conda activate pydfc
python "/data/origami/dFC/CODEs/pydfc/dFC/task_dFC/dFC_assessment.py" \
--dataset_info $DATASET_INFO \
--participant_id $SUBJECT_ID

conda deactivate
