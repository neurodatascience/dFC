#!/bin/sh
#
#$ -cwd
#$ -o logs/ML_out.txt
#$ -e logs/ML_err.txt
#$ -l h_vmem=64G
#$ -q origami.q

DATASET_INFO="./dataset_info.json"

source /data/origami/dFC/anaconda3/etc/profile.d/conda.sh
conda activate pydfc
python "/data/origami/dFC/CODEs/pydfc/dFC/task_dFC/ML.py" \
--dataset_info $DATASET_INFO

conda deactivate
