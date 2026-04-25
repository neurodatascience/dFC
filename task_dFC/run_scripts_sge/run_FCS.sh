#!/bin/sh
#
#$ -cwd
#$ -o logs/fcs_out.txt
#$ -e logs/fcs_err.txt
#$ -l h_vmem=64G
#$ -q origami.q

DATASET_INFO="./dataset_info.json"
METHODS_CONFIG="./methods_config.json"

source /data/origami/dFC/anaconda3/etc/profile.d/conda.sh
conda activate pydfc
python "/data/origami/dFC/CODEs/pydfc/dFC/task_dFC/FCS_estimate.py" \
--dataset_info $DATASET_INFO \
--methods_config $METHODS_CONFIG

conda deactivate
