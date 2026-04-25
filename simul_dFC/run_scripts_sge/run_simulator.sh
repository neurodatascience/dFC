#!/bin/sh
#
#$ -cwd
#$ -j y
#$ -o logs/simul_out.txt
#$ -e logs/simul_err.txt
#$ -q origami.q
#$ -l h_vmem=8G
#$ -t 1:200

DATASET_INFO="./dataset_info.json"

source /data/origami/dFC/anaconda3/etc/profile.d/conda.sh
conda activate pydfc

python "/data/origami/dFC/CODEs/pydfc/dFC/simul_dFC/task_data_simulator.py" \
--dataset_info $DATASET_INFO

conda deactivate
