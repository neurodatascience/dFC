#!/bin/bash
#
#SBATCH --job-name=assess_dfc_job
#SBATCH --output=logs/dfc_out_%A_%a.txt
#SBATCH --error=logs/dfc_err_%A_%a.txt
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --requeue

SUBJECT_LIST="./subj_list.txt"
DATASET_INFO="./dataset_info.json"
METHODS_CONFIG="./methods_config.json"

echo "Number subjects found: $(cat $SUBJECT_LIST | wc -l)"

SUBJECT_ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo "Subject ID: $SUBJECT_ID"

module purge
module load StdEnv/2023
module load python/3.11.5
source "/home/mt00/venvs/pydfc_env/bin/activate"

# Verify CVMFS and Python environment are healthy on this node
python -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "CVMFS/Python broken on node $SLURMD_NODENAME, requeuing task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}..."
    scontrol requeue ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
    exit 0
fi

python "/home/mt00/pydfc/dFC/task_dFC/dFC_assessment.py" \
    --dataset_info $DATASET_INFO \
    --methods_config $METHODS_CONFIG \
    --participant_id $SUBJECT_ID

deactivate
