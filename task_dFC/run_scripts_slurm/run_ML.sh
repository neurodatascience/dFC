#!/bin/bash
#
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ML_out_%A_%a.txt   # %A = array job ID, %a = task ID
#SBATCH --error=logs/ML_err_%A_%a.txt
#SBATCH --mem=128G
#SBATCH --requeue

DATASET_INFO="./dataset_info.json"

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

python "/home/mt00/pydfc/dFC/task_dFC/ML.py" \
    --dataset_info $DATASET_INFO

deactivate
