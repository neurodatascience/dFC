#!/bin/bash
#
#SBATCH --job-name=fit_fcs_job
#SBATCH --output=logs/fcs_out_%A_%a.txt
#SBATCH --error=logs/fcs_err_%A_%a.txt
#SBATCH --time=7-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue

DATASET_INFO="./dataset_info.json"
METHODS_CONFIG="./methods_config.json"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module purge
module load StdEnv/2023
module load python/3.11.5
source "/home/mt00/venvs/pydfc_env/bin/activate"

# Verify CVMFS and Python environment are healthy on this node
python -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "CVMFS/Python broken on node $SLURMD_NODENAME, requeuing..."
    REQUEUE_ID=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
    REQUEUE_ID=${REQUEUE_ID:-$SLURM_JOB_ID}
    scontrol requeue $REQUEUE_ID
    exit 0
fi

python "/home/mt00/pydfc/dFC/task_dFC/FCS_estimate.py" \
    --dataset_info $DATASET_INFO \
    --methods_config $METHODS_CONFIG

deactivate
