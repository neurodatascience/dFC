#!/bin/sh
#
#$ -N fit_fcs_job
#$ -o logs/fcs_out.txt
#$ -e logs/fcs_err.txt
#$ -l h_rt=168:00:00
#$ -pe smp 8
#$ -l h_vmem=8g
#$ -q YOUR_QUEUE

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# For conda environments, replace the two lines above with:
#   CONDA_SH="/path/to/conda/etc/profile.d/conda.sh"
#   CONDA_ENV="pydfc"
# -----------------------------------------------------------

DATASET_INFO="./dataset_info.json"
METHODS_CONFIG="./methods_config.json"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Activate virtual environment
source "$VENV_PATH"
# For conda: source "$CONDA_SH" && conda activate "$CONDA_ENV"

python "$PYDFC_CODE_DIR/task_dFC/FCS_estimate.py" \
--dataset_info $DATASET_INFO \
--methods_config $METHODS_CONFIG

deactivate
