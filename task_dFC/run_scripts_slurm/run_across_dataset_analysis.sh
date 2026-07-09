#!/bin/sh
#
#SBATCH --job-name=across_dataset_analysis
#SBATCH --output=logs/%x_out.txt
#SBATCH --error=logs/%x_err.txt
#SBATCH --time=05:00:00
#SBATCH --mem=32G
# Note: run sbatch from your multi_dataset_analysis/codes directory, or uncomment and set --chdir:
# #SBATCH --chdir=/path/to/multi_dataset_analysis/codes

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# -----------------------------------------------------------

set -euo pipefail
trap 'echo "ERROR: Script failed at line $LINENO with exit code $?" >&2' ERR

mkdir -p logs
source "$VENV_PATH"

MULTI_DATASET_INFO="$PYDFC_CODE_DIR/task_dFC/run_scripts_slurm/multi_dataset_info.json"

SCRIPT_NAME=${1:-}
SIMUL_OR_REAL=${2:-real}
SCRIPT_DIR="$PYDFC_CODE_DIR/task_dFC/multi_dataset_analysis"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

if [ -z "$SCRIPT_NAME" ]; then
    echo "Usage: sbatch run_analysis.sh <script_name> [real|simulated]"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' not found."
    exit 1
fi

case "$SCRIPT_NAME" in
  performance_predict.py | performance_factor.py | ml_results.py | dfc_visualization.py | embedding_visualization.py | sample_matrix_visualization.py | task_presence_binarization.py | task_timing_stats.py | cohensd.py)
    python "$SCRIPT_PATH" --multi_dataset_info "$MULTI_DATASET_INFO" --simul_or_real "$SIMUL_OR_REAL"
    ;;
  *)
    echo "Unknown script: $SCRIPT_NAME"
    exit 1
    ;;
esac

deactivate
