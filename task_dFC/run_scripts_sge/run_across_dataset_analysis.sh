#!/bin/sh
#
#$ -N across_dataset_analysis
#$ -o logs/across_dataset_analysis_out.txt
#$ -e logs/across_dataset_analysis_err.txt
#$ -l h_rt=05:00:00
#$ -l h_vmem=32g
#$ -q YOUR_QUEUE

# ---- Cluster configuration (set these for your system) ----
VENV_PATH="/path/to/your/venv/bin/activate"
PYDFC_CODE_DIR="/path/to/pydfc"
# -----------------------------------------------------------

set -euo pipefail

mkdir -p logs
source "$VENV_PATH"

MULTI_DATASET_INFO="$PYDFC_CODE_DIR/task_dFC/run_scripts_sge/multi_dataset_info.json"

SCRIPT_NAME=${1:-}
SIMUL_OR_REAL=${2:-real}
SCRIPT_DIR="$PYDFC_CODE_DIR/task_dFC/multi_dataset_analysis"
SCRIPT_PATH="$SCRIPT_DIR/$SCRIPT_NAME"

if [ -z "$SCRIPT_NAME" ]; then
    echo "Usage: qsub run_across_dataset_analysis.sh <script_name> [real|simulated]"
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
