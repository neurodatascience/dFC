#!/bin/bash
#
#$ -cwd
#$ -o logs/fmriprep_out.log
#$ -e logs/fmriprep_err.log
#$ -l h_rt=24:00:00
#$ -l h_vmem=32G
#$ -q origami.q

# TODO replace with local paths
source "/data/origami/dFC/anaconda3/etc/profile.d/conda.sh"
conda activate nipoppy_env

SUBJECT_LIST="./subj_list.txt"
GLOBAL_CONFIG="../proc/global_configs.json"

echo "Number subjects found: `cat $SUBJECT_LIST | wc -l`"

SUBJECT_ID=`sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST`
echo "Subject ID: $SUBJECT_ID"

python "/data/origami/dFC/CODEs/nipoppy/nipoppy/workflow/proc_pipe/fmriprep/run_fmriprep.py" \
--global_config $GLOBAL_CONFIG \
--participant_id $SUBJECT_ID

conda deactivate
