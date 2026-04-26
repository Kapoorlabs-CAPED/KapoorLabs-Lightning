#!/bin/bash
# Runs inside the SLURM allocation as a login shell (bash -l).
LUSTRE="/lustre/fsn1/projects/rech/jsy/uzj81mi/demo"
JOB_ID="$1"
CKPT="$2"
CONFIG="${3:-}"
module purge
module load anaconda-py3
conda activate torchenv
ARGS=(--job-id "$JOB_ID" --checkpoint "$CKPT")
if [ -n "$CONFIG" ]; then
  ARGS+=(--config "$CONFIG")
fi
python ${LUSTRE}/demo_predict.py "${ARGS[@]}"
