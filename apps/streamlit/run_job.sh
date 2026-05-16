#!/bin/bash
# Runs inside the SLURM allocation as a login shell (bash -l).
# Usage: run_job.sh <job_id> <ckpt> [config] [event_threshold] [nms_iou] [batch_size_predict]
LUSTRE="/lustre/fsn1/projects/rech/jsy/uzj81mi/demo"
JOB_ID="$1"
CKPT="$2"
CONFIG="${3:-}"
EVENT_THRESHOLD="${4:-}"
NMS_IOU="${5:-}"
BATCH_SIZE_PREDICT="${6:-}"
module purge
module load anaconda-py3
conda activate torchenv
ARGS=(--job-id "$JOB_ID" --checkpoint "$CKPT")
if [ -n "$CONFIG" ]; then
  ARGS+=(--config "$CONFIG")
fi
if [ -n "$EVENT_THRESHOLD" ]; then
  ARGS+=(--event-threshold "$EVENT_THRESHOLD")
fi
if [ -n "$NMS_IOU" ]; then
  ARGS+=(--nms-iou-threshold "$NMS_IOU")
fi
if [ -n "$BATCH_SIZE_PREDICT" ]; then
  ARGS+=(--batch-size-predict "$BATCH_SIZE_PREDICT")
fi
python ${LUSTRE}/demo_predict.py "${ARGS[@]}"
