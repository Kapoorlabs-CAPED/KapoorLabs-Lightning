#!/bin/bash
# SSH to Jean Zay and submit prediction job (heavy / A100 GPU variant).
# Usage: ./submit_heavy_job.sh <job_id> <ckpt> [config] [event_threshold] [nms_iou] [batch_size_predict]

set -euo pipefail

JOB_ID="${1:?Error: job ID required}"
CKPT="${2:?Error: checkpoint path required}"
CONFIG="${3:-}"
EVENT_THRESHOLD="${4:-}"
NMS_IOU="${5:-}"
BATCH_SIZE_PREDICT="${6:-}"
REMOTE="uzj81mi@jean-zay.idris.fr"
LUSTRE="/lustre/fsn1/projects/rech/jsy/uzj81mi/demo"

ssh "${REMOTE}" "mkdir -p ${LUSTRE}/results/${JOB_ID} ${LUSTRE}/logs && sbatch \
  --account=jsy@a100 \
  --partition=visu \
  --cpus-per-task=40 \
  --hint=nomultithread \
  --time=4:00:00 \
  --job-name=oneat_demo \
  --output=${LUSTRE}/logs/%x_%j.out \
  --error=${LUSTRE}/logs/%x_%j.err \
  --wrap='bash -l ${LUSTRE}/run_job.sh ${JOB_ID} ${CKPT} \"${CONFIG}\" \"${EVENT_THRESHOLD}\" \"${NMS_IOU}\" \"${BATCH_SIZE_PREDICT}\"'"
