#!/bin/bash
# SSH to Jean Zay and submit an inception cell-fate prediction job.
# Usage:
#   ./submit_inception_job.sh <job_id> <model_dir> <xml_path> <demo_name> <tracklet_length> <time_window_start> <time_window_end>
#
# All paths are LUSTRE-side (compute-node visible). The streamlit app
# resolves the local sshfs mount → lustre path before invoking us.

set -euo pipefail

JOB_ID="${1:?Error: job ID required}"
MODEL_DIR="${2:?Error: model dir required}"
XML_PATH="${3:?Error: xml path required}"
DEMO_NAME="${4:?Error: demo name required}"
TRACKLET_LENGTH="${5:-25}"
TIME_WINDOW_START="${6:-0}"
TIME_WINDOW_END="${7:--1}"

REMOTE="uzj81mi@jean-zay.idris.fr"
LUSTRE="/lustre/fsn1/projects/rech/jsy/uzj81mi/demo_inception"

ssh "${REMOTE}" "mkdir -p ${LUSTRE}/results/${JOB_ID} ${LUSTRE}/logs && sbatch \
  --account=jsy@a100 \
  --partition=visu \
  --cpus-per-task=10 \
  --time=04:00:00 \
  --job-name=inception_demo \
  --output=${LUSTRE}/logs/%x_%j.out \
  --error=${LUSTRE}/logs/%x_%j.err \
  --wrap='bash -l ${LUSTRE}/run_inception_job.sh ${JOB_ID} ${MODEL_DIR} ${XML_PATH} ${DEMO_NAME} ${TRACKLET_LENGTH} ${TIME_WINDOW_START} ${TIME_WINDOW_END}'"
