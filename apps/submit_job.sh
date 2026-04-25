#!/bin/bash
# SSH to Jean Zay and submit prediction job.
# Usage: ./submit_job.sh <job_id>

set -euo pipefail

JOB_ID="${1:?Error: job ID required}"
REMOTE="uzj81mi@jean-zay.idris.fr"
LUSTRE="/lustre/fsn1/projects/rech/jsy/uzj81mi/demo"

ssh "${REMOTE}" "mkdir -p ${LUSTRE}/results/${JOB_ID} ${LUSTRE}/logs && sbatch \
  --account=jsy@a100 \
  --partition=visu \
  --cpus-per-task=10 \
  --time=04:00:00 \
  --job-name=oneat_demo \
  --output=${LUSTRE}/logs/%x_%j.out \
  --error=${LUSTRE}/logs/%x_%j.err \
  --wrap='bash -l ${LUSTRE}/run_job.sh ${JOB_ID}'"
