#!/bin/bash
# Runs inside the SLURM allocation as a login shell (bash -l).
# Calls predict-cellfate.py with the right Hydra overrides for the
# Inception model the demo user picked.
#
# Usage:
#   run_inception_job.sh <job_id> <model_dir> <xml_path> <demo_name> <tracklet_length> <time_window_start> <time_window_end>

set -euo pipefail

LUSTRE="/lustre/fsn1/projects/rech/jsy/uzj81mi/demo_inception"
SCRIPT_ROOT="/lustre/fswork/projects/rech/jsy/uzj81mi/KapoorLabs-Lightning/scripts"

JOB_ID="$1"
MODEL_DIR="$2"
XML_PATH="$3"
DEMO_NAME="$4"
TRACKLET_LENGTH="${5:-25}"
TIME_WINDOW_START="${6:-0}"
TIME_WINDOW_END="${7:--1}"

RESULTS_DIR="${LUSTRE}/results/${JOB_ID}"
mkdir -p "${RESULTS_DIR}"

module purge
module load anaconda-py3
conda activate torchenv

cd "${SCRIPT_ROOT}/model_prediction"

# predict-cellfate.py is Hydra-driven; every model_dir on disk carries a
# training_config.json the script picks up via load_arch_from_training_config.
# We point its train_data_paths at the user's demo subdir and override
# the per-run knobs (xml path, tracklet length, time window).
srun --unbuffered python predict-cellfate.py \
    train_data_paths=cellfate_predict_jeanzay \
    train_data_paths.demo_name="${DEMO_NAME}" \
    train_data_paths.xml_path="${XML_PATH}" \
    train_data_paths.log_path="${MODEL_DIR}" \
    train_data_paths.output_dir="${RESULTS_DIR}" \
    parameters.input_mode=xml \
    parameters.tracklet_length="${TRACKLET_LENGTH}" \
    parameters.time_window="[${TIME_WINDOW_START},${TIME_WINDOW_END}]" \
    parameters.transition_time_determination=true

echo "DONE" > "${RESULTS_DIR}/status.txt"
