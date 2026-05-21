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

# predict-cellfate.py is Hydra-driven; the model_dir on disk carries a
# training_config.json the script picks up via load_arch_from_training_config.
# The cellfate scenario uses ``experiment_data_paths`` (not the
# train_data_paths key the other scripts use) — fields are
# ``xml_file`` / ``checkpoint_path`` / ``output_dir`` (see
# scripts/conf/experiment_data_paths/cellfate_predict_jeanzay.yaml).
# GT annotation paths in the yaml point at hardcoded fifth_dataset
# locations. Override them per-demo so each curated demo's own
# ground_truth/ folder is used. predict-cellfate.py only writes a
# confusion matrix when these paths exist on disk, so without the
# override the metrics section comes back empty.
GT_BASAL="${LUSTRE}/uploads/${DEMO_NAME}/ground_truth/basal_cells_nuclei_annotations.csv"
GT_GOBLET="${LUSTRE}/uploads/${DEMO_NAME}/ground_truth/goblet_cells_nuclei_annotations.csv"
GT_RADIAL="${LUSTRE}/uploads/${DEMO_NAME}/ground_truth/radially_intercalating_cells_nuclei_annotations.csv"

srun --unbuffered python predict-cellfate.py \
    experiment_data_paths=cellfate_predict_jeanzay \
    experiment_data_paths.xml_file="${XML_PATH}" \
    experiment_data_paths.checkpoint_path="${MODEL_DIR}" \
    experiment_data_paths.output_dir="${RESULTS_DIR}" \
    experiment_data_paths.basal_gt_annotations="${GT_BASAL}" \
    experiment_data_paths.goblet_gt_annotations="${GT_GOBLET}" \
    experiment_data_paths.radially_intercalating_gt_annotations="${GT_RADIAL}" \
    parameters.input_mode=xml \
    parameters.accelerator=cpu \
    parameters.tracklet_length="${TRACKLET_LENGTH}" \
    parameters.time_window="[${TIME_WINDOW_START},${TIME_WINDOW_END}]" \
    parameters.transition_time_determination=true

echo "DONE" > "${RESULTS_DIR}/status.txt"
