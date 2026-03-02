#!/bin/bash
#SBATCH --job-name=merge_h5
#SBATCH --output=merge_h5_%j.out
#SBATCH --error=merge_h5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=grete:shared

# Load modules
module purge
module load miniforge3

# Activate conda environment
source activate torchenv

BASE_DIR="/projects/extern/nhr/nhr_ni/nhr_ni_test/nhr_ni_test_27040/dir.project/oneat_training"

# Merge the two H5 datasets
srun --unbuffered python merge_h5_datasets.py \
    "${BASE_DIR}/oneat_kapoorlabs.h5" \
    "${BASE_DIR}/spheroids/oneat_spheroids.h5" \
    --output "${BASE_DIR}/oneat_combined.h5"

echo "H5 merge complete"
