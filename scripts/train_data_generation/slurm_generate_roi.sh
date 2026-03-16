#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=gen_roi_h5
#SBATCH --output=gen_roi_h5_%j.out
#SBATCH --error=gen_roi_h5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
##SBATCH --partition=standard96

# Load modules
module purge
module load miniforge3

# Activate conda environment
source activate torchenv

# Generate Roi H5 dataset from paired low/high SNR images
srun --unbuffered python generate-roi-training-data.py

echo "Roi generation complete"
