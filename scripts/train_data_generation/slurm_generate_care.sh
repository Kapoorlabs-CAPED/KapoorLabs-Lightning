#!/bin/bash
#SBATCH --job-name=gen_care_h5
#SBATCH --output=gen_care_h5_%j.out
#SBATCH --error=gen_care_h5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
##SBATCH --partition=grete:shared

# Load modules
module purge
module load miniforge3

# Activate conda environment
source activate torchenv

# Generate CARE H5 dataset from paired low/high SNR images
srun --unbuffered python generate-care-training-data.py

echo "CARE H5 generation complete"
