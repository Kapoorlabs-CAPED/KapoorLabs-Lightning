#!/bin/bash
#SBATCH --job-name=care_predict
#SBATCH --output=care_predict_%j.out
#SBATCH --error=care_predict_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=grete:shared
#SBATCH --mem=32G

# Load modules
module purge
module load cuda
module load miniforge3

# Activate conda environment
source activate torchenv

# Run CARE denoising prediction
srun --unbuffered python predict-care.py
