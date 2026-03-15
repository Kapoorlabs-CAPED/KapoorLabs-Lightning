#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=oneat_default
#SBATCH --output=oneat_default_%j.out
#SBATCH --error=oneat_default_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=grete:shared
#SBATCH --mem=32G

# No time restriction (infinite/max allowed by cluster)

# Load modules (adjust as needed for your HPC)
module purge
module load cuda
module load miniforge3

# Activate conda environment
source activate torchenv


# Run training with default parameters
srun --unbuffered python lightning-oneat.py
