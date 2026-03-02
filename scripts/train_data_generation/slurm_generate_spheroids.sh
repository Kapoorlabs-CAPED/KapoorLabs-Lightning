#!/bin/bash
#SBATCH --job-name=gen_spheroids
#SBATCH --output=gen_spheroids_%j.out
#SBATCH --error=gen_spheroids_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=grete:shared

# Load modules
module purge
module load miniforge3

# Activate conda environment
source activate torchenv

# Generate spheroids H5 dataset
# Uses scenario_generate_oneat_spheroids.yaml which points to gwdg_spheroids.yaml
srun --unbuffered python generate-oneat-training-data.py \
    --config-name=scenario_generate_oneat_spheroids

echo "Spheroids H5 generation complete"
