#!/bin/bash
#SBATCH --job-name=oneat_heavy
#SBATCH --output=oneat_heavy_%j.out
#SBATCH --error=oneat_heavy_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=grete:shared
#SBATCH --mem=32G

# No time restriction (infinite/max allowed by cluster)

# Load modules (adjust as needed for your HPC)
module purge
module load cuda
module load anaconda3

# Activate conda environment
source activate capedenv

# Change to script directory
cd /lustre/fswork/projects/rech/jsy/uzj81mi/KapoorLabs-Lightning/scripts/model_training

# Run training with heavy transform preset
python lightning-oneat.py \
    parameters.transform_preset=heavy \
    train_data_paths.log_path='/lustre/fsn1/projects/rech/jsy/uzj81mi/oneat_mitosis_model_heavy/' \
    train_data_paths.experiment_name='heavy_aug'
