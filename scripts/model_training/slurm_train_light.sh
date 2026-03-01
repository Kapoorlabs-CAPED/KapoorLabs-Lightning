#!/bin/bash
#SBATCH --job-name=oneat_light
#SBATCH --output=oneat_light_%j.out
#SBATCH --error=oneat_light_%j.err
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



# Run training with light transform preset
python lightning-oneat.py \
    parameters.transform_preset=light \
    train_data_paths.log_path='/projects/extern/nhr/nhr_ni/nhr_ni_test/nhr_ni_test_27040/dir.project/oneat_mitosis_model_light/' \
    train_data_paths.experiment_name='light_aug'
