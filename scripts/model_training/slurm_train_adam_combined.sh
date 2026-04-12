#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=oneat_combined
#SBATCH --output=oneat_combined_%j.out
#SBATCH --error=oneat_combined_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=grete:shared
#SBATCH --mem=32G
#SBATCH --requeue
#SBATCH --signal=SIGTERM@180
echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo


# Load modules
module purge
module load cuda
module load miniforge3

# Activate conda environment
source activate torchenv

# Run training with combined dataset
# Uses --config-name to override default scenario with combined config
srun --unbuffered python lightning-oneat-adam.py \
    --config-name=scenario_train_oneat_combined \
    parameters.learning_rate=1.0e-3 \
    parameters.transform_preset=heavy
