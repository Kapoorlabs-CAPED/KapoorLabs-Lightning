#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=inception
#SBATCH --output=inception_%j.out
#SBATCH --error=inception_%j.err
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
srun --unbuffered python lightning-cellfate.py \
    parameters.transform_preset=heavy \
    train_data_paths.log_path=/projects/extern/nhr/nhr_ni/nhr_ni_test/nhr_ni_test_27040/dir.project/cellfate_model_heavy/ \
    train_data_paths.experiment_name=cellfate_nuclei_heavy
