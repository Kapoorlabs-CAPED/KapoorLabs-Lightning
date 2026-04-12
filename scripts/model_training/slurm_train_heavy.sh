#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=oneat_heavy
#SBATCH --output=oneat_heavy_%j.out
#SBATCH --error=oneat_heavy_%j.err
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


# No time restriction (infinite/max allowed by cluster)

# Load modules (adjust as needed for your HPC)
module purge
module load cuda
module load miniforge3

# Activate conda environment
source activate torchenv

# Run training with heavy transform preset
srun --unbuffered python lightning-oneat.py \
    parameters.transform_preset=heavy \
    train_data_paths.log_path='/projects/extern/nhr/nhr_ni/nhr_ni_test/nhr_ni_test_27040/dir.project/oneat_mitosis_model_heavy/' \
    train_data_paths.experiment_name='heavy_aug'
