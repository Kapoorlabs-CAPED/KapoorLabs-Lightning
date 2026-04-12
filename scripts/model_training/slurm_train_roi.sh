#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --job-name=roi_train
#SBATCH --output=roi_train_%j.out
#SBATCH --error=roi_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=grete:shared
#SBATCH --mem=64G

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

# Run CARE denoising training
srun --unbuffered python lightning-roi.py
