#!/bin/bash
#SBATCH --nodes=1             # Number of nodes 
#SBATCH -A lzc@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1       # Allocate 4 GPUs per node
#SBATCH --partition=gpu_p5
#SBATCH --job-name=care_train               # Jobname 
#SBATCH --cpus-per-task=40
#SBATCH --output=cell.o%j            # Output file 
#SBATCH --error=cell.o%j            # Error file 
#SBATCH --time=20:00:00       # Expected runtime HH:MM:SS (max 100h)
module purge # purging modules inherited by default

module load anaconda-py3/2020.11
#conda init bash # deactivating environments inherited by default
conda deactivate
conda activate torchenv

module load cuda/11.8.0
srun --unbuffered python lightning-care.py
