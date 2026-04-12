#!/bin/bash
#SBATCH --nodes=1
#SBATCH -A lzc@a100
#SBATCH -C a100
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_p5
#SBATCH --job-name=cellfate_train
#SBATCH --cpus-per-task=40
#SBATCH --output=cellfate.o%j
#SBATCH --error=cellfate.o%j
#SBATCH --time=20:00:00

#SBATCH --requeue
#SBATCH --signal=SIGTERM@180
echo "running in shell: " "$SHELL"
export NCCL_SOCKET_IFNAME=lo


module purge
module load anaconda-py3
conda deactivate
conda activate torchenv
module load cuda/11.8.0

# Override train_data_paths config group to use Jean Zay paths
# without modifying scenario_train_cellfate.yaml
srun --unbuffered python lightning-cellfate.py 
