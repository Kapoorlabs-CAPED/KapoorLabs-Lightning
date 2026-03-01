#!/usr/bin/env python3
"""
Learning Rate Hyperparameter Sweep for ONEAT with Adam Optimizer and Heavy Augmentations
Submits independent jobs (no dependencies) for different learning rates.
"""

import subprocess
from pathlib import Path

# Base directory for all experiments
BASE_DIR = "/projects/extern/nhr/nhr_ni/nhr_ni_test/nhr_ni_test_27040/dir.project"

# Define learning rate configurations
job_matrix = [
    {"learning_rate": 1.0e-2, "folder": "oneat_adam_heavy_lr_1e-2", "exp_name": "adam_heavy_lr_1e-2"},
    {"learning_rate": 5.0e-3, "folder": "oneat_adam_heavy_lr_5e-3", "exp_name": "adam_heavy_lr_5e-3"},
    {"learning_rate": 1.0e-3, "folder": "oneat_adam_heavy_lr_1e-3", "exp_name": "adam_heavy_lr_1e-3"},
    {"learning_rate": 5.0e-4, "folder": "oneat_adam_heavy_lr_5e-4", "exp_name": "adam_heavy_lr_5e-4"},
    {"learning_rate": 1.0e-4, "folder": "oneat_adam_heavy_lr_1e-4", "exp_name": "adam_heavy_lr_1e-4"},
]

# SLURM script template for GWDG
slurm_script_template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={output_file}
#SBATCH --error={error_file}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100:1
#SBATCH --partition=grete:shared
#SBATCH --mem=32G

# No time restriction

# Load modules
module purge
module load cuda
module load miniforge3

# Activate conda environment
source activate torchenv

# Run training with Adam optimizer, heavy augmentations, and specific learning rate
srun --unbuffered python lightning-oneat-adam.py \\
    parameters.learning_rate={learning_rate} \\
    parameters.transform_preset=heavy \\
    train_data_paths.log_path='{log_path}' \\
    train_data_paths.experiment_name='{exp_name}'
"""


def submit_job(config, job_idx):
    """Submit a single SLURM job with the given configuration."""
    # Create job name and file names
    lr_str = f"{config['learning_rate']:.0e}".replace(".", "_").replace("-", "m")
    job_name = f"oneat_adam_heavy_lr_{lr_str}"
    error_file = f"oneat_adam_heavy_lr_{lr_str}_%j.err"
    output_file = f"oneat_adam_heavy_lr_{lr_str}_%j.out"
    log_path = f"{BASE_DIR}/{config['folder']}/"

    # Format the SLURM script with config values
    slurm_script = slurm_script_template.format(
        job_name=job_name,
        error_file=error_file,
        output_file=output_file,
        learning_rate=config["learning_rate"],
        log_path=log_path,
        exp_name=config["exp_name"],
    )

    # Write temporary script file
    script_path = Path(f"temp_job_lr_sweep_{job_idx}.sh")
    with open(script_path, "w") as f:
        f.write(slurm_script)

    # Submit job (no dependencies - all run in parallel)
    cmd = ["sbatch", str(script_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        return job_id
    else:
        print(f"Error submitting job: {result.stderr}")
        return None


def main():
    """Submit all jobs in the matrix (parallel, no dependencies)."""
    print(f"ONEAT Adam Heavy Augmentation - Learning Rate Sweep")
    print(f"=" * 60)
    print(f"Total jobs to submit: {len(job_matrix)}")
    print(f"Jobs will run in PARALLEL (no dependencies)")
    print(f"=" * 60)

    job_ids = []

    for idx, config in enumerate(job_matrix):
        job_id = submit_job(config, idx)

        if job_id:
            job_ids.append(job_id)
            print(f"Submitted job {idx+1}/{len(job_matrix)}: {job_id} (lr={config['learning_rate']:.0e})")

    print(f"\n{'=' * 60}")
    print(f"Submitted {len(job_ids)} jobs")
    print(f"Job IDs: {', '.join(job_ids)}")
    print(f"\nTo check status: squeue -u $USER")
    print(f"To cancel all: scancel {' '.join(job_ids)}")

    # Clean up temp scripts
    print(f"\nTemp scripts created: temp_job_lr_sweep_*.sh")
    print(f"You can delete them after jobs start: rm temp_job_lr_sweep_*.sh")


if __name__ == "__main__":
    main()
