#!/bin/bash
# Runs inside the SLURM allocation as a login shell (bash -l).
LUSTRE="/lustre/fsn1/projects/rech/jsy/uzj81mi/demo"
module purge
module load anaconda-py3
conda activate torchenv
python ${LUSTRE}/demo_predict.py --job-id "$1"
