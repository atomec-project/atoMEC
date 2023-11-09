#!/bin/bash -l
#SBATCH --partition=rome
#SBATCH --ntasks=10
#SBATCH --nodes=1
#SBATCH --mem=80G

env={ENV}
module purge
conda activate $env
export OMP_NUM_THREADS=1

python -u test.py > test.log
