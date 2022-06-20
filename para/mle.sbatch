#!/bin/bash

#SBATCH --job-name=minib22
#SBATCH --output=mini.out
#SBATCH --error=mini.err
#SBATCH --time=12:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=28
#SBATCH --mem-per-cpu=2000
#SBATCH --constraint=fdr
#SBATCH --array=1-100

# Load Python modules
module load python/anaconda-2020.02

python para_$SLURM_ARRAY_TASK_ID.py 