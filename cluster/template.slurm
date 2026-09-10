#!/bin/bash
#SBATCH --job-name=__JOBNAME__
#SBATCH --partition=__PARTITION__
#SBATCH --account=__ACCOUNT__
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=__CPUS__
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --output=__LOGDIR__/%x.%A_%a.out
#SBATCH --error=__LOGDIR__/%x.%A_%a.err

source ~/.bashrc
conda activate __CONDA_ENV__
cd __REPO__

python run.py __ARGS__