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
cd __REPO__ 2>/dev/null || true

BINDS=""
[ -d /share_zeta ] && BINDS="$BINDS -B /share_zeta:/share_zeta"

singularity exec --nv $BINDS --pwd __REPO__ __CONTAINER__ python3 __REPO__/run.py __ARGS__
