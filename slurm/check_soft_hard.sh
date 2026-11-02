#!/bin/bash
#SBATCH --job-name=soft_hard
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_soft_hard.out
#SBATCH --error=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_soft_hard.err

PROJECT=/mnt/scratch2/users/2641309/projects/coop_v2v
cd $PROJECT
mkdir -p logs
flight env activate conda@opencood
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v
export PYTHONPATH=$PROJECT:$PYTHONPATH
python -u check_soft_hard.py 2>&1 | tee logs/soft_hard_${SLURM_JOB_ID}.log
