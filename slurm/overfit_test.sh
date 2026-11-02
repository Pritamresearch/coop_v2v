#!/bin/bash
#SBATCH --job-name=overfit_test
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/overfit_%j.out
#SBATCH --error=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/overfit_%j.err

PROJECT=/mnt/scratch2/users/2641309/projects/coop_v2v
cd $PROJECT
mkdir -p logs

flight env activate conda@opencood
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$PROJECT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "====== Overfit-8 diagnostic at $(date) ======"
python -u run_overfit_test.py 2>&1 | tee logs/overfit_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
