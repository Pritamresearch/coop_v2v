#!/bin/bash
#SBATCH --job-name=overfit_test
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/overfit_%j.out
#SBATCH --error=logs/overfit_%j.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$PROJECT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "====== Overfit-8 diagnostic at $(date) ======"
python -u run_overfit_test.py 2>&1 | tee logs/overfit_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
