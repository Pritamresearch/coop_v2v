#!/bin/bash
#SBATCH --job-name=coop_v2v_train
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j_train.out
#SBATCH --error=logs/slurm_%j_train.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== Training all baselines at $(date) ======"
#python run_baselines.py 2>&1 | tee logs/baselines_${SLURM_JOB_ID}.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u run_baselines.py --only ours_full --epochs 120 2>&1 | tee logs/baselines_${SLURM_JOB_ID}.log


echo "====== Ablations at $(date) ======"
python run_ablations.py --only A1,A2 2>&1 | tee logs/ablations_${SLURM_JOB_ID}.log

echo "====== Done at $(date) ======"
