#!/bin/bash
#SBATCH --job-name=score_thresh_sweep
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=logs/slurm_%j_score_thresh_sweep.out
#SBATCH --error=logs/slurm_%j_score_thresh_sweep.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== score_thresh_sweep on no_fusion_camfix/last.pth, at $(date) ======"
python -u score_thresh_sweep.py results/checkpoints/no_fusion_camfix/last.pth no_fusion
echo "====== Done at $(date) ======"