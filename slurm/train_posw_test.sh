#!/bin/bash
#SBATCH --job-name=posw_test
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j_posw_test.out
#SBATCH --error=logs/slurm_%j_posw_test.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== pos_weight=20 validation (10 epochs) at $(date) ======"
python -u quick_posw_test.py 2>&1 | tee logs/posw_test_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
