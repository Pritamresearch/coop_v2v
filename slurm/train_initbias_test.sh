#!/bin/bash
#SBATCH --job-name=initbias_test
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm_%j_initbias_test.out
#SBATCH --error=logs/slurm_%j_initbias_test.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== init_bias=0.1 validation (10 epochs) at $(date) ======"
python -u quick_initbias_test.py 2>&1 | tee logs/initbias_test_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
