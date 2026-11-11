#!/bin/bash
#SBATCH --job-name=soft_hard
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j_soft_hard.out
#SBATCH --error=logs/slurm_%j_soft_hard.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
python -u check_soft_hard.py 2>&1 | tee logs/soft_hard_${SLURM_JOB_ID}.log
