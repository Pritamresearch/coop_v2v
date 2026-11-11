#!/bin/bash
#SBATCH --job-name=lidar_oracle
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j_lidar_oracle.out
#SBATCH --error=logs/slurm_%j_lidar_oracle.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== LiDAR-oracle training at $(date) ======"
python -u run_lidar_oracle.py --epochs 120 --warmup 40
echo "====== Done at $(date) ======"