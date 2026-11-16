#!/bin/bash
#SBATCH --job-name=imgaug_readout
#SBATCH --partition=k2-hipri
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:55:00
#SBATCH --output=logs/slurm_%j_imgaug_readout.out
#SBATCH --error=logs/slurm_%j_imgaug_readout.err
#
# Pre-registered epoch-20 image-augmentation readout (CPU).
# Submitted with sbatch, not srun, so it survives the submitting session ending.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
OUT=${COOP_DIAG:-/mnt/scratch2/users/2641309/tmp_claude_diag}
echo "====== imgaug readout at $(date) on $(hostname) ======"
python -u scripts/imgaug_readout.py --frames 200 --bootstrap 2000 --budget-min 145 \
       --out "$OUT/imgaug_readout_ep20.json"
echo "====== Done at $(date) ======"
