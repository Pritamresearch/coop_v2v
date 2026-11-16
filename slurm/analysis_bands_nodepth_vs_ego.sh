#!/bin/bash
#SBATCH --job-name=bands_nodepth_ego
#SBATCH --partition=k2-hipri
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:55:00
#SBATCH --output=logs/slurm_%j_bands_nodepth_ego.out
#SBATCH --error=logs/slurm_%j_bands_nodepth_ego.err
#
# Per-distance-band comparison of the matched 60-epoch pair, same val frames:
#   ours_nodepth60 (cooperative) vs ego_armb60 (ego-only)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
OUT=${COOP_DIAG:-/mnt/scratch2/users/2641309/tmp_claude_diag}
R=${COOP_RESULTS:-$(pwd)/results}
echo "====== bands nodepth-vs-ego at $(date) on $(hostname) ======"
python -u scripts/diag_bands.py compare \
    ours_nodepth60 "$R/checkpoints/ours_nodepth60/last.pth" \
    ego_armb60     "$R/checkpoints/ego_armb60/last.pth" \
    "$OUT/bands_nodepth60_vs_ego60.json" 200 145
echo "====== Done at $(date) ======"
