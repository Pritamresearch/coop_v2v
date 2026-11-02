#!/bin/bash
#SBATCH --job-name=diag_fwd
#SBATCH --partition=k2-hipri
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/scratch2/users/2641309/tmp_claude_diag/slurm_%j.out
cd /mnt/scratch2/users/2641309/projects/coop_v2v
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v
D=/mnt/scratch2/users/2641309/tmp_claude_diag
python -u $D/cache_fwd.py $D/nf_last.pth no_fusion $D/nf_cache.pt 40 12
python -u $D/cache_fwd.py $D/of_last.pth ours_full $D/of_cache.pt 24 0
