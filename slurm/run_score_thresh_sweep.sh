#!/bin/bash
#SBATCH --job-name=score_thresh_sweep
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_score_thresh_sweep.out
#SBATCH --error=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_score_thresh_sweep.err

PROJECT=/mnt/scratch2/users/2641309/projects/coop_v2v
cd $PROJECT
mkdir -p logs
flight env activate conda@opencood
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v
export PYTHONPATH=$PROJECT:$PYTHONPATH

echo "====== score_thresh_sweep on no_fusion_camfix/last.pth, at $(date) ======"
python -u score_thresh_sweep.py results/checkpoints/no_fusion_camfix/last.pth no_fusion
echo "====== Done at $(date) ======"