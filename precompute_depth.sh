#!/bin/bash
#SBATCH --job-name=depth_prior
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/depth_prior_%j.out
#SBATCH --error=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/depth_prior_%j.err

PROJECT=/mnt/scratch2/users/2641309/projects/coop_v2v
cd $PROJECT
mkdir -p logs

flight env activate conda@opencood
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v

export PYTHONPATH=$PROJECT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "====== Depth prior precomputation at $(date) ======"
python -u precompute_depth_prior.py \
    --splits train val test \
    --d_min 4.0 --d_max 50.0 \
    --depth_K 64 \
    --downsample 8
echo "====== Done at $(date) ======"

