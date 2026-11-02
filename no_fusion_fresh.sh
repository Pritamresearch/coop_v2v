#!/bin/bash
#SBATCH --job-name=no_fusion_fresh
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_no_fusion_fresh.out
#SBATCH --error=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_no_fusion_fresh.err

PROJECT=/mnt/scratch2/users/2641309/projects/coop_v2v
cd $PROJECT
mkdir -p logs
flight env activate conda@opencood
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v
export PYTHONPATH=$PROJECT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "====== fresh no_fusion (single-agent baseline), current codebase + backbone-LR fix, at $(date) ======"
python -u -c "
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders

base = Config()
base.validate()
tr, va, te = build_loaders(base)
cfg = make('no_fusion', base)
cfg = replace(cfg, exp_name='no_fusion_fresh', epochs=120)
print(f'Training: {cfg.exp_name}  grid={cfg.grid_H}x{cfg.grid_W}  bev=±{cfg.bev_x_max}m  img={cfg.lss_img_H}x{cfg.lss_img_W}  epochs={cfg.epochs}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/no_fusion_fresh_\${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
