#!/bin/bash
#SBATCH --job-name=ours_full_camfix
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j_ours_full_camfix.out
#SBATCH --error=logs/slurm_%j_ours_full_camfix.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== ours_full_camfix: FRESH start (not resumed) -- grid fix + NMS fix + pos_weight fix + CAMERA-CONVENTION fix + depth_K fix, at $(date) ======"
python -u -c "
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders

base = Config()
base.validate()
assert base.bev_x_max == 51.2 and base.cell_m == 0.4 and base.grid_H == 256, \
    f'grid fix not active -- got bev_x_max={base.bev_x_max} cell_m={base.cell_m} grid_H={base.grid_H}'
assert base.depth_K == 128, f'depth_K fix not active -- got depth_K={base.depth_K}'
tr, va, te = build_loaders(base)
cfg = make('ours_full', base)
cfg = replace(cfg, exp_name='ours_full_camfix', epochs=120)
print(f'Training: {cfg.exp_name}  grid={cfg.grid_H}x{cfg.grid_W}  bev=+/-{cfg.bev_x_max}m  depth_K={cfg.depth_K}  epochs={cfg.epochs}  resume={cfg.resume}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/ours_full_camfix_\${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"