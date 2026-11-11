#!/bin/bash
#SBATCH --job-name=no_fusion_gridfix
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j_no_fusion_gridfix.out
#SBATCH --error=logs/slurm_%j_no_fusion_gridfix.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== no_fusion_gridfix: baseline + grid fix (+/-51.2m,0.4m,256x256) + NMS fix + pos_weight=4.0 fix, at $(date) ======"
python -u -c "
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders

base = Config()
base.validate()
assert base.bev_x_max == 51.2 and base.cell_m == 0.4 and base.grid_H == 256, \
    f'grid fix not active -- got bev_x_max={base.bev_x_max} cell_m={base.cell_m} grid_H={base.grid_H}'
tr, va, te = build_loaders(base)
cfg = make('no_fusion', base)
cfg = replace(cfg, exp_name='no_fusion_gridfix', epochs=120)
print(f'Training: {cfg.exp_name}  grid={cfg.grid_H}x{cfg.grid_W}  bev=+/-{cfg.bev_x_max}m  cell_m={cfg.cell_m}  img={cfg.lss_img_H}x{cfg.lss_img_W}  epochs={cfg.epochs}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/no_fusion_gridfix_\${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"