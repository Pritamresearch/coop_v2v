#!/bin/bash
#SBATCH --job-name=no_fusion_camfix
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_no_fusion_camfix.out
#SBATCH --error=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_no_fusion_camfix.err

PROJECT=/mnt/scratch2/users/2641309/projects/coop_v2v
cd $PROJECT
mkdir -p logs
flight env activate conda@opencood
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v
export PYTHONPATH=$PROJECT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "====== no_fusion_camfix: FRESH start (not resumed) -- grid fix + NMS fix + pos_weight fix + CAMERA-CONVENTION fix + depth_K fix, at $(date) ======"
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
cfg = make('no_fusion', base)
cfg = replace(cfg, exp_name='no_fusion_camfix', epochs=120)
print(f'Training: {cfg.exp_name}  grid={cfg.grid_H}x{cfg.grid_W}  bev=+/-{cfg.bev_x_max}m  depth_K={cfg.depth_K}  epochs={cfg.epochs}  resume={cfg.resume}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/no_fusion_camfix_\${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"