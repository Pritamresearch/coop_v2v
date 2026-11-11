#!/bin/bash
#SBATCH --job-name=ours_full_lidardepth
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100,k2-gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j_ours_full_lidardepth.out
#SBATCH --error=logs/slurm_%j_ours_full_lidardepth.err
#
# ours_full (full cooperative model) trained with LiDAR depth supervision + CenterNet heatmap loss.
# Resumes automatically from results/checkpoints/ours_full_lidardepth/last.pth
# when it exists, so the same script can be chained past the 3-day limit:
#   j=$(sbatch --parsable slurm/ours_full_lidardepth.sh)
#   sbatch --dependency=afterany:$j slurm/ours_full_lidardepth.sh

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== ours_full_lidardepth at $(date) on $(hostname) ======"
# pre-training check: collaborator features must land on the right ego cells, with and without augmentation
python -u scripts/check_collab_warp.py --frames 8 || { echo 'collaborator warp check FAILED -- not training'; exit 1; }
python -u -c "
import os
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders
base = Config()
base.validate()
cfg = make('ours_full', base)
cfg = replace(cfg, exp_name='ours_full_lidardepth', epochs=120,
              depth_source='lidar',                      # LiDAR depth labels (train split only)
              hm_norm='centernet', hm_pos_weight=1.0,    # standard CenterNet heatmap loss
              gauss_mode='centerpoint',                  # CenterPoint Gaussian, same on train and val
              eval_score_thresh=0.05)
cfg = replace(cfg, resume=os.path.exists(f'{cfg.root}/checkpoints/{cfg.exp_name}/last.pth'))
cfg.validate()
tr, va, te = build_loaders(cfg)
# refuse to train on a partially restored LiDAR set (frames without ego .pcd are dropped)
assert len(tr.dataset) >= 6000, f'only {len(tr.dataset)} train frames have LiDAR -- restore incomplete?'
assert len(te.dataset) > 0 and len(va.dataset) > 0
print(f'Training: {cfg.exp_name}  resume={cfg.resume}  train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}  '
      f'depth={cfg.depth_source} hm={cfg.hm_norm}/pw{cfg.hm_pos_weight} gauss={cfg.gauss_mode} thresh={cfg.eval_score_thresh}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/ours_full_lidardepth_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
