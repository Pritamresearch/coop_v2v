#!/bin/bash
#SBATCH --job-name=ours_nodepth60
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100,k2-gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j_ours_nodepth60.out
#SBATCH --error=logs/slurm_%j_ours_nodepth60.err
#
# 'NO DEPTH TERM' ABLATION: ours_nodepth (sender utility = ||F||, no exp(-alpha*D)) with the
# collaborator warp fix (a624570) + depth A/B arm B. Deliberately labelled as an ablation: the
# depth-aware utility was measured degenerate (rho 0.977 vs magnitude_only) and per-pixel depth
# entropy is ANTI-correlated with depth error beyond 20 m, so this is the honest baseline that a
# future learned-reliability depth term must beat.
# (depth_loss_weight='inv_sqrt_freq': ~2x far-range depth mass near truth at epoch 20).
# No image augmentation (that A/B is separate). 120 epochs, chained past the 3-day limit.
# Resumes automatically from results/checkpoints/ours_nodepth60/last.pth
# when it exists, so the same script can be chained past the 3-day limit:
#   j=$(sbatch --parsable slurm/ours_nodepth60.sh)
#   sbatch --dependency=afterany:$j slurm/ours_nodepth60.sh

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== ours_nodepth60 at $(date) on $(hostname) ======"
python -u scripts/check_need_uncovered.py --frames 4 || { echo 'GATE FAILED: need/coverage'; exit 1; }
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
cfg = make('ours_nodepth', base)
cfg = replace(cfg, exp_name='ours_nodepth60',
              epochs=60, warmup_epochs=20, lr_warmup_epochs=3, save_every=10,
              # 60-epoch budget: the 120-epoch ego-only run plateaued by ~ep58 (AP50 best
              # 0.0365 @58; epochs 60-120 mean 0.0251, within 1 sd of the peak). EVERY
              # absolute-epoch schedule is rescaled by 60/120: lambda warmup 40->20, LR
              # warmup 5->3 (2.5 rounded up), save_every 20->10. Gumbel T is already
              # relative to cfg.epochs.

              depth_source='lidar',                      # LiDAR depth labels (train split only)
              hm_norm='centernet', hm_pos_weight=1.0,    # standard CenterNet heatmap loss
              gauss_mode='centerpoint',                  # CenterPoint Gaussian, same on train and val
              eval_score_thresh=0.05,
              depth_loss_weight='inv_sqrt_freq',        # depth A/B winner (arm B)
              )
cfg = replace(cfg, resume=os.path.exists(f'{cfg.root}/checkpoints/{cfg.exp_name}/last.pth'))
cfg.validate()
tr, va, te = build_loaders(cfg)
# refuse to train on a partially restored LiDAR set (frames without ego .pcd are dropped)
assert len(tr.dataset) >= 6000, f'only {len(tr.dataset)} train frames have LiDAR -- restore incomplete?'
assert len(te.dataset) > 0 and len(va.dataset) > 0
print(f'Training: {cfg.exp_name}  resume={cfg.resume}  train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}  '
      f'depth={cfg.depth_source} hm={cfg.hm_norm}/pw{cfg.hm_pos_weight} gauss={cfg.gauss_mode} thresh={cfg.eval_score_thresh} depth_w={cfg.depth_loss_weight} '
      f'epochs={cfg.epochs} lam_warm={cfg.warmup_epochs} lr_warm={cfg.lr_warmup_epochs}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/ours_nodepth60_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
