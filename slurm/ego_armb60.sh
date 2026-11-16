#!/bin/bash
#SBATCH --job-name=ego_armb60
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100,k2-gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j_ego_armb60.out
#SBATCH --error=logs/slurm_%j_ego_armb60.err
#
# Matched ego-only baseline for ours_nodepth60: arm B depth (inv_sqrt_freq), FRESH start,
# 60-epoch budget with every absolute-epoch schedule rescaled identically. Same seed.
# Resumes automatically from results/checkpoints/ego_armb60/last.pth
# when it exists, so the same script can be chained past the 3-day limit:
#   j=$(sbatch --parsable slurm/ego_armb60.sh)
#   sbatch --dependency=afterany:$j slurm/ego_armb60.sh

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
# Locate slurm/_env.sh. Under sbatch the batch script may be staged to the node's
# spool directory, so ${BASH_SOURCE[0]} does not necessarily point into the repo.
# Try that first, then the submit directory, then $PWD. Fail loudly -- a silent
# miss here leaves the job running against whatever environment it inherited.
_COOP_ENV=""
for _c in "$(dirname "${BASH_SOURCE[0]}")/_env.sh" \
          "${SLURM_SUBMIT_DIR:-/nonexistent}/slurm/_env.sh" \
          "$PWD/slurm/_env.sh"; do
    if [ -f "$_c" ]; then _COOP_ENV="$_c"; break; fi
done
if [ -z "$_COOP_ENV" ]; then
    echo "FATAL: cannot locate slurm/_env.sh (tried BASH_SOURCE, SLURM_SUBMIT_DIR, PWD)" >&2
    exit 1
fi
# shellcheck disable=SC1090
source "$_COOP_ENV"
# Fail loudly. Placed AFTER sourcing _env.sh so conda activation is not subject to -u.
#   -e         stop at the first failing command
#   -u         an unset variable is an error
#   pipefail   `python ... | tee x.log` reports PYTHON's status, not tee's
# Without pipefail a crashed python left the pipeline exit code at 0, so the job
# reported COMPLETED and the trailing "Done" echo printed as if nothing was wrong.
set -euo pipefail
echo "====== ego_armb60 at $(date) on $(hostname) ======"
# pre-training check: collaborator features must land on the right ego cells, with and without augmentation
python -u -c "
import os
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders
base = Config()
base.validate()
cfg = make('no_fusion', base)
cfg = replace(cfg, exp_name='ego_armb60',
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
              depth_loss_weight='inv_sqrt_freq',        # arm B
              )
cfg = replace(cfg, resume=os.path.exists(f'{cfg.root}/checkpoints/{cfg.exp_name}/last.pth'))
cfg.validate()
tr, va, te = build_loaders(cfg)
# refuse to train on a partially restored LiDAR set (frames without ego .pcd are dropped)
assert len(tr.dataset) >= 6000, f'only {len(tr.dataset)} train frames have LiDAR -- restore incomplete?'
assert len(te.dataset) > 0 and len(va.dataset) > 0
print(f'Training: {cfg.exp_name}  resume={cfg.resume}  train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}  '
      f'depth={cfg.depth_source} hm={cfg.hm_norm}/pw{cfg.hm_pos_weight} gauss={cfg.gauss_mode} thresh={cfg.eval_score_thresh} depth_w={cfg.depth_loss_weight} epochs={cfg.epochs} lam_warm={cfg.warmup_epochs} lr_warm={cfg.lr_warmup_epochs}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/ego_armb60_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
