#!/bin/bash
#SBATCH --job-name=ab_depth_logbins
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100,k2-gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm_%j_ab_depth_logbins.out
#SBATCH --error=logs/slurm_%j_ab_depth_logbins.err
#
# Far-range depth A/B, arm A: log-spaced depth bins.
# Otherwise identical to slurm/no_fusion_lidardepth.sh (the baseline): same seed and 120-epoch LR
# schedule, but stops after epoch index 20 so epoch_20.pth matches the baseline's epoch_20.pth.
# Saves epoch_<N>.pth every 2 epochs; the baseline has validation AP every 2 epochs in train_log.csv
# but only epoch_20.pth as a checkpoint, so per-band diagnostics can match only at epoch 20.
# Resumes automatically from results/checkpoints/ab_depth_logbins/last.pth
# when it exists, so the same script can be chained past the 3-day limit:
#   j=$(sbatch --parsable slurm/ab_depth_logbins.sh)
#   sbatch --dependency=afterany:$j slurm/ab_depth_logbins.sh

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
echo "====== ab_depth_logbins at $(date) on $(hostname) ======"
python -u -c "
import os
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders
base = Config()
base.validate()
cfg = make('no_fusion', base)
cfg = replace(cfg, exp_name='ab_depth_logbins', epochs=120,
              depth_source='lidar',                      # LiDAR depth labels (train split only)
              hm_norm='centernet', hm_pos_weight=1.0,    # standard CenterNet heatmap loss
              gauss_mode='centerpoint',                  # CenterPoint Gaussian, same on train and val
              eval_score_thresh=0.05,
              depth_bins='log',                         # A/B arm A: log-spaced depth bins
              max_epoch=20, save_every=2)               # epoch_2.pth ... epoch_20.pth for epoch-matched comparison
cfg = replace(cfg, resume=os.path.exists(f'{cfg.root}/checkpoints/{cfg.exp_name}/last.pth'))
cfg.validate()
tr, va, te = build_loaders(cfg)
# refuse to train on a partially restored LiDAR set (frames without ego .pcd are dropped)
assert len(tr.dataset) >= 6000, f'only {len(tr.dataset)} train frames have LiDAR -- restore incomplete?'
assert len(te.dataset) > 0 and len(va.dataset) > 0
print(f'Training: {cfg.exp_name}  resume={cfg.resume}  train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}  '
      f'depth={cfg.depth_source} hm={cfg.hm_norm}/pw{cfg.hm_pos_weight} gauss={cfg.gauss_mode} thresh={cfg.eval_score_thresh} bins={cfg.depth_bins} depth_w={cfg.depth_loss_weight} max_epoch={cfg.max_epoch} save_every={cfg.save_every}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/ab_depth_logbins_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
