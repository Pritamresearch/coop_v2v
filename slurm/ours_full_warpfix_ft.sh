#!/bin/bash
#SBATCH --job-name=ours_full_warpfix_ft
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100,k2-gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j_ours_full_warpfix_ft.out
#SBATCH --error=logs/slurm_%j_ours_full_warpfix_ft.err
#
# Quick check of the warp fix (commit a624570): fine-tune ours_full_lidardepth from its epoch-20 checkpoint for 4 epochs
# (21-24) with the FIXED augmentation, checkpoint every epoch. Same settings and LR schedule as the original run.
# Resumes automatically from results/checkpoints/ours_full_warpfix_ft/last.pth
# when it exists, so the same script can be chained past the 3-day limit:
#   j=$(sbatch --parsable slurm/ours_full_warpfix_ft.sh)
#   sbatch --dependency=afterany:$j slurm/ours_full_warpfix_ft.sh

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
echo "====== ours_full_warpfix_ft at $(date) on $(hostname) ======"
CK=results/checkpoints/ours_full_warpfix_ft; mkdir -p $CK
[ -f $CK/last.pth ] || cp "${COOP_SEED_CKPT:-/mnt/scratch2/users/2641309/tmp_claude_diag/oursfull_ep20.pth}" $CK/last.pth   # start point: ours_full epoch 20
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
cfg = replace(cfg, exp_name='ours_full_warpfix_ft', epochs=120,
              depth_source='lidar',                      # LiDAR depth labels (train split only)
              hm_norm='centernet', hm_pos_weight=1.0,    # standard CenterNet heatmap loss
              gauss_mode='centerpoint',                  # CenterPoint Gaussian, same on train and val
              eval_score_thresh=0.05,
              max_epoch=24, save_every=1)                # epochs 21-24, keep every checkpoint
cfg = replace(cfg, resume=os.path.exists(f'{cfg.root}/checkpoints/{cfg.exp_name}/last.pth'))
cfg.validate()
tr, va, te = build_loaders(cfg)
# refuse to train on a partially restored LiDAR set (frames without ego .pcd are dropped)
assert cfg.resume, 'expected to resume from the epoch-20 checkpoint'
assert len(tr.dataset) >= 6000, f'only {len(tr.dataset)} train frames have LiDAR -- restore incomplete?'
assert len(te.dataset) > 0 and len(va.dataset) > 0
print(f'Training: {cfg.exp_name}  resume={cfg.resume}  train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}  '
      f'depth={cfg.depth_source} hm={cfg.hm_norm}/pw{cfg.hm_pos_weight} gauss={cfg.gauss_mode} thresh={cfg.eval_score_thresh} max_epoch={cfg.max_epoch} save_every={cfg.save_every}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/ours_full_warpfix_ft_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
