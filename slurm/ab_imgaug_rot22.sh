#!/bin/bash
#SBATCH --job-name=ab_imgaug_rot22
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100,k2-gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm_%j_ab_imgaug_rot22.out
#SBATCH --error=logs/slurm_%j_ab_imgaug_rot22.err
#
# Image-augmentation A/B, BEVDet geometric image aug + BEV flip/scale, rotation narrowed to +-22.5 deg (BEVDet's range).
# Ego-only, depth arm B (inv_sqrt_freq), same seed and 120-epoch LR schedule as the baseline
# ab_depth_reweight, stopped after epoch 20 with checkpoints every 2 epochs so the two arms and
# the baseline can be compared epoch for epoch.
# Resumes automatically from results/checkpoints/ab_imgaug_rot22/last.pth
# when it exists, so the same script can be chained past the 3-day limit:
#   j=$(sbatch --parsable slurm/ab_imgaug_rot22.sh)
#   sbatch --dependency=afterany:$j slurm/ab_imgaug_rot22.sh

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
echo "====== ab_imgaug_rot22 at $(date) on $(hostname) ======"
# Gates: all three must pass with the augmentation this arm uses, or we do not train.
python -u -m src.data.img_augment || { echo 'GATE FAILED: image-aug post_rot/post_tran'; exit 1; }
python -u scripts/check_collab_warp.py --frames 6 || { echo 'GATE FAILED: collaborator warp (flip/scale)'; exit 1; }
python -u roundtrip_test.py || { echo 'GATE FAILED: frustum round-trip'; exit 1; }
COOP_IMG_AUG=1 python -u e2e_geometry_integration_test.py || { echo 'GATE FAILED: e2e coverage with image aug'; exit 1; }
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
cfg = replace(cfg, exp_name='ab_imgaug_rot22', epochs=120,
              depth_source='lidar',                      # LiDAR depth labels (train split only)
              hm_norm='centernet', hm_pos_weight=1.0,    # standard CenterNet heatmap loss
              gauss_mode='centerpoint',                  # CenterPoint Gaussian, same on train and val
              eval_score_thresh=0.05,
              depth_loss_weight='inv_sqrt_freq',        # arm B
              max_epoch=20, save_every=2,
              img_aug=True,                             # BEVDet resize/crop/flip/rotate, no photometric
              bev_flip=True, bev_scale_range=(0.95, 1.05),
              bev_rot_deg=22.5)
cfg = replace(cfg, resume=os.path.exists(f'{cfg.root}/checkpoints/{cfg.exp_name}/last.pth'))
cfg.validate()
tr, va, te = build_loaders(cfg)
# refuse to train on a partially restored LiDAR set (frames without ego .pcd are dropped)
assert len(tr.dataset) >= 6000, f'only {len(tr.dataset)} train frames have LiDAR -- restore incomplete?'
assert len(te.dataset) > 0 and len(va.dataset) > 0
print(f'Training: {cfg.exp_name}  resume={cfg.resume}  train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}  '
      f'depth={cfg.depth_source} hm={cfg.hm_norm}/pw{cfg.hm_pos_weight} gauss={cfg.gauss_mode} thresh={cfg.eval_score_thresh} depth_w={cfg.depth_loss_weight} img_aug={cfg.img_aug} rot={cfg.bev_rot_deg} flip={cfg.bev_flip} scale={cfg.bev_scale_range}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/ab_imgaug_rot22_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
