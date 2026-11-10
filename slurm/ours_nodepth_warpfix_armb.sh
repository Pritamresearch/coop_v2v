#!/bin/bash
#SBATCH --job-name=ours_nodepth_warpfix_armb
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100,k2-gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_ours_nodepth_warpfix_armb.out
#SBATCH --error=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_ours_nodepth_warpfix_armb.err
#
# 'NO DEPTH TERM' ABLATION: ours_nodepth (sender utility = ||F||, no exp(-alpha*D)) with the
# collaborator warp fix (a624570) + depth A/B arm B. Deliberately labelled as an ablation: the
# depth-aware utility was measured degenerate (rho 0.977 vs magnitude_only) and per-pixel depth
# entropy is ANTI-correlated with depth error beyond 20 m, so this is the honest baseline that a
# future learned-reliability depth term must beat.
# (depth_loss_weight='inv_sqrt_freq': ~2x far-range depth mass near truth at epoch 20).
# No image augmentation (that A/B is separate). 120 epochs, chained past the 3-day limit.
# Resumes automatically from results/checkpoints/ours_nodepth_warpfix_armb/last.pth
# when it exists, so the same script can be chained past the 3-day limit:
#   j=$(sbatch --parsable slurm/ours_nodepth_warpfix_armb.sh)
#   sbatch --dependency=afterany:$j slurm/ours_nodepth_warpfix_armb.sh

PROJECT=/mnt/scratch2/users/2641309/projects/coop_v2v
cd $PROJECT
mkdir -p logs

flight env activate conda@opencood
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v

export PYTHONPATH=$PROJECT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "====== ours_nodepth_warpfix_armb at $(date) on $(hostname) ======"
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
cfg = replace(cfg, exp_name='ours_nodepth_warpfix_armb', epochs=120,
              depth_source='lidar',                      # LiDAR depth labels (train split only)
              hm_norm='centernet', hm_pos_weight=1.0,    # standard CenterNet heatmap loss
              gauss_mode='centerpoint',                  # CenterPoint Gaussian, same on train and val
              eval_score_thresh=0.05,
              depth_loss_weight='inv_sqrt_freq',        # depth A/B winner (arm B)
              save_every=10)
cfg = replace(cfg, resume=os.path.exists(f'{cfg.root}/checkpoints/{cfg.exp_name}/last.pth'))
cfg.validate()
tr, va, te = build_loaders(cfg)
# refuse to train on a partially restored LiDAR set (frames without ego .pcd are dropped)
assert len(tr.dataset) >= 6000, f'only {len(tr.dataset)} train frames have LiDAR -- restore incomplete?'
assert len(te.dataset) > 0 and len(va.dataset) > 0
print(f'Training: {cfg.exp_name}  resume={cfg.resume}  train={len(tr.dataset)} val={len(va.dataset)} test={len(te.dataset)}  '
      f'depth={cfg.depth_source} hm={cfg.hm_norm}/pw{cfg.hm_pos_weight} gauss={cfg.gauss_mode} thresh={cfg.eval_score_thresh} depth_w={cfg.depth_loss_weight}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/ours_nodepth_warpfix_armb_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
