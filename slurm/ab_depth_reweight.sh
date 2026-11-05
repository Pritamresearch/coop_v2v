#!/bin/bash
#SBATCH --job-name=ab_depth_reweight
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100,k2-gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_ab_depth_reweight.out
#SBATCH --error=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_ab_depth_reweight.err
#
# Far-range depth A/B, arm B: uniform bins + depth loss weighted 1/sqrt(bin frequency).
# Otherwise identical to slurm/no_fusion_lidardepth.sh (the baseline): same seed and 120-epoch LR
# schedule, but stops after epoch index 20 so epoch_20.pth matches the baseline's epoch_20.pth.
# Saves epoch_<N>.pth every 2 epochs; the baseline has validation AP every 2 epochs in train_log.csv
# but only epoch_20.pth as a checkpoint, so per-band diagnostics can match only at epoch 20.
# Resumes automatically from results/checkpoints/ab_depth_reweight/last.pth
# when it exists, so the same script can be chained past the 3-day limit:
#   j=$(sbatch --parsable slurm/ab_depth_reweight.sh)
#   sbatch --dependency=afterany:$j slurm/ab_depth_reweight.sh

PROJECT=/mnt/scratch2/users/2641309/projects/coop_v2v
cd $PROJECT
mkdir -p logs

flight env activate conda@opencood
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v

export PYTHONPATH=$PROJECT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "====== ab_depth_reweight at $(date) on $(hostname) ======"
python -u -c "
import os
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders
base = Config()
base.validate()
cfg = make('no_fusion', base)
cfg = replace(cfg, exp_name='ab_depth_reweight', epochs=120,
              depth_source='lidar',                      # LiDAR depth labels (train split only)
              hm_norm='centernet', hm_pos_weight=1.0,    # standard CenterNet heatmap loss
              gauss_mode='centerpoint',                  # CenterPoint Gaussian, same on train and val
              eval_score_thresh=0.05,
              depth_loss_weight='inv_sqrt_freq',          # A/B arm B: range-reweighted depth loss
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
" 2>&1 | tee logs/ab_depth_reweight_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
