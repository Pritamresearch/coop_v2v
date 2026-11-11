#!/bin/bash
#SBATCH --job-name=aug_full
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j_aug_full.out
#SBATCH --error=logs/slurm_%j_aug_full.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== Full augmented run 120 epochs at $(date) ======"
python -u -c "
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders

base = Config()
base.validate()
tr, va, te = build_loaders(base)
cfg = make('ours_full', base)
cfg = replace(cfg, exp_name='ours_full_aug_full', epochs=120)
print(f'Training: {cfg.exp_name}  augment={cfg.augment}  epochs={cfg.epochs}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/aug_full_\${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
