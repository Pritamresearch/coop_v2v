#!/bin/bash
#SBATCH --job-name=nofusion_test
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm_%j_nofusion_test.out
#SBATCH --error=logs/slurm_%j_nofusion_test.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
python -u -c "
import sys; sys.path.insert(0,'.')
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders

base = Config()
base.validate()
tr, va, te = build_loaders(base)
cfg = make('no_fusion', base)
cfg = replace(cfg, epochs=15)
print(f'Training: {cfg.exp_name}  epochs={cfg.epochs}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/nofusion_test_\${SLURM_JOB_ID}.log
