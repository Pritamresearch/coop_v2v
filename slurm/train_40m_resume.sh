#!/bin/bash
#SBATCH --job-name=40m_res
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/slurm_%j_40m_resume.out
#SBATCH --error=logs/slurm_%j_40m_resume.err

# All machine-specific settings (repo root, conda, dataset root) live in slurm/_env.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_env.sh"
echo "====== 40m BEV range, resume to 120 epochs at $(date) ======"
python -u -c "
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders

base = Config()
base.validate()
tr, va, te = build_loaders(base)
cfg = make('ours_full', base)
cfg = replace(cfg, exp_name='ours_full_40m', epochs=120, resume=True)
print(f'Resuming: {cfg.exp_name}  epochs={cfg.epochs}')
train(cfg, tr, va, te)
" 2>&1 | tee logs/40m_resume_\${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
