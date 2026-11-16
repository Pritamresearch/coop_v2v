#!/bin/bash
#SBATCH --job-name=no_fusion_gridfix_resume
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --dependency=afterany:10387363
#SBATCH --output=logs/slurm_%j_no_fusion_gridfix_resume.out
#SBATCH --error=logs/slurm_%j_no_fusion_gridfix_resume.err

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
echo "====== no_fusion_gridfix: RESUME from last.pth, at $(date) ======"
python -u -c "
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders

base = Config()
base.validate()
assert base.bev_x_max == 51.2 and base.cell_m == 0.4 and base.grid_H == 256, \
    f'grid fix not active -- got bev_x_max={base.bev_x_max} cell_m={base.cell_m} grid_H={base.grid_H}'
tr, va, te = build_loaders(base)
cfg = make('no_fusion', base)
cfg = replace(cfg, exp_name='no_fusion_gridfix', epochs=120, resume=True)
print(f'Resuming: {cfg.exp_name}  grid={cfg.grid_H}x{cfg.grid_W}  bev=+/-{cfg.bev_x_max}m  epochs={cfg.epochs}  resume={cfg.resume}')
train(cfg, tr, va, te)
" 2>&1 | tee -a logs/no_fusion_gridfix_resume_\${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"