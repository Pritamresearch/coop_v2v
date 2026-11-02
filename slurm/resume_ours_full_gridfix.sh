#!/bin/bash
#SBATCH --job-name=ours_full_gridfix_resume
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --dependency=afterany:10387364
#SBATCH --output=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_ours_full_gridfix_resume.out
#SBATCH --error=/mnt/scratch2/users/2641309/projects/coop_v2v/logs/slurm_%j_ours_full_gridfix_resume.err

PROJECT=/mnt/scratch2/users/2641309/projects/coop_v2v
cd $PROJECT
mkdir -p logs
flight env activate conda@opencood
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v
export PYTHONPATH=$PROJECT:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "====== ours_full_gridfix: RESUME from last.pth, at $(date) ======"
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
cfg = make('ours_full', base)
cfg = replace(cfg, exp_name='ours_full_gridfix', epochs=120, resume=True)
print(f'Resuming: {cfg.exp_name}  grid={cfg.grid_H}x{cfg.grid_W}  bev=+/-{cfg.bev_x_max}m  epochs={cfg.epochs}  resume={cfg.resume}')
train(cfg, tr, va, te)
" 2>&1 | tee -a logs/ours_full_gridfix_resume_\${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"