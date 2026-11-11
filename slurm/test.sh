#!/bin/bash
#SBATCH --job-name=coop_v2v_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_%j_test.out
#SBATCH --error=logs/slurm_%j_test.err

PROJECT=
cd $PROJECT

flight env activate conda@opencood
conda activate coop_v2v
export PYTHONPATH=$PROJECT:$PYTHONPATH

echo "====== Test evaluation at $(date) ======"
# Loads best.pth for all presets, generates 3D box figures + curves + robustness
python run_test.py --all --n_qual 5 2>&1 | tee logs/test_${SLURM_JOB_ID}.log

echo "====== Done at $(date) ======"
