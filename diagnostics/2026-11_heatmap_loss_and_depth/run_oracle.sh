#!/bin/bash
#SBATCH --job-name=depth_oracle
#SBATCH --partition=k2-gpu-h100,k2-gpu-a100,k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/scratch2/users/2641309/tmp_claude_diag/ab/slurm_%j_oracle.out
cd /mnt/scratch2/users/2641309/projects/coop_v2v
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v
nvidia-smi -L
set -e
python -u /mnt/scratch2/users/2641309/tmp_claude_diag/ab/oracle_depth.py eval /mnt/scratch2/users/2641309/tmp_claude_diag/ab/nf_epoch_40.pth epoch40
python -u /mnt/scratch2/users/2641309/tmp_claude_diag/ab/oracle_depth.py finetune lidar 3400 /mnt/scratch2/users/2641309/tmp_claude_diag/ab/ckpt/depth_lidar_3400.pth
python -u /mnt/scratch2/users/2641309/tmp_claude_diag/ab/oracle_depth.py finetune zoe   3400 /mnt/scratch2/users/2641309/tmp_claude_diag/ab/ckpt/depth_zoe_3400.pth
python -u /mnt/scratch2/users/2641309/tmp_claude_diag/ab/oracle_depth.py evalpred finetuned /mnt/scratch2/users/2641309/tmp_claude_diag/ab/nf_epoch_40.pth /mnt/scratch2/users/2641309/tmp_claude_diag/ab/ckpt/depth_zoe_3400.pth /mnt/scratch2/users/2641309/tmp_claude_diag/ab/ckpt/depth_lidar_3400.pth
