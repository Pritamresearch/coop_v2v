#!/bin/bash
#SBATCH --job-name=ab_patched
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/scratch2/users/2641309/tmp_claude_diag/ab/slurm_%j_patched.out
cd /mnt/scratch2/users/2641309/projects/coop_v2v
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v
nvidia-smi -L
python -u /mnt/scratch2/users/2641309/tmp_claude_diag/ab/finetune_ab.py patched /mnt/scratch2/users/2641309/tmp_claude_diag/ab/ckpt 6374 3 || exit 1
python -u /mnt/scratch2/users/2641309/tmp_claude_diag/ab/eval_ab.py 400 /mnt/scratch2/users/2641309/tmp_claude_diag/ab/eval_patched.tsv  /mnt/scratch2/users/2641309/tmp_claude_diag/ab/ckpt/patched_2125.pth /mnt/scratch2/users/2641309/tmp_claude_diag/ab/ckpt/patched_4249.pth /mnt/scratch2/users/2641309/tmp_claude_diag/ab/ckpt/patched_6374.pth
