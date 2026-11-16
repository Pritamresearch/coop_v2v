#!/bin/bash
#SBATCH --job-name=imgaug_readout
#SBATCH --partition=k2-hipri
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:55:00
#SBATCH --output=logs/slurm_%j_imgaug_readout.out
#SBATCH --error=logs/slurm_%j_imgaug_readout.err
#
# Pre-registered epoch-20 image-augmentation readout (CPU).
# Submitted with sbatch, not srun, so it survives the submitting session ending.
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
OUT=${COOP_DIAG:-/mnt/scratch2/users/2641309/tmp_claude_diag}
echo "====== imgaug readout at $(date) on $(hostname) ======"
python -u scripts/imgaug_readout.py --frames 200 --bootstrap 2000 --budget-min 145 \
       --out "$OUT/imgaug_readout_ep20.json"
echo "====== Done at $(date) ======"
