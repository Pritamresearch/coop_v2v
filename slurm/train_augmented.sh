#!/bin/bash
#SBATCH --job-name=aug_test
#SBATCH --partition=k2-gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-04:00:00
#SBATCH --output=logs/slurm_%j_aug_test.out
#SBATCH --error=logs/slurm_%j_aug_test.err

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
# Fail loudly. Placed AFTER sourcing _env.sh so conda activation is not subject to -u.
#   -e         stop at the first failing command
#   -u         an unset variable is an error
#   pipefail   `python ... | tee x.log` reports PYTHON's status, not tee's
# Without pipefail a crashed python left the pipeline exit code at 0, so the job
# reported COMPLETED and the trailing "Done" echo printed as if nothing was wrong.
set -euo pipefail
echo "====== BEV augmentation run at $(date) ======"
python -u quick_augment_test.py 2>&1 | tee logs/aug_test_${SLURM_JOB_ID}.log
echo "====== Done at $(date) ======"
