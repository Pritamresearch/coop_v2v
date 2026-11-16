#!/bin/bash
#SBATCH --job-name=bands_nodepth_ego
#SBATCH --partition=k2-hipri
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:55:00
#SBATCH --output=logs/slurm_%j_bands_nodepth_ego.out
#SBATCH --error=logs/slurm_%j_bands_nodepth_ego.err
#
# Per-distance-band comparison of the matched 60-epoch pair, same val frames:
#   ours_nodepth60 (cooperative) vs ego_armb60 (ego-only)
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
OUT=${COOP_DIAG:-/mnt/scratch2/users/2641309/tmp_claude_diag}
R=${COOP_RESULTS:-$(pwd)/results}
echo "====== bands nodepth-vs-ego at $(date) on $(hostname) ======"
python -u scripts/diag_bands.py compare \
    ours_nodepth60 "$R/checkpoints/ours_nodepth60/last.pth" \
    ego_armb60     "$R/checkpoints/ego_armb60/last.pth" \
    "$OUT/bands_nodepth60_vs_ego60.json" 200 145
echo "====== Done at $(date) ======"
