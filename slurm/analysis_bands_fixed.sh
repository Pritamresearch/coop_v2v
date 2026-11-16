#!/bin/bash
#SBATCH --job-name=bands_fixed
#SBATCH --partition=k2-hipri
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:55:00
#SBATCH --output=logs/slurm_%j_bands_fixed.out
#SBATCH --error=logs/slurm_%j_bands_fixed.err
#
# Band comparison re-run on the FIXED hard_topk (a8382d5). Both models in one pass on
# the same frames, as the pairing requires -- not only the cooperative arm.
_COOP_ENV=""
for _c in "$(dirname "${BASH_SOURCE[0]}")/_env.sh" "${SLURM_SUBMIT_DIR:-/nonexistent}/slurm/_env.sh" "$PWD/slurm/_env.sh"; do
    if [ -f "$_c" ]; then _COOP_ENV="$_c"; break; fi
done
[ -n "$_COOP_ENV" ] || { echo "FATAL: cannot locate slurm/_env.sh" >&2; exit 1; }
# shellcheck disable=SC1090
source "$_COOP_ENV"
set -euo pipefail
OUT=${COOP_DIAG:-/mnt/scratch2/users/2641309/tmp_claude_diag}
R=${COOP_RESULTS:-$(pwd)/results}
echo "====== bands FIXED at $(date) on $(hostname) ======"
rc=0
python -u scripts/diag_bands.py compare \
    ours_nodepth60 "$R/checkpoints/ours_nodepth60/last.pth" \
    ego_armb60     "$R/checkpoints/ego_armb60/last.pth" \
    "$OUT/bands_nodepth60_vs_ego60_FIXED.json" 200 145 || rc=$?
echo "====== Done at $(date), rc=$rc ======"
exit $rc
