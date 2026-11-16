#!/bin/bash
#SBATCH --job-name=recov_bytes
#SBATCH --partition=k2-hipri
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:55:00
#SBATCH --array=0-5
#SBATCH --output=logs/slurm_%A_%a_recov.out
#SBATCH --error=logs/slurm_%A_%a_recov.err
#
# Recoverable-subset cooperation checks, 6 shards over val frames:
#   ours_nodepth60 (cooperative) vs ego_armb60 (ego-only), both at the final epoch.
# Shard 0 additionally measures transmitted bytes under the fixed accounting.
_COOP_ENV=""
for _c in "$(dirname "${BASH_SOURCE[0]}")/_env.sh" \
          "${SLURM_SUBMIT_DIR:-/nonexistent}/slurm/_env.sh" \
          "$PWD/slurm/_env.sh"; do
    if [ -f "$_c" ]; then _COOP_ENV="$_c"; break; fi
done
if [ -z "$_COOP_ENV" ]; then
    echo "FATAL: cannot locate slurm/_env.sh" >&2; exit 1
fi
# shellcheck disable=SC1090
source "$_COOP_ENV"

OUT=${COOP_DIAG:-/mnt/scratch2/users/2641309/tmp_claude_diag}
R=${COOP_RESULTS:-$(pwd)/results}
mkdir -p "$OUT"
echo "====== recoverable shard ${SLURM_ARRAY_TASK_ID} at $(date) on $(hostname) ======"
python -u scripts/fusion_checks.py "${SLURM_ARRAY_TASK_ID}" 6 300 \
    "$R/checkpoints/ours_nodepth60/last.pth" \
    "$R/checkpoints/ego_armb60/last.pth" \
    "$OUT/fc60_shard${SLURM_ARRAY_TASK_ID}.pkl" 150

if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then
    python -u scripts/measure_bytes.py --run ours_nodepth60 --frames 200 \
        --out "$OUT/bytes_ours_nodepth60.json"
fi
echo "====== Done at $(date) ======"
