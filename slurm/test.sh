#!/bin/bash
#SBATCH --job-name=coop_v2v_test
#SBATCH --partition=k2-gpu-v100,k2-gpu-a100,k2-gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j_test.out
#SBATCH --error=logs/slurm_%j_test.err
#
# HELD-OUT TEST SPLIT. Read diagnostics/2026-11_cooperation/PREREG_COOPERATION.md first.
#
# Validation has been used for decisions (60-epoch budget, image-aug A/B, threshold
# calibration, depth arm), so val is no longer an unbiased estimate for these methods.
# Final numbers for every method therefore come from ONE evaluation on test, run after
# every choice is frozen. Nothing is tuned, selected or early-stopped on test.
#
# This script refuses to run until that freeze is declared explicitly:
#   sbatch --export=ALL,COOP_TEST_FREEZE=1 slurm/test.sh
# The guard exists because a stray submission of this script would quietly spend the
# one evaluation the protocol allows.

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

# Fail loudly. Placed AFTER sourcing _env.sh so conda activation is not subject to -u.
set -euo pipefail

if [ "${COOP_TEST_FREEZE:-0}" != "1" ]; then
    echo "REFUSING to evaluate on the held-out test split." >&2
    echo "All method choices must be frozen first; then resubmit with:" >&2
    echo "  sbatch --export=ALL,COOP_TEST_FREEZE=1 slurm/test.sh" >&2
    exit 1
fi

echo "====== Test evaluation at $(date) on $(hostname) ======"
echo "COOP_TEST_FREEZE=1 given: the method choices are declared frozen."
rc=0
python -u run_test.py --all --n_qual 5 2>&1 | tee "logs/test_${SLURM_JOB_ID}.log" || rc=$?
echo "====== Done at $(date), rc=$rc ======"
exit $rc
