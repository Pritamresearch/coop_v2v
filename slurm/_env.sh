# Shared SLURM preamble. Sourced by every slurm/*.sh.
#
# Cluster-specific settings live HERE ONLY, so moving to another machine means editing one file
# (or exporting these variables); the job scripts themselves stay portable.
#
#   COOP_CONDA_SH   path to conda's profile.d/conda.sh   (default: the Kelvin2 flight env)
#   COOP_CONDA_ENV  conda environment name               (default: coop_v2v)
#   OPV2V_ROOT      dataset root                         (see src/config.py)
#   COOP_RESULTS    output root                          (see src/config.py)

# Repo root is derived from this file's location -- no absolute path needed.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT" || exit 1
mkdir -p logs

COOP_CONDA_SH="${COOP_CONDA_SH:-$HOME/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh}"
COOP_CONDA_ENV="${COOP_CONDA_ENV:-coop_v2v}"
if [ -f "$COOP_CONDA_SH" ]; then
    command -v flight >/dev/null 2>&1 && flight env activate conda@opencood
    # shellcheck disable=SC1090
    source "$COOP_CONDA_SH"
    conda activate "$COOP_CONDA_ENV"
else
    echo "WARNING: $COOP_CONDA_SH not found; assuming the environment is already active." >&2
fi

# Dataset root. The repo-relative default (./data/OPV2V) works via the symlink created at setup;
# this keeps jobs working even if that symlink is missing.
export OPV2V_ROOT="${OPV2V_ROOT:-/mnt/scratch2/users/2641309/datasets/OPV2V}"

export PYTHONPATH="$PROJECT:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
