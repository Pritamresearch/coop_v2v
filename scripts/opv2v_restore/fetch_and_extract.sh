#!/bin/bash
#SBATCH --partition=k2-hipri
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:59:00
# usage: sbatch fetch_and_extract.sh <box_file_id> <archive_name> <extract_into> <suffix> [--only-existing-dirs]
set -u
# override with env vars: RESTORE_PY (python with `requests`), RESTORE_ARCHIVES (where full .zip files are kept)
T=/mnt/scratch2/users/2641309/projects/coop_v2v/scripts/opv2v_restore
PY=${RESTORE_PY:-/mnt/scratch2/users/2641309/tools/dl_venv/bin/python}
A=${RESTORE_ARCHIVES:-/mnt/scratch2/users/2641309/datasets/OPV2V_archives}
FID=$1; NAME=$2; OUT=$3; SUF=$4; EXTRA=${5:-}
SRC="https://ucla.app.box.com/index.php?rm=box_download_shared_file&shared_name=vxetzti0z0dv97yeh1jgxdprxxqucs2a&file_id=$FID"
EXPECTED=$(curl -sSL -A "Mozilla/5.0" -r 0-0 -D - -o /dev/null "$SRC" | tr -d '\r' | awk -F/ 'tolower($0) ~ /^content-range/ {print $NF}' | tail -1)
echo "$(date) $NAME expected size: $EXPECTED bytes"
have() { stat -c %s "$A/$NAME" 2>/dev/null || echo 0; }
if [ "$(have)" != "$EXPECTED" ]; then
  echo "$(date) downloading (resuming from $(have) bytes)"
  curl -sSL -A "Mozilla/5.0" --retry 10 --retry-delay 5 -C - -o "$A/$NAME" "$SRC"
fi
if [ "$(have)" != "$EXPECTED" ]; then echo "$(date) incomplete: $(have)/$EXPECTED -- next job in chain will resume"; exit 0; fi
echo "$(date) download complete; testing archive index"
$PY $T/boxzip.py list "$A/$NAME" | tail -3
echo "$(date) extracting '$SUF' into $OUT $EXTRA"
$PY -u $T/boxzip.py extract "$A/$NAME" "$OUT" "$SUF" $EXTRA
echo "$(date) done"
