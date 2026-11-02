"""
Restores LSS depth-bin resolution after the BEV-grid widening increased
lss_d_max (50.0 -> 90.0) without adjusting depth_K, which had coarsened
per-bin depth from ~0.72m to ~1.34m (confirmed via depth_preprocessing_audit.py,
2026-10-27: mean real quantization error 0.337m = 0.84 BEV cells).

depth_K: 64 -> 128 restores bin width to 86.0/128 = 0.672m, slightly finer
than the original 46.0/64 = 0.719m.

Run from the repo root on kelvin2, AFTER patch_frustum_convention.py:
    python patch_depth_k.py
Then verify:
    python -c "from src.config import Config; c=Config(); c.validate(); print(c.depth_K)"
Expect: 128
"""
from __future__ import annotations

import shutil
import time

PATH = "src/config.py"
OLD = "    depth_K: int = 64"
NEW = "    depth_K: int = 128"


def main():
    with open(PATH, "r") as f:
        src = f.read()

    n = src.count(OLD)
    if n != 1:
        raise SystemExit(f"ABORT: expected exactly 1 occurrence of {OLD!r}, found {n}.")

    backup_path = f"{PATH}.bak_depthK64_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(PATH, backup_path)
    print(f"Backed up current config to {backup_path}")

    src = src.replace(OLD, NEW)
    with open(PATH, "w") as f:
        f.write(src)

    print(f"Patched {PATH}: {OLD.strip()}  ->  {NEW.strip()}")


if __name__ == "__main__":
    main()