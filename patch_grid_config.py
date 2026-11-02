"""
Widens the BEV grid from +/-20m (80x80 @ 0.5m) to +/-51.2m (256x256 @ 0.4m) --
the IFTR/RayFusion/HEAL benchmark convention -- and extends lss_d_max so the
LSS frustum actually reaches the new grid's far corners (sqrt(51.2^2+51.2^2)
= 72.4m).

Confirmed empirically (check_grid_range.py, 2026-10-24) that the current
+/-20m grid excludes ~95% of real GT vehicles (1 box vs 233 boxes across the
same 20 real validation frames). See project doc:
claude/diagnosis_2026-10-24_bev_grid_too_small_dominant_finding.md

Exact-match patch with assert-count==1 guards, same pattern used for the
NMS/pos_weight fixes earlier this session. Makes a timestamped backup first.

Run from the repo root on kelvin2:
    python patch_grid_config.py
Then verify:
    python -c "from src.config import Config; c=Config(); c.validate(); \
print(c.bev_x_min, c.bev_x_max, c.bev_y_min, c.bev_y_max, c.cell_m, \
c.grid_H, c.grid_W, c.lss_d_max)"
Expect: -51.2 51.2 -51.2 51.2 0.4 256 256 90.0
"""
from __future__ import annotations

import shutil
import time

PATH = "src/config.py"

REPLACEMENTS = [
    ("    lss_d_max:    float = 50.0",
     "    lss_d_max:    float = 90.0"),
    ("    bev_x_min: float = -20.0",
     "    bev_x_min: float = -51.2"),
    ("    bev_x_max: float =  20.0",
     "    bev_x_max: float =  51.2"),
    ("    bev_y_min: float = -20.0",
     "    bev_y_min: float = -51.2"),
    ("    bev_y_max: float =  20.0",
     "    bev_y_max: float =  51.2"),
    ("    cell_m:    float =  0.5",
     "    cell_m:    float =  0.4"),
    ("    grid_H:  int = 80",
     "    grid_H:  int = 256"),
    ("    grid_W:  int = 80",
     "    grid_W:  int = 256"),
]


def main():
    with open(PATH, "r") as f:
        src = f.read()

    for old, _ in REPLACEMENTS:
        n = src.count(old)
        if n != 1:
            raise SystemExit(
                f"ABORT: expected exactly 1 occurrence of {old!r}, found {n}. "
                f"File may have changed since the lines were last checked -- "
                f"not patching anything."
            )

    backup_path = f"{PATH}.bak_grid20_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(PATH, backup_path)
    print(f"Backed up current config to {backup_path}")

    for old, new in REPLACEMENTS:
        assert src.count(old) == 1
        src = src.replace(old, new)

    with open(PATH, "w") as f:
        f.write(src)

    print(f"Patched {PATH}:")
    for old, new in REPLACEMENTS:
        print(f"  {old.strip()}  ->  {new.strip()}")
    print("\nNow run py_compile and the Config() verification command in the docstring above.")


if __name__ == "__main__":
    main()