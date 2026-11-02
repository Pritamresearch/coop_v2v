"""
Checks whether the current BEV grid (bev_x/y = +/-20m) is the reason most
real validation frames show zero GT boxes -- found empirically via
check_camera_convention.py (19/20 real frames had 0 GT boxes; 1 had exactly 1).

bev_augment.py's own docstring says: "the ego vehicle almost always follows
a lead vehicle at ~30m forward" -- which is already outside a +/-20m grid.
This script builds a SECOND loader with a widened grid (+/-60m) and compares
GT-box counts per frame against the current (+/-20m) grid, on the same
underlying frames (validation loader is not shuffled, so frame order should
match between the two loaders).

Run on the login node:
    python check_grid_range.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from dataclasses import replace

import torch

from src.config import Config
from src.train import build_loaders


def _squeeze_to_ndim(x, expected_ndim: int):
    nd = x.dim() if torch.is_tensor(x) else x.ndim
    if nd == expected_ndim + 1:
        x = x[0]
    elif nd != expected_ndim:
        raise ValueError(f"unexpected ndim {nd}, expected {expected_ndim} or {expected_ndim+1}")
    return x


def box_counts(cfg, n_samples=20, label=""):
    print(f"\nBuilding loader for {label}: "
          f"bev_x=[{cfg.bev_x_min},{cfg.bev_x_max}] bev_y=[{cfg.bev_y_min},{cfg.bev_y_max}] "
          f"grid={cfg.grid_H}x{cfg.grid_W} cell_m={cfg.cell_m}")
    tr, va, te = build_loaders(cfg)
    it = iter(va)
    counts = []
    for i in range(n_samples):
        try:
            scene = next(it)
        except StopIteration:
            break
        boxes = _squeeze_to_ndim(scene["gt"]["boxes"], 2)
        n = boxes.shape[0]
        counts.append(n)
    return counts


def main():
    cfg_narrow = Config()
    cfg_narrow.validate()

    cfg_wide = replace(cfg_narrow,
                        bev_x_min=-60.0, bev_x_max=60.0,
                        bev_y_min=-60.0, bev_y_max=60.0,
                        grid_H=240, grid_W=240)
    cfg_wide.validate()

    n_samples = 20
    counts_narrow = box_counts(cfg_narrow, n_samples, "CURRENT grid (+/-20m)")
    print(f"  per-frame box counts: {counts_narrow}")
    print(f"  total={sum(counts_narrow)}  mean={sum(counts_narrow)/len(counts_narrow):.2f}  "
          f"frames_with_zero={sum(1 for c in counts_narrow if c == 0)}/{len(counts_narrow)}")

    counts_wide = box_counts(cfg_wide, n_samples, "WIDENED grid (+/-60m, diagnostic only)")
    print(f"  per-frame box counts: {counts_wide}")
    print(f"  total={sum(counts_wide)}  mean={sum(counts_wide)/len(counts_wide):.2f}  "
          f"frames_with_zero={sum(1 for c in counts_wide if c == 0)}/{len(counts_wide)}")

    print("\n" + "=" * 70)
    ratio = (sum(counts_wide) / max(sum(counts_narrow), 1))
    print(f"Widened-grid total boxes / current-grid total boxes = {ratio:.1f}x")
    if sum(counts_narrow) <= 2 and sum(counts_wide) > 10 * max(sum(counts_narrow), 1):
        print("CONFIRMED: the current +/-20m BEV grid is excluding the large majority of real")
        print("GT vehicles. This is a strong, direct, likely-dominant explanation for the low")
        print("AP50/precision numbers -- a model trained and evaluated against near-empty")
        print("targets in ~95% of frames cannot perform well regardless of any other fix.")
        print("This is independent of, and more consequential than, the NMS/pos_weight fixes")
        print("or the camera-convention question investigated separately.")
    else:
        print("The widened grid did not recover dramatically more boxes than expected, or the")
        print("current grid already captures a reasonable fraction -- do not assume the grid")
        print("size is the dominant issue without this clear a signal. Report both count lists")
        print("for a closer look.")
    print("=" * 70)


if __name__ == "__main__":
    main()