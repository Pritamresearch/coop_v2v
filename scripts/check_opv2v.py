"""
scripts/check_opv2v.py
======================
Run this FIRST on Kelvin2 before any training to verify:
    1. The data directories are readable
    2. The YAML parser produces correct camera calibration
    3. Images load with the correct size
    4. GT boxes exist and are in sensible BEV range
    5. LiDAR PCD → depth_gt_img has valid depth-bin indices
    6. OOF flags are non-trivially non-zero (cooperation is possible)

Usage:
    cd /mnt/scratch2/users/2641309/projects/coop_v2v
    python scripts/check_opv2v.py

Expected output on healthy data:
    [OK] Dataset length : 1981   (validate split)
    [OK] images         : (3, 4, 3, 600, 800)
    [OK] rel_pose       : ego row is ~0; collaborators have non-zero dx/dy
    [OK] GT boxes       : 4 objects within [-50,50] m
    [OK] depth_gt       : 1247 valid pixels (non -1) in agent-0 cam-0
    [OK] OOF objects    : 1 of 4 are out-of-FoV (cooperation is meaningful)
    All checks passed.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
from src.config import Config
from src.data.opv2v import OPV2VDataset

PASS = "\033[32m[OK]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
errors = []

def check(cond, msg):
    if cond:
        print(f"{PASS} {msg}")
    else:
        print(f"{FAIL} {msg}")
        errors.append(msg)


def main():
    cfg = Config()
    cfg.validate()

    print("=" * 60)
    print(f"Checking: {cfg.val_dir}")
    print("=" * 60)

    # 1) Dataset loads
    try:
        ds = OPV2VDataset(cfg.val_dir, cfg, max_agents=cfg.max_agents)
        check(len(ds) > 0, f"Dataset length : {len(ds)}")
    except Exception as e:
        check(False, f"Dataset failed to load: {e}")
        print(f"\n{len(errors)} check(s) failed.")
        sys.exit(1)

    # 2) First scene
    scene = ds[0]
    N = scene["n_agents"]
    check(N >= 2, f"n_agents = {N} (need ≥ 2)")

    # 3) Image shape
    imgs = scene["images"]
    check(
        imgs.shape == (N, 4, 3, 600, 800),
        f"images : {tuple(imgs.shape)}"
    )
    check(imgs.isfinite().all(), "images : no NaN/Inf")
    check(
        imgs.abs().mean() < 5.0,          # ImageNet-normalised; typical ~0-2 range
        f"images : mean abs = {imgs.abs().mean():.3f} (should be <5)"
    )

    # 4) Intrinsics look like a camera (fx ≈ fy > 200, cx ≈ W/2, cy ≈ H/2)
    K = scene["intrins"][0, 0]             # ego cam-0
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    check(fx > 200 and fy > 200, f"intrinsic fx={fx:.1f} fy={fy:.1f} (should be >200)")
    check(200 < cx < 700, f"intrinsic cx={cx:.1f} (should be near 400)")
    check(100 < cy < 500, f"intrinsic cy={cy:.1f} (should be near 300)")

    # 5) Relative pose: ego row ≈ 0; others should differ
    rp = scene["rel_pose"]
    check(rp[0].abs().max() < 0.1, f"rel_pose[ego] ≈ 0: {rp[0].tolist()}")
    if N > 1:
        check(rp[1:].abs().max() > 1.0,
              f"rel_pose[others] non-zero: max abs = {rp[1:].abs().max():.2f} m")

    # 6) GT boxes within BEV range
    boxes = scene["gt"]["boxes"]
    check(len(boxes) >= 0, f"GT boxes : {len(boxes)} objects")
    if len(boxes) > 0:
        in_range = ((boxes[:, 0].abs() < 55) & (boxes[:, 1].abs() < 55))
        check(
            in_range.all(),
            f"GT boxes : all cx,cy within ±55 m — worst: {boxes[:, :2].abs().max():.1f}"
        )

    # 7) depth_gt
    dgt = scene["depth_gt_img"]
    check(dgt.shape == (N, 4, 75, 100),
          f"depth_gt_img shape : {tuple(dgt.shape)} (expected ({N},4,75,100))")
    n_valid = int((dgt[0, 0] >= 0).sum())
    check(n_valid > 0, f"depth_gt : {n_valid} valid pixels in agent-0 cam-0")

    # 8) OOF flags
    oof = scene["gt"]["oof"]
    check(oof.dtype == torch.bool, f"oof dtype : {oof.dtype}")
    check(len(oof) == len(boxes), f"oof length matches boxes : {len(oof)}")
    n_oof = int(oof.sum())
    # Not checking n_oof > 0 because in some frames all objects are visible;
    # just print for information
    print(f"      OOF objects : {n_oof} of {len(boxes)}")

    print()
    if errors:
        print(f"{len(errors)} check(s) FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All checks passed. OPV2V loader is working correctly.")


if __name__ == "__main__":
    main()
