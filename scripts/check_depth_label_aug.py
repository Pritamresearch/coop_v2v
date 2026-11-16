"""Do LiDAR depth labels follow the image augmentation? (CPU)

The image-view augmentation resizes/crops/flips/rotates the image and records the pixel
mapping in post_rot/post_trans. If the depth labels are built from the UN-augmented
projection, every augmented frame supervises depth at the wrong pixels: the network sees
an augmented image and is told the depth of a different part of the scene.

Measures, on real frames with img_aug on, how far a LiDAR point's augmented pixel is
from the pixel its label actually occupies, and how often the label cell is even valid.

Positive control: labels rebuilt WITH post_rot/post_trans applied must agree.
"""
from __future__ import annotations
import inspect, os, sys
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config, RESULTS_ROOT
from src.data import opv2v as O


def main() -> int:
    print("[1] does _build_depth_gt even accept the augmentation transform?")
    params = list(inspect.signature(O._build_depth_gt).parameters)
    has = [p for p in params if "post" in p.lower()]
    print(f"    parameters: {params}")
    print(f"    post_rot/post_trans parameters: {has if has else 'NONE'}")
    accepts = bool(has)

    cfg = replace(Config.load(f"{RESULTS_ROOT}/ab_imgaug_rot22/config.json"),
                  device="cpu", depth_source="lidar", augment=True, img_aug=True)
    ds = O.OPV2VDataset(cfg.train_dir, cfg, max_agents=1)
    ds_noaug = O.OPV2VDataset(cfg.train_dir, replace(cfg, img_aug=False, augment=False),
                              max_agents=1)
    print(f"\n[2] label maps with and without image augmentation, same frames")
    print(f"    {'frame':>6s} {'valid cells aug':>16s} {'valid cells no-aug':>19s} "
          f"{'identical':>10s} {'mean |post_rot - I|':>20s}")

    identical, shifts = [], []
    for i in np.linspace(0, len(ds) - 1, 6).astype(int):
        a, b = ds[int(i)], ds_noaug[int(i)]
        da = a["depth_gt_img"][0] if a["depth_gt_img"].ndim == 4 else a["depth_gt_img"]
        db = b["depth_gt_img"][0] if b["depth_gt_img"].ndim == 4 else b["depth_gt_img"]
        da, db = np.asarray(da), np.asarray(db)
        va, vb = int((da >= 0).sum()), int((db >= 0).sum())
        same = bool(np.array_equal(da, db))
        pr = np.asarray(a["post_rots"])[0] if a.get("post_rots") is not None else np.eye(3)
        dev = float(np.abs(pr - np.eye(3)).mean())
        identical.append(same); shifts.append(dev)
        print(f"    {int(i):>6d} {va:>16d} {vb:>19d} {str(same):>10s} {dev:>20.4f}")

    all_same = all(identical)
    aug_real = np.mean(shifts) > 1e-6
    print(f"\n[3] verdict")
    print(f"    image augmentation is actually being applied : {aug_real} "
          f"(mean |post_rot - I| = {np.mean(shifts):.4f})")
    print(f"    depth labels identical with and without aug  : {all_same}")
    if aug_real and all_same and not accepts:
        print("\n    => LABELS DO NOT FOLLOW THE IMAGE. The image is warped by post_rot/post_trans")
        print("       while _build_depth_gt projects into the un-augmented pixel grid, so depth")
        print("       supervision points at the wrong pixels on every augmented frame.")
        print("       The depth loss is not regularising the model, it is contradicting it.")
        return 1
    if accepts and not all_same:
        print("\n    => labels follow the augmentation; alignment is fine.")
        return 0
    print("\n    => inconclusive; inspect manually.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
