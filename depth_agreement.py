#!/usr/bin/env python
"""
depth_agreement.py -- CPU-only. Quantify how far the precomputed ZoeDepth
monocular depth priors are from LiDAR-projected ground truth, in DEPTH BINS.
"""
import argparse
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, ".")
from src.config import Config
from src.data.opv2v import OPV2VDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="frames to compare")
    args = ap.parse_args()

    base = Config()
    bin_m = (base.lss_d_max - base.lss_d_min) / base.depth_K
    print("=" * 74)
    print("ZoeDepth prior vs LiDAR depth GT -- bin-level agreement")
    print("=" * 74)
    print(f"depth_K={base.depth_K}  range={base.lss_d_min}-{base.lss_d_max} m  "
          f"bin width={bin_m:.3f} m")
    print(f"frames={args.n}\n")

    cfg_lidar = replace(base, depth_source="lidar")
    cfg_zoe = replace(base, depth_source="zoedepth")

    ds_l = OPV2VDataset(cfg_lidar.train_dir, cfg_lidar,
                        max_agents=cfg_lidar.max_agents,
                        use_gt_pose=cfg_lidar.use_gt_pose)
    ds_z = OPV2VDataset(cfg_zoe.train_dir, cfg_zoe,
                        max_agents=cfg_zoe.max_agents,
                        use_gt_pose=cfg_zoe.use_gt_pose)

    n = min(args.n, len(ds_l), len(ds_z))
    pairs = []
    n_both_valid_total = 0
    n_lidar_valid_total = 0
    n_zoe_valid_total = 0
    n_px_total = 0

    for i in range(n):
        try:
            sl = ds_l[i]
            sz = ds_z[i]
        except Exception as e:
            print(f"frame {i}: load failed ({type(e).__name__}: {e})")
            continue

        dl = np.asarray(sl["depth_gt_img"])
        dz = np.asarray(sz["depth_gt_img"])
        if dl.shape != dz.shape:
            print(f"frame {i}: shape mismatch {dl.shape} vs {dz.shape}, skipping")
            continue

        n_px_total += dl.size
        vl = dl >= 0
        vz = dz >= 0
        both = vl & vz
        n_lidar_valid_total += int(vl.sum())
        n_zoe_valid_total += int(vz.sum())
        n_both_valid_total += int(both.sum())

        if both.any():
            pairs.append((dl[both].astype(np.int64), dz[both].astype(np.int64)))

    if not pairs:
        print("No pixels where BOTH sources are valid. Cannot compare.")
        print(f"  lidar-valid px: {n_lidar_valid_total}")
        print(f"  zoe-valid px:   {n_zoe_valid_total}")
        print("  -> check that .pcd files and *_depth_prior.npy files both exist.")
        return

    lid = np.concatenate([p[0] for p in pairs])
    zoe = np.concatenate([p[1] for p in pairs])
    d = zoe - lid
    ad = np.abs(d)

    print(f"frames compared: {len(pairs)}")
    print(f"pixels total:            {n_px_total}")
    print(f"  LiDAR-valid:           {n_lidar_valid_total} "
          f"({100.0*n_lidar_valid_total/max(n_px_total,1):.2f}%)")
    print(f"  ZoeDepth-valid:        {n_zoe_valid_total} "
          f"({100.0*n_zoe_valid_total/max(n_px_total,1):.2f}%)")
    print(f"  both valid (compared): {n_both_valid_total} "
          f"({100.0*n_both_valid_total/max(n_px_total,1):.2f}%)\n")

    print("SIGNED bin error (zoe - lidar); positive = Zoe predicts FARTHER")
    print(f"  mean   {d.mean():+.2f} bins  ({d.mean()*bin_m:+.2f} m)")
    print(f"  median {np.median(d):+.2f} bins  ({np.median(d)*bin_m:+.2f} m)\n")

    print("ABSOLUTE bin error")
    for q in (50, 75, 90, 95, 99):
        v = np.percentile(ad, q)
        print(f"  p{q:<3d} {v:7.2f} bins  ({v*bin_m:6.2f} m)")
    print(f"  mean {ad.mean():7.2f} bins  ({ad.mean()*bin_m:6.2f} m)\n")

    for thr in (1, 2, 4, 8):
        frac = float((ad > thr).mean())
        print(f"  fraction off by more than {thr:2d} bins "
              f"({thr*bin_m:5.2f} m): {100.0*frac:5.1f}%")

    print("\nError vs true distance (binned by LiDAR depth bin):")
    ranges = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 48), (48, 64)]
    print(f"  {'lidar bin':>12s} {'approx m':>14s} {'n px':>10s} "
          f"{'mean |err| bins':>16s} {'m':>8s}")
    for (lo, hi) in ranges:
        m = (lid >= lo) & (lid < hi)
        if m.any():
            e = ad[m]
            lo_m = base.lss_d_min + lo * bin_m
            hi_m = base.lss_d_min + hi * bin_m
            print(f"  {f'{lo}-{hi}':>12s} {f'{lo_m:.1f}-{hi_m:.1f}':>14s} "
                  f"{e.size:>10d} {e.mean():>16.2f} {e.mean()*bin_m:>8.2f}")


if __name__ == "__main__":
    main()
