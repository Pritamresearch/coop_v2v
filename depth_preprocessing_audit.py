"""
Comprehensive, CPU-only audit of the dataset/preprocessing pipeline at the
NEW grid size (+/-51.2m, cell_m=0.4, grid=256x256, lss_d_max=90.0), run
against REAL OPV2V data (not synthetic). Purpose: find a concrete, numeric
root cause for why AP50 is still ~0.00-0.02 after ~20-26 epochs on the
corrected grid, rather than continuing to guess.

Checks, each self-contained and printed clearly:
  1. Real vehicle-distance distribution vs [lss_d_min, lss_d_max] coverage.
  2. Depth-bin quantization error (real LiDAR points, real camera extrinsics)
     -- converted into expected BEV positional error vs cell_m.
  3. Camera-axis convention (A vs B), now with real statistical power
     (hundreds of GT boxes, not the n=1 from the earlier check).
  4. Heatmap positive:negative cell density at the new grid size, to sanity
     check whether pos_weight=4.0 is remotely proportionate.
  5. Boundary/indexing check: any real boxes silently dropped or landing on
     an out-of-range cell near the new +/-51.2m edges.

Run on the login node (CPU only, no GPU needed):
    python depth_preprocessing_audit.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import torch

from src.config import Config
from src.train import build_loaders


def _squeeze_to_ndim(x, expected_ndim: int):
    nd = x.dim() if torch.is_tensor(x) else x.ndim
    if nd == expected_ndim + 1:
        x = x[0]
    elif nd != expected_ndim:
        raise ValueError(f"unexpected ndim {nd}, expected {expected_ndim} or {expected_ndim+1}; "
                          f"shape={tuple(x.shape)}")
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def project_point(pt_ego, R, t, K, convention: str):
    pt_cam = R.T @ (pt_ego - t)
    if convention == "A":
        depth = pt_cam[2]
        if depth <= 0.5:
            return None
        u = K[0, 0] * pt_cam[0] / depth + K[0, 2]
        v = K[1, 1] * pt_cam[1] / depth + K[1, 2]
    elif convention == "B":
        depth = pt_cam[0]
        if depth <= 0.5:
            return None
        u = K[0, 0] * pt_cam[1] / depth + K[0, 2]
        v = K[1, 1] * (-pt_cam[2]) / depth + K[1, 2]
    else:
        raise ValueError(convention)
    return u, v, depth


def main():
    cfg = Config()
    cfg.validate()
    print(f"Config: bev=+/-{cfg.bev_x_max}m  cell_m={cfg.cell_m}  grid={cfg.grid_H}x{cfg.grid_W}  "
          f"lss_d_min={cfg.lss_d_min}  lss_d_max={cfg.lss_d_max}  depth_K={cfg.depth_K}")
    bin_width = (cfg.lss_d_max - cfg.lss_d_min) / cfg.depth_K
    print(f"Depth-bin width = ({cfg.lss_d_max}-{cfg.lss_d_min})/{cfg.depth_K} = {bin_width:.4f} m/bin")
    bin_edges = np.linspace(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    print("\nLoading real data via build_loaders(cfg) ...")
    tr, va, te = build_loaders(cfg)
    ds = va.dataset if hasattr(va, "dataset") else va
    IMG_H = getattr(ds, "IMG_H", None)
    IMG_W = getattr(ds, "IMG_W", None)

    n_samples = 60
    it = iter(va)

    all_box_dists = []          # distance of each real GT box from ego (meters)
    all_quant_err_m = []        # |real_depth - nearest_bin_center| for projected GT boxes
    n_vis_A = n_vis_B = n_boxes_total = 0
    pos_cells_total = 0
    neg_cells_total = 0
    n_boundary_dropped = 0
    n_frames = 0

    for _ in range(n_samples):
        try:
            scene = next(it)
        except StopIteration:
            break
        n_frames += 1

        rots = _squeeze_to_ndim(scene["rots"], 4)
        trans = _squeeze_to_ndim(scene["trans"], 3)
        intrins = _squeeze_to_ndim(scene["intrins"], 4)
        gt = scene["gt"]
        boxes = _squeeze_to_ndim(gt["boxes"], 2)
        heat = _squeeze_to_ndim(gt["heat"], 2)   # (H, W) -- assumes single-channel heatmap

        if IMG_H is None:
            imgs = _squeeze_to_ndim(scene["images"], 5)
            IMG_H, IMG_W = imgs.shape[-2], imgs.shape[-1]

        ego_rots = rots[0]
        ego_trans = trans[0]
        ego_intr = intrins[0]

        # heatmap density
        pos_mask = heat > 0.5  # peak cells; use a loose threshold since heat is a soft gaussian
        pos_cells_total += int(pos_mask.sum())
        neg_cells_total += int((~pos_mask).sum())

        for b in range(boxes.shape[0]):
            pt_ego = boxes[b, :3].astype(np.float64)
            n_boxes_total += 1
            dist = float(np.linalg.norm(pt_ego[:2]))  # planar distance from ego, matches bev range semantics
            all_box_dists.append(dist)

            # boundary/indexing check: does this box's (x,y) map inside the declared bev range?
            cx, cy = pt_ego[0], pt_ego[1]
            if not (cfg.bev_x_min <= cx <= cfg.bev_x_max and cfg.bev_y_min <= cy <= cfg.bev_y_max):
                n_boundary_dropped += 1

            vis_a = vis_b = False
            best_depth_for_quant = None
            for cam in range(ego_rots.shape[0]):
                R = ego_rots[cam].astype(np.float64)
                t = ego_trans[cam].astype(np.float64)
                K = ego_intr[cam].astype(np.float64)

                res_a = project_point(pt_ego, R, t, K, "A")
                if res_a is not None:
                    u, v, d = res_a
                    if 0 <= u < IMG_W and 0 <= v < IMG_H:
                        vis_a = True

                res_b = project_point(pt_ego, R, t, K, "B")
                if res_b is not None:
                    u, v, d = res_b
                    if 0 <= u < IMG_W and 0 <= v < IMG_H:
                        vis_b = True
                        if best_depth_for_quant is None or d < best_depth_for_quant:
                            best_depth_for_quant = d

            n_vis_A += int(vis_a)
            n_vis_B += int(vis_b)

            if best_depth_for_quant is not None and cfg.lss_d_min <= best_depth_for_quant <= cfg.lss_d_max:
                nearest_bin = bin_centers[np.argmin(np.abs(bin_centers - best_depth_for_quant))]
                all_quant_err_m.append(abs(best_depth_for_quant - nearest_bin))

    print(f"\nFrames examined: {n_frames}")
    print(f"Total GT boxes examined: {n_boxes_total}")
    if n_boxes_total == 0:
        print("No GT boxes found -- cannot proceed with the rest of the audit.")
        return

    # --- 1. distance distribution vs depth range coverage ---
    dists = np.array(all_box_dists)
    print("\n[1] Real vehicle planar-distance distribution (meters from ego):")
    print(f"    min={dists.min():.1f}  p25={np.percentile(dists,25):.1f}  "
          f"median={np.median(dists):.1f}  p75={np.percentile(dists,75):.1f}  max={dists.max():.1f}")
    below_dmin = int((dists < cfg.lss_d_min).sum())
    above_dmax = int((dists > cfg.lss_d_max).sum())
    print(f"    boxes with distance < lss_d_min({cfg.lss_d_min}): {below_dmin}/{n_boxes_total} "
          f"({100*below_dmin/n_boxes_total:.1f}%)")
    print(f"    boxes with distance > lss_d_max({cfg.lss_d_max}): {above_dmax}/{n_boxes_total} "
          f"({100*above_dmax/n_boxes_total:.1f}%)")

    # --- 2. depth quantization error ---
    print("\n[2] Depth-bin quantization error (real camera-visible GT boxes, convention B):")
    if len(all_quant_err_m) > 0:
        qerr = np.array(all_quant_err_m)
        print(f"    n={len(qerr)}  mean={qerr.mean():.3f}m  median={np.median(qerr):.3f}m  "
              f"max={qerr.max():.3f}m  (bin_width={bin_width:.3f}m, so max possible = {bin_width/2:.3f}m)")
        n_cells = qerr.mean() / cfg.cell_m
        print(f"    mean quantization error as a fraction of one BEV cell ({cfg.cell_m}m): {n_cells:.2f} cells")
        if n_cells > 1.0:
            print("    >>> LARGE relative to cell size -- likely a real, material contributor to localization error.")
        elif n_cells > 0.3:
            print("    >>> Moderate relative to cell size -- a plausible contributing factor, not obviously dominant alone.")
        else:
            print("    >>> Small relative to cell size -- unlikely to be the dominant cause by itself.")
    else:
        print("    No camera-visible, in-range boxes found to measure.")

    # --- 3. camera convention at scale ---
    rate_a = n_vis_A / n_boxes_total
    rate_b = n_vis_B / n_boxes_total
    print(f"\n[3] Camera-axis convention check (n={n_boxes_total}, real statistical power now):")
    print(f"    Convention A (frustum_to_ego-style): {n_vis_A}/{n_boxes_total} visible ({100*rate_a:.1f}%)")
    print(f"    Convention B (CARLA-native, _check_oof-style): {n_vis_B}/{n_boxes_total} visible ({100*rate_b:.1f}%)")
    if rate_a < 0.1 and rate_b > 0.3:
        print("    >>> VERDICT: Convention B correct, Convention A (frustum_to_ego's actual assumption) looks broken.")
        print("    >>> This would mean the LSS backbone's camera->BEV lift is WRONG in every training run --")
        print("    >>> and would fully explain near-zero AP regardless of any other fix.")
    elif rate_b < 0.1 and rate_a > 0.3:
        print("    >>> VERDICT: Convention A correct (frustum_to_ego is fine); B is the one that's off (affects only")
        print("    >>> the oof_recall stat and L_depth supervision, not the main backbone -- less critical).")
    else:
        print("    >>> Still inconclusive or both look reasonable -- see raw rates above.")

    # --- 4. heatmap density ---
    total_cells = pos_cells_total + neg_cells_total
    print(f"\n[4] Heatmap positive:negative cell density at the new {cfg.grid_H}x{cfg.grid_W} grid:")
    print(f"    positive-ish cells (heat>0.5): {pos_cells_total}  negative cells: {neg_cells_total}  "
          f"ratio neg:pos = {neg_cells_total/max(pos_cells_total,1):.1f}:1")
    print(f"    (current gaussian_focal_loss pos_weight=4.0 -- compare informally against this ratio)")

    # --- 5. boundary/indexing ---
    print(f"\n[5] Boundary/indexing sanity: boxes whose (x,y) falls outside the declared bev range "
          f"despite being in the GT list: {n_boundary_dropped}/{n_boxes_total}")
    if n_boundary_dropped > 0:
        print("    >>> Unexpected -- these should have been filtered upstream. Worth a closer look.")
    else:
        print("    >>> Clean -- no boxes found outside the declared range.")


if __name__ == "__main__":
    main()  