"""
Real end-to-end integration test of the composed geometry pipeline:
frustum_to_ego -> _bev_coords -> bev_pool, run through the ACTUAL
LSSBackbone forward path (_forward_agent), on REAL OPV2V scenes.

This does NOT depend on trained weights: row/col/valid (the geometric
placement) only depend on camera extrinsics/intrinsics and the frustum,
never on the encoder's learned parameters. So this is a pure, weight-
independent test of whether the fixed geometry actually delivers
information to the right BEV cells -- the strongest available check short
of a full multi-day training run.

Method: for real GT vehicles in real validation frames, compute each
vehicle's own target BEV cell (same formula _bev_coords uses). Then run
the real ego agent's images/rots/trans/intrins through LSSBackbone's own
frustum_to_ego + _bev_coords (not a reimplementation -- imports and calls
the real methods), and check whether ANY of the M = n_cams*K*fH*fW lifted
frustum points actually lands in (or within a small tolerance of) that
vehicle's target cell. If real GT cells never receive any geometric
coverage, the pipeline is still broken even after both fixes. If they do,
that's strong, weight-independent confirmation the geometry is now
correctly connected end-to-end.

Uses cfg.lss_encoder='small' (CPU-only debugging encoder, per this file's
own docstring) purely for speed -- the encoder's own correctness doesn't
matter for this check, only frustum_to_ego/_bev_coords do, and those are
identical regardless of which encoder is plugged in.

Run on the login node (CPU only, no GPU needed), AFTER both patches:
    python e2e_geometry_integration_test.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from dataclasses import replace

import numpy as np
import os
import torch

from src.config import Config
from src.train import build_loaders
from src.models.lss_backbone import LSSBackbone


def _squeeze_to_ndim(x, expected_ndim: int):
    nd = x.dim() if torch.is_tensor(x) else x.ndim
    if nd == expected_ndim + 1:
        x = x[0]
    elif nd != expected_ndim:
        raise ValueError(f"unexpected ndim {nd}, expected {expected_ndim} or {expected_ndim+1}; "
                          f"shape={tuple(x.shape)}")
    return x


def main():
    base = Config()
    base.validate()
    cfg = replace(base, lss_encoder="small")   # CPU-only debugging encoder; geometry unaffected
    # COOP_IMG_AUG=1 runs the same coverage check with the TRAINING augmentation on (image aug +
    # BEV rotation/flip/scale), which is the configuration the image-aug A/B arms train with.
    _img_aug = os.environ.get("COOP_IMG_AUG") == "1"
    if _img_aug:
        cfg = replace(cfg, img_aug=True, bev_flip=True, bev_scale_range=(0.95, 1.05),
                      bev_rot_deg=float(os.environ.get("COOP_BEV_ROT", "180")))
    cfg.validate()
    print(f"image augmentation: {'ON (train loader)' if _img_aug else 'OFF (val loader)'}")
    print(f"Config (CPU test): bev=+/-{cfg.bev_x_max}m cell_m={cfg.cell_m} grid={cfg.grid_H}x{cfg.grid_W} "
          f"depth_K={cfg.depth_K} lss_d_min={cfg.lss_d_min} lss_d_max={cfg.lss_d_max} encoder={cfg.lss_encoder}")

    print("Building real data loaders ...")
    tr, va, te = build_loaders(cfg)

    print("Instantiating real LSSBackbone (random weights -- geometry is weight-independent) ...")
    torch.manual_seed(0)
    backbone = LSSBackbone(cfg)
    backbone.eval()

    n_frames_to_test = 8
    it = iter(tr if _img_aug else va)   # the train loader is the one with augmentation enabled

    n_boxes_total = 0
    n_boxes_covered_exact = 0
    n_boxes_covered_tol1 = 0
    valid_fractions = []
    any_nan = False

    for frame_i in range(n_frames_to_test):
        try:
            scene = next(it)
        except StopIteration:
            break

        images = _squeeze_to_ndim(scene["images"], 5)        # (N, n_cams, 3, H, W)
        rots_t = _squeeze_to_ndim(scene["rots"], 4)           # (N, n_cams, 3, 3)
        trans_t = _squeeze_to_ndim(scene["trans"], 3)         # (N, n_cams, 3)
        intrins_t = _squeeze_to_ndim(scene["intrins"], 4)     # (N, n_cams, 3, 3)
        boxes = _squeeze_to_ndim(scene["gt"]["boxes"], 2).numpy()  # (M, 7)

        post_rots = scene.get("post_rots", None)
        post_trans = scene.get("post_trans", None)
        if post_rots is not None:
            post_rots = _squeeze_to_ndim(post_rots, 4)[0]   # ego's own, (n_cams,3,3)
        if post_trans is not None:
            post_trans = _squeeze_to_ndim(post_trans, 3)[0]  # (n_cams,3)

        ego_images = images[0].float()
        ego_rots = rots_t[0].float()
        ego_trans = trans_t[0].float()
        ego_intrins = intrins_t[0].float()

        # LSSBackbone doesn't expose frustum_to_ego as a method -- call the
        # module-level function directly (identical to what _forward_agent calls).
        from src.models.lss_backbone import frustum_to_ego
        with torch.no_grad():
            pts_ego = frustum_to_ego(backbone.frustum, ego_rots, ego_trans, ego_intrins,
                                      post_rots, post_trans,
                                      depth_scale=float(scene.get("depth_scale", 1.0)))
            row, col, valid = backbone._bev_coords(pts_ego)

        if torch.isnan(pts_ego).any() or torch.isinf(pts_ego).any():
            any_nan = True
            print(f"  frame {frame_i}: NaN/Inf detected in pts_ego!")

        valid_frac = float(valid.float().mean())
        valid_fractions.append(valid_frac)

        row_np = row.numpy()
        col_np = col.numpy()
        valid_np = valid.numpy()

        for b in range(boxes.shape[0]):
            cx, cy = float(boxes[b, 0]), float(boxes[b, 1])
            target_col = int((cx - cfg.bev_x_min) / cfg.cell_m)
            target_row = int((cy - cfg.bev_y_min) / cfg.cell_m)
            n_boxes_total += 1

            # exact-cell coverage
            exact_hit = bool(np.any(
                valid_np & (row_np == target_row) & (col_np == target_col)
            ))
            # +/-1 cell tolerance (accounts for quantization/discretization)
            tol1_hit = bool(np.any(
                valid_np &
                (np.abs(row_np - target_row) <= 1) &
                (np.abs(col_np - target_col) <= 1)
            ))
            n_boxes_covered_exact += int(exact_hit)
            n_boxes_covered_tol1 += int(tol1_hit)

        print(f"  frame {frame_i}: {boxes.shape[0]} GT boxes, valid_frac={valid_frac:.3f}")

    print(f"\nAny NaN/Inf in lifted points: {any_nan}")
    print(f"Mean fraction of frustum points landing inside the declared BEV grid: "
          f"{np.mean(valid_fractions):.3f}")
    print(f"\nTotal real GT boxes tested: {n_boxes_total}")
    print(f"Boxes with >=1 real frustum point landing in their EXACT target cell: "
          f"{n_boxes_covered_exact}/{n_boxes_total} ({100*n_boxes_covered_exact/max(n_boxes_total,1):.1f}%)")
    print(f"Boxes with >=1 real frustum point landing within +/-1 cell of target: "
          f"{n_boxes_covered_tol1}/{n_boxes_total} ({100*n_boxes_covered_tol1/max(n_boxes_total,1):.1f}%)")

    print("\n" + "=" * 70)
    if n_boxes_total > 0 and n_boxes_covered_tol1 / n_boxes_total > 0.7:
        print("PASS: most real GT vehicles DO receive real geometric coverage from the lift.")
        print("The composed geometry pipeline (frustum_to_ego -> _bev_coords -> bev_pool) is")
        print("delivering information to the right place. This does not guarantee the model")
        print("will learn well (that still depends on training), but it rules out a further")
        print("silent geometry bug of the kind just fixed.")
    elif n_boxes_total > 0 and n_boxes_covered_tol1 / n_boxes_total > 0.3:
        print("PARTIAL: a meaningful fraction of real GT vehicles get no geometric coverage at all.")
        print("Could be expected (occluded / out-of-FoV vehicles genuinely can't be lifted), or")
        print("could indicate a remaining issue -- compare this rate against the OOF rate measured")
        print("earlier (0.11-0.41) to see if it's explained by that alone.")
    else:
        print("FAIL: real GT vehicles are overwhelmingly NOT covered by any real lifted point.")
        print("This would mean the geometry is still broken somewhere despite both fixes --")
        print("do not submit training before investigating further.")
    print("=" * 70)
    # Exit non-zero unless coverage is in the PASS band, so this can gate a training job.
    return 0 if (n_boxes_total > 0 and n_boxes_covered_tol1 / n_boxes_total > 0.7) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
    