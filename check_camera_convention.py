"""
Empirical check: which camera-frame axis convention is actually correct
for this project's real OPV2V data -- the one frustum_to_ego assumes
(z=depth, standard pinhole, NO axis permutation applied to rots/trans),
or the one _check_oof/_build_depth_gt assume (x=depth, matching CARLA's
documented raw sensor convention with an explicit y,-z,x permutation).

These two are NOT both correct -- they disagree on what rots/trans (from
_cam_extr_to_rot_trans, a raw inversion of OPV2V's stored extrinsic) mean.
If convention B (CARLA-native, used by _check_oof/_build_depth_gt) is the
real one, then frustum_to_ego -- the ACTUAL neural-network forward path
used in every training run -- is lifting every camera's features into the
wrong BEV location, systematically, for every frame. This would be a much
larger finding than anything found so far this session.

Method: for a batch of REAL frames, project every real GT box centre into
every one of the ego's 4 real cameras under BOTH conventions, and check
which one puts a sensible fraction of real, physically-visible vehicles
in-bounds. The correct convention should give a plausible visibility rate
(OPV2V GT is the union of all CAVs' visible vehicles -- ego genuinely
can't see all of it, so expect well under 100%, but a real, non-degenerate
rate). The wrong convention should look close to degenerate/near-zero or
otherwise clearly senseless, since a true axis-convention mismatch
effectively scrambles where points land.

Run on the login node (CPU is fine, this only touches the dataloader,
no model forward pass):
    python check_camera_convention.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import torch

from src.config import Config
from src.train import build_loaders


def _squeeze_to_ndim(x, expected_ndim: int):
    """Drop a leading batch dim ONLY if actual ndim is exactly one more than
    the documented per-sample ndim. Does NOT guess from shape[0]==1 -- that
    heuristic is wrong whenever a per-sample dim (e.g. a variable number of
    GT boxes) can itself legitimately be 1."""
    nd = x.dim() if torch.is_tensor(x) else x.ndim
    if nd == expected_ndim + 1:
        x = x[0]
    elif nd != expected_ndim:
        raise ValueError(f"unexpected ndim {nd}, expected {expected_ndim} or {expected_ndim+1}; "
                          f"shape={tuple(x.shape)}")
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


def project_point(pt_ego, R, t, K, convention: str):
    """
    pt_ego : (3,) point in ego/LiDAR frame
    R      : (3,3) cam->ego rotation (raw, from _cam_extr_to_rot_trans)
    t      : (3,)  cam origin in ego frame
    K      : (3,3) camera intrinsics
    convention: 'A' (frustum_to_ego-style, z=depth, no permutation)
                'B' (CARLA-native, x=depth, y,-z,x permutation)
    Returns (u, v, depth) -- depth <= 0 means behind camera under that convention.
    """
    pt_cam = R.T @ (pt_ego - t)   # raw cam-local point, same for both conventions
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

    print(f"Current grid: bev_x=[{cfg.bev_x_min},{cfg.bev_x_max}] "
          f"bev_y=[{cfg.bev_y_min},{cfg.bev_y_max}]  grid={cfg.grid_H}x{cfg.grid_W}  "
          f"cell_m={cfg.cell_m}")
    print("Loading real data via build_loaders(cfg) ...")
    tr, va, te = build_loaders(cfg)

    n_samples = 20
    # read the real image resolution from the dataset object (same IMG_H/IMG_W
    # _check_oof itself uses), falling back to the images tensor's own shape
    # if the attribute isn't exposed -- not cfg.lss_img_H/W, which is the LSS
    # encoder's resized input and may differ from the actual loaded resolution.
    ds = va.dataset if hasattr(va, "dataset") else va
    IMG_H = getattr(ds, "IMG_H", None)
    IMG_W = getattr(ds, "IMG_W", None)

    seen = 0
    n_boxes_total = 0
    n_vis_A = 0
    n_vis_B = 0

    it = iter(va)
    for _ in range(n_samples):
        try:
            scene = next(it)
        except StopIteration:
            break
        seen += 1

        # documented per-sample ndims: rots (N,4,3,3)=4, trans (N,4,3)=3,
        # intrins (N,4,3,3)=4, boxes (M,7)=2, images (N,4,3,H,W)=5
        rots = _squeeze_to_ndim(scene["rots"], 4)
        trans = _squeeze_to_ndim(scene["trans"], 3)
        intrins = _squeeze_to_ndim(scene["intrins"], 4)
        gt = scene["gt"]
        boxes = _squeeze_to_ndim(gt["boxes"], 2)

        if IMG_H is None:
            imgs = _squeeze_to_ndim(scene["images"], 5)
            IMG_H, IMG_W = imgs.shape[-2], imgs.shape[-1]

        print(f"  sample {seen}: rots.shape={rots.shape} boxes.shape={boxes.shape} "
              f"(n_boxes_this_frame={boxes.shape[0]})")

        # ego = agent 0 (N is the first dim of rots/trans/intrins after squeeze)
        ego_rots = rots[0]      # (4,3,3)
        ego_trans = trans[0]    # (4,3)
        ego_intr = intrins[0]   # (4,3,3)

        for b in range(boxes.shape[0]):
            pt_ego = boxes[b, :3].astype(np.float64)
            n_boxes_total += 1

            vis_a = False
            vis_b = False
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

            n_vis_A += int(vis_a)
            n_vis_B += int(vis_b)

    print(f"\nSamples examined: {seen}")
    print(f"Total GT boxes examined: {n_boxes_total}")
    if n_boxes_total == 0:
        print("No GT boxes found in the examined samples -- cannot conclude anything. "
              "Try increasing n_samples or check the loader is returning real data.")
        return

    rate_a = n_vis_A / n_boxes_total
    rate_b = n_vis_B / n_boxes_total
    print(f"\nConvention A (frustum_to_ego-style, z=depth, NO permutation):")
    print(f"  {n_vis_A}/{n_boxes_total} GT boxes visible in >=1 of ego's 4 cameras "
          f"({rate_a*100:.1f}%)")
    print(f"\nConvention B (CARLA-native, x=depth, y/-z/x permutation, matches _check_oof):")
    print(f"  {n_vis_B}/{n_boxes_total} GT boxes visible in >=1 of ego's 4 cameras "
          f"({rate_b*100:.1f}%)")

    print("\n" + "=" * 70)
    if rate_a < 0.05 and rate_b > 0.15:
        print("VERDICT: Convention B (CARLA-native / _check_oof's convention) looks correct.")
        print("Convention A (frustum_to_ego's assumption) looks broken -- near-zero visibility")
        print("is what a genuine axis-convention mismatch produces. This would mean the LSS")
        print("backbone's camera->BEV lifting (frustum_to_ego) is using the wrong convention")
        print("in every training run.")
    elif rate_b < 0.05 and rate_a > 0.15:
        print("VERDICT: Convention A (frustum_to_ego's assumption) looks correct.")
        print("Convention B (_check_oof/_build_depth_gt's assumption) looks broken instead --")
        print("those two functions would be the ones with the wrong convention (affecting the")
        print("out-of-FoV recall stat and the depth-supervision loss, not the main backbone).")
    elif rate_a > 0.15 and rate_b > 0.15:
        print("INCONCLUSIVE: both conventions give plausible-looking visibility rates. This")
        print("could mean the camera layout is close to symmetric in a way that doesn't")
        print("discriminate well, or something about this test's assumptions is off. Do not")
        print("conclude either way from this alone -- report both numbers back for a next step")
        print("(e.g. checking a handful of individual cases by hand, or visualizing one frame).")
    else:
        print("INCONCLUSIVE: both rates are low/degenerate. Something else may be off in this")
        print("test (e.g. a shape assumption about rots/trans/intrins/boxes) rather than a")
        print("clean confirmation either way. Report both numbers and the raw shapes printed")
        print("above before concluding anything.")
    print("=" * 70)


if __name__ == "__main__":
    main()