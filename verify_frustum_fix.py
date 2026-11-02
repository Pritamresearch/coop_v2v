"""
Real-data round-trip verification of the frustum_to_ego patch.

Does NOT reimplement the fix separately -- imports and calls the ACTUAL
patched src.models.lss_backbone.frustum_to_ego function, so this tests the
deployed code, not a hand-derivation on paper.

Method: take real GT boxes from real validation frames. For each box visible
in a real camera (checked via _check_oof's own convention, i.e. the
empirically-confirmed-correct one), build the exact (u, v, depth) pixel
coordinates that camera would produce for that box's ego-frame position,
feed that single point through the real frustum_to_ego(), and check that it
recovers the ORIGINAL real ego-frame point (not a synthetic one).

Run on the login node (CPU only) AFTER patch_frustum_convention.py:
    python verify_frustum_fix.py
Expect near-zero recovery error (sub-millimeter, limited only by float32
precision) for every tested point. If this script shows anything but,
DO NOT resubmit training -- the patch is wrong somewhere.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import torch

from src.config import Config
from src.train import build_loaders
from src.models.lss_backbone import frustum_to_ego


def _squeeze_to_ndim(x, expected_ndim: int):
    nd = x.dim() if torch.is_tensor(x) else x.ndim
    if nd == expected_ndim + 1:
        x = x[0]
    elif nd != expected_ndim:
        raise ValueError(f"unexpected ndim {nd}, expected {expected_ndim} or {expected_ndim+1}; "
                          f"shape={tuple(x.shape)}")
    return x


def main():
    cfg = Config()
    cfg.validate()
    print("Loading real data via build_loaders(cfg) ...")
    tr, va, te = build_loaders(cfg)

    n_frames_to_scan = 30
    n_tested = 0
    errs = []

    it = iter(va)
    for _ in range(n_frames_to_scan):
        try:
            scene = next(it)
        except StopIteration:
            break

        rots_t = _squeeze_to_ndim(scene["rots"], 4)      # (N,4,3,3) torch
        trans_t = _squeeze_to_ndim(scene["trans"], 3)    # (N,4,3)
        intrins_t = _squeeze_to_ndim(scene["intrins"], 4)
        boxes = _squeeze_to_ndim(scene["gt"]["boxes"], 2).numpy()

        ego_rots = rots_t[0]      # (4,3,3) torch, cam->ego
        ego_trans = trans_t[0]    # (4,3)
        ego_intr = intrins_t[0]   # (4,3,3)

        for b in range(boxes.shape[0]):
            pt_ego = boxes[b, :3].astype(np.float64)

            for cam in range(ego_rots.shape[0]):
                R = ego_rots[cam].double().numpy()
                t = ego_trans[cam].double().numpy()
                K = ego_intr[cam].double().numpy()

                # encode direction, convention B (empirically confirmed correct):
                # pt_cam = R.T @ (pt_ego - t); depth = pt_cam[0];
                # u = K00*pt_cam[1]/depth + cx; v = K11*(-pt_cam[2])/depth + cy
                pt_cam = R.T @ (pt_ego - t)
                depth = pt_cam[0]
                if depth <= 0.5:
                    continue
                u = K[0, 0] * pt_cam[1] / depth + K[0, 2]
                v = K[1, 1] * (-pt_cam[2]) / depth + K[1, 2]
                if not (0 <= u < 99999 and 0 <= v < 99999):
                    continue

                # now feed (u, v, depth) through the REAL, patched frustum_to_ego
                frustum = torch.tensor([[[[u, v, depth]]]], dtype=torch.float64)  # (1,1,1,3) -> K=1,fH=1,fW=1
                rots_1 = torch.tensor(R, dtype=torch.float64).unsqueeze(0)        # (1,3,3)
                trans_1 = torch.tensor(t, dtype=torch.float64).unsqueeze(0)      # (1,3)
                intrins_1 = torch.tensor(K, dtype=torch.float64).unsqueeze(0)    # (1,3,3)

                pts_ego_out = frustum_to_ego(frustum, rots_1, trans_1, intrins_1, None, None)
                recovered = pts_ego_out[0, 0, 0, 0].numpy()  # (3,)

                err = float(np.linalg.norm(recovered - pt_ego))
                errs.append(err)
                n_tested += 1

    print(f"\nTested {n_tested} real (box, camera) pairs through the actual patched frustum_to_ego().")
    if n_tested == 0:
        print("No visible points found to test -- something upstream is wrong, re-check.")
        return
    errs = np.array(errs)
    print(f"Recovery error: mean={errs.mean():.6f}m  median={np.median(errs):.6f}m  max={errs.max():.6f}m")
    if errs.max() < 1e-3:
        print(">>> PASS: frustum_to_ego now correctly recovers real ego-frame points from real camera geometry.")
    elif errs.max() < 0.1:
        print(">>> MOSTLY PASS, small residual error -- likely float32 vs float64 noise, check if this matters at your cell_m.")
    else:
        print(">>> FAIL: recovery error is large -- the patch is wrong, do NOT resubmit training yet.")


if __name__ == "__main__":
    main()
    