"""
Fixes the camera-axis convention bug in frustum_to_ego (src/models/lss_backbone.py).

Confirmed 2026-10-27 against real OPV2V data (n=791 real GT boxes, 60 real
validation frames): the standard-pinhole un-projection this function used
(K_inv @ [u*d, v*d, d], giving camera-local axes X=right, Y=down, Z=forward)
gives 0.0% visibility when checked against real camera geometry -- completely
degenerate. The actual convention this dataset's rots/trans/intrins matrices
use is CARLA-native (x=forward/depth, y=right, z=up), matching src/data/
opv2v.py's _check_oof/_build_depth_gt, which gave 97.5% visibility on the
same real data.

This means the LSS backbone's camera->BEV lift has been placing every
camera feature at the wrong location in BEV space, in every forward pass,
in every training run this project has done. It is the leading suspect for
why AP50 stayed near-zero even after the BEV-grid-size fix (2026-10-24).

Fix: after the existing K_inv un-projection (unchanged), permute the result
from standard-pinhole axes into this dataset's real camera-local axes:
    new_x (depth) = old_Z
    new_y (right) = old_X
    new_z (up)    = -old_Y
This is an EXACT algebraic inverse of _check_oof's encode-direction formula
(verified by hand: see claude/diagnosis_2026-10-27_frustum_convention_fix.md).

Exact-match patch with assert-count==1 guard, same pattern as prior fixes
this session. Makes a timestamped backup first.

Run from the repo root on kelvin2:
    python patch_frustum_convention.py
Then verify with verify_frustum_fix.py (real-data round trip) BEFORE
resubmitting any training job.
"""
from __future__ import annotations

import shutil
import time

PATH = "src/models/lss_backbone.py"

OLD = '''    # un-project via K^{-1}: (u*d, v*d, d) -> cam 3D
    d       = pts[..., 2:3]
    uv1     = torch.cat([pts[..., :2] * d, d], dim=-1)
    K_inv   = torch.inverse(intrins)
    pts_cam = torch.einsum("nij,nkhwj->nkhwi", K_inv, uv1)

    # cam -> ego
    pts_ego = (torch.einsum("nij,nkhwj->nkhwi", rots, pts_cam)
               + trans.view(n_cams, 1, 1, 1, 3))
    return pts_ego'''

NEW = '''    # un-project via K^{-1}: (u*d, v*d, d) -> cam 3D (standard-pinhole axes:
    # X=right, Y=down, Z=forward/depth -- NOT this dataset's real convention,
    # see permutation below)
    d       = pts[..., 2:3]
    uv1     = torch.cat([pts[..., :2] * d, d], dim=-1)
    K_inv   = torch.inverse(intrins)
    pts_cam_std = torch.einsum("nij,nkhwj->nkhwi", K_inv, uv1)

    # Permute into this dataset's REAL camera-local convention (confirmed
    # empirically against real OPV2V data, 2026-10-27, n=791 GT boxes:
    # CARLA-native, x=forward/depth, y=right, z=up -- matches
    # _check_oof/_build_depth_gt in src/data/opv2v.py). The un-projection
    # above is algebraically correct for a standard pinhole camera; this
    # dataset's rots/trans/intrins just don't use that axis layout.
    #   new_x (depth) = std Z
    #   new_y (right) = std X
    #   new_z (up)    = -std Y
    pts_cam = torch.stack(
        [pts_cam_std[..., 2], pts_cam_std[..., 0], -pts_cam_std[..., 1]],
        dim=-1,
    )

    # cam -> ego
    pts_ego = (torch.einsum("nij,nkhwj->nkhwi", rots, pts_cam)
               + trans.view(n_cams, 1, 1, 1, 3))
    return pts_ego'''


def main():
    with open(PATH, "r") as f:
        src = f.read()

    n = src.count(OLD)
    if n != 1:
        raise SystemExit(
            f"ABORT: expected exactly 1 occurrence of the target block, found {n}. "
            f"File may have changed since this patch was written -- not patching anything."
        )

    backup_path = f"{PATH}.bak_stdpinhole_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(PATH, backup_path)
    print(f"Backed up current file to {backup_path}")

    src = src.replace(OLD, NEW)
    with open(PATH, "w") as f:
        f.write(src)

    print(f"Patched {PATH}: frustum_to_ego now permutes into the real camera-axis convention.")
    print("Next: run verify_frustum_fix.py (real-data round trip) before resubmitting training.")


if __name__ == "__main__":
    main()