"""Where the augmentations plug into an LSS forward pass (torch).

Standard LSS (Philion & Fidler 2020) `get_geometry`:
    points = frustum - post_trans ; points = inv(post_rots) @ points   # undo IDA
    points = [x*d, y*d, d] ; points = rots @ inv(K) @ points + trans    # cam -> ego
Your dataset already returns post_rots/post_trans or rescaled K. With
coper_diag.augment.ida_matrix(M):
    post_rots  = M[:2,:2] embedded in 3x3, post_trans = [M[0,2], M[1,2], 0]
and K stays the ORIGINAL intrinsics (LSS undoes M first). Alternatively pass
K' = M @ K and identity post-transforms; both are exact.

BDA is then one line after get_geometry, before voxel pooling:
    geom = bda_geometry(geom, A)       # A from bda_matrix, per sample
and the SAME A is applied to the sample's GT boxes in the dataset
(augment.bda_apply_boxes), and for cooperation to every collaborator->ego
pose (T' = A @ T) so warped collaborator features stay aligned.
At eval time A = identity.
"""
from __future__ import annotations

import numpy as np
import torch


def ida_to_post(M: np.ndarray):
    """3x3 pixel affine -> LSS (post_rots 3x3, post_trans 3)."""
    pr = torch.eye(3, dtype=torch.float32)
    pr[:2, :2] = torch.as_tensor(M[:2, :2], dtype=torch.float32)
    pt = torch.zeros(3, dtype=torch.float32)
    pt[:2] = torch.as_tensor(M[:2, 2], dtype=torch.float32)
    return pr, pt


def bda_geometry(geom: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """geom: (B, N, D, H, W, 3) ego-frame frustum points; A: (B, 4, 4)."""
    B = geom.shape[0]
    R = A[:, :3, :3].to(geom).view(B, 1, 1, 1, 1, 3, 3)
    t = A[:, :3, 3].to(geom).view(B, 1, 1, 1, 1, 3)
    return (R @ geom.unsqueeze(-1)).squeeze(-1) + t
