"""BEV box geometry: corners, rotated IoU (shapely), angle wrapping."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def wrap_pi(a):
    """Wrap angles to (-pi, pi]."""
    a = np.asarray(a, dtype=np.float64)
    out = (a + np.pi) % (2.0 * np.pi) - np.pi
    # map -pi to +pi so the interval is (-pi, pi]
    return np.where(np.isclose(out, -np.pi), np.pi, out)


def wrap_half_pi(a):
    """Flip-aware wrap to (-pi/2, pi/2]: a 180 deg flip counts as zero error."""
    return wrap_pi(2.0 * np.asarray(a, dtype=np.float64)) / 2.0


def bev_corners(box) -> np.ndarray:
    """4x2 BEV corners of one box (x, y, z, l, w, h, yaw), counter-clockwise."""
    x, y, _, l, w, _, yaw = [float(v) for v in box[:7]]
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[l / 2, w / 2], [-l / 2, w / 2], [-l / 2, -w / 2], [l / 2, -w / 2]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def _poly(box) -> Polygon:
    return Polygon(bev_corners(box))


def bev_iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Rotated BEV IoU, shape (len(pred), len(gt)).

    Uses a circumscribed-circle prefilter, so cost is ~O(P*G) cheap checks
    plus shapely only for overlapping candidates.
    """
    P, G = len(pred), len(gt)
    iou = np.zeros((P, G), dtype=np.float64)
    if P == 0 or G == 0:
        return iou
    rp = 0.5 * np.hypot(pred[:, 3], pred[:, 4])
    rg = 0.5 * np.hypot(gt[:, 3], gt[:, 4])
    d = np.hypot(pred[:, None, 0] - gt[None, :, 0], pred[:, None, 1] - gt[None, :, 1])
    cand = d < (rp[:, None] + rg[None, :])
    if not cand.any():
        return iou
    pp = {i: _poly(pred[i]) for i in np.unique(np.nonzero(cand)[0])}
    gp = {j: _poly(gt[j]) for j in np.unique(np.nonzero(cand)[1])}
    for i, j in zip(*np.nonzero(cand)):
        inter = pp[i].intersection(gp[j]).area
        if inter > 0:
            iou[i, j] = inter / (pp[i].area + gp[j].area - inter)
    return iou
