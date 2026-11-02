"""Which GT boxes can the ego's cameras actually see?

Why this matters: in OPV2V each CAV yaml lists surrounding vehicles from the
simulator, and cooperative GT is the union over CAVs. An ego-only camera
detector trained on boxes it cannot see (fully occluded, or only visible to
another CAV) can only fit them by memorising scene-specific context. That
produces exactly "train loss keeps falling, val gets worse", and it affects
yaw most because heading of an unseen car can only be memorised.

HEAL's camera path filters GT with the ego's `bev_visibility.png` (from
OPV2V's `additional` download) in `generate_visible_object_center`; its
docstring describes a 256x256 map at 0.39 m/px, heading up. I have NOT verified
the exact pixel convention (which axis is up, sign of y, centre pixel), so this
module makes it a parameter and gives you `overlay_boxes_on_map()` to check
alignment visually on a few frames before trusting any number.

Three visibility definitions, from strongest to weakest evidence:
  1. LiDAR point count inside the box (needs the ego .pcd)       -> occlusion
  2. bev_visibility.png lookup (needs OPV2V 'additional' files)  -> occlusion
  3. camera frustum test (always available)                      -> FOV only;
     with 4 cameras OPV2V coverage is near 360 deg, so this alone marks
     almost nothing invisible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import bev_corners


# --- 1. LiDAR points in box -------------------------------------------------
def points_in_boxes_count(points: np.ndarray, boxes: np.ndarray, z_margin=0.3) -> np.ndarray:
    """Count points (P,3+) inside each 3D box (N,7). Points and boxes must be in
    the same frame (ego LiDAR frame). Box z is taken as the box CENTRE; if your
    yaml-derived boxes use the bottom centre, shift by h/2 first."""
    boxes = np.asarray(boxes, float).reshape(-1, 7)
    pts = np.asarray(points, float)[:, :3]
    out = np.zeros(len(boxes), int)
    for k, (x, y, z, l, w, h, yaw) in enumerate(boxes):
        d = pts - [x, y, z]
        c, s = np.cos(-yaw), np.sin(-yaw)
        lx = d[:, 0] * c - d[:, 1] * s
        ly = d[:, 0] * s + d[:, 1] * c
        m = (np.abs(lx) <= l / 2) & (np.abs(ly) <= w / 2) & (np.abs(d[:, 2]) <= h / 2 + z_margin)
        out[k] = int(m.sum())
    return out


# --- 2. bev_visibility.png --------------------------------------------------
@dataclass
class VisMapSpec:
    size_px: int = 256
    res_m: float = 0.39
    # pixel (row, col) of the ego; default: image centre
    centre_rc: tuple = (128, 128)
    # how ego +x / +y map to image axes. Default "heading up": +x -> up (-row),
    # +y (left) -> left (-col). VERIFY with overlay_boxes_on_map().
    x_to_row: float = -1.0
    y_to_col: float = -1.0


def ego_xy_to_px(xy: np.ndarray, spec: VisMapSpec):
    r = spec.centre_rc[0] + spec.x_to_row * xy[:, 0] / spec.res_m
    c = spec.centre_rc[1] + spec.y_to_col * xy[:, 1] / spec.res_m
    return r, c


def visible_from_map(boxes: np.ndarray, vis_map: np.ndarray, spec: VisMapSpec,
                     min_frac=0.0, n_samples=7) -> np.ndarray:
    """A box is visible if more than `min_frac` of sample points over its BEV
    footprint fall on visible (non-zero) map pixels. Boxes outside the map are
    marked not visible."""
    boxes = np.asarray(boxes, float).reshape(-1, 7)
    g = np.linspace(-0.45, 0.45, n_samples)
    uu, vv = np.meshgrid(g, g)
    out = np.zeros(len(boxes), bool)
    for k, b in enumerate(boxes):
        x, y, _, l, w, _, yaw = b
        c, s = np.cos(yaw), np.sin(yaw)
        px = x + uu.ravel() * l * c - vv.ravel() * w * s
        py = y + uu.ravel() * l * s + vv.ravel() * w * c
        r, cc = ego_xy_to_px(np.stack([px, py], 1), spec)
        r = np.round(r).astype(int); cc = np.round(cc).astype(int)
        inside = (r >= 0) & (r < vis_map.shape[0]) & (cc >= 0) & (cc < vis_map.shape[1])
        if not inside.any():
            continue
        hit = np.zeros(len(r), bool)
        hit[inside] = vis_map[r[inside], cc[inside]] > 0
        out[k] = hit.mean() > min_frac
    return out


def overlay_boxes_on_map(vis_map, boxes, spec: VisMapSpec, out_path):
    """Save a PNG with GT footprints drawn on the visibility map. Check on 3-5
    frames that boxes sit on the road/visible region before using the filter."""
    import cv2
    img = vis_map.copy()
    if img.ndim == 2:
        img = cv2.cvtColor((img > 0).astype(np.uint8) * 120, cv2.COLOR_GRAY2BGR)
    for b in np.asarray(boxes).reshape(-1, 7):
        cr = bev_corners(b)
        r, c = ego_xy_to_px(cr, spec)
        poly = np.stack([c, r], 1).round().astype(np.int32)
        cv2.polylines(img, [poly], True, (0, 0, 255), 1)
        hr, hc = ego_xy_to_px(np.array([[b[0] + b[3] / 2 * np.cos(b[6]), b[1] + b[3] / 2 * np.sin(b[6])]]), spec)
        cr0, cc0 = ego_xy_to_px(b[None, :2], spec)
        cv2.line(img, (int(cc0[0]), int(cr0[0])), (int(hc[0]), int(hr[0])), (0, 255, 0), 1)
    cv2.imwrite(str(out_path), img)


# --- 3. camera frustum ------------------------------------------------------
def in_any_camera_frustum(boxes, cams, img_hw, d_max=50.0, min_corners=1) -> np.ndarray:
    """cams: list of (K 3x3, ego2cam 4x4 in OpenCV camera axes). A box counts
    if >= min_corners of its 8 corners project inside some image at depth
    in (0.1, d_max]."""
    boxes = np.asarray(boxes, float).reshape(-1, 7)
    H, W = img_hw
    out = np.zeros(len(boxes), bool)
    for k, b in enumerate(boxes):
        c2 = bev_corners(b)
        z0, z1 = b[2] - b[5] / 2, b[2] + b[5] / 2
        pts = np.concatenate([np.c_[c2, np.full(4, z0)], np.c_[c2, np.full(4, z1)]])
        ph = np.c_[pts, np.ones(8)]
        for K, e2c in cams:
            pc = (ph @ e2c.T)[:, :3]
            z = pc[:, 2]
            uv = pc @ K.T
            uv = uv[:, :2] / np.maximum(uv[:, 2:3], 1e-6)
            ok = (z > 0.1) & (z <= d_max) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if ok.sum() >= min_corners:
                out[k] = True
                break
    return out
