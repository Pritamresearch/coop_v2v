"""
bev_augment.py -- Drop-in BEV augmentation for OPV2VDataset.__getitem__.

PURPOSE
-------
The training data has a strong positional bias: the ego vehicle almost always
follows a lead vehicle at ~30m forward. Without augmentation, the detection
head exploits this by learning "always predict at (109,165)" -- a fixed BEV
cell -- rather than learning to detect arbitrary objects.

BEV rotation breaks this shortcut by rotating the entire scene (images,
camera extrinsics, GT boxes, relative poses) by a random angle, so the lead
vehicle appears at varied BEV locations across training.

USAGE
-----
In OPV2VDataset.__getitem__, just before building the scene dict, call:

    if self.augment:
        scene_tensors = bev_rotate_scene(scene_tensors, max_angle_deg=180.0)

Where scene_tensors is a dict with keys:
    rots      (N, n_cams, 3, 3)  cam->ego rotation
    trans     (N, n_cams, 3)     cam origin in ego frame
    rel_pose  (N, 3)             [dx, dy, dyaw]
    gt_boxes  (M, 7)             [cx,cy,cz,w,l,h,yaw]

The GT heatmap (heat/offset/mask) should be rebuilt from the rotated gt_boxes
using _build_bev_gt AFTER this function returns.
"""
import math
import numpy as np
import torch


def _rot2d(angle_rad: float) -> np.ndarray:
    """2D rotation matrix for angle in radians."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def bev_rotate_scene(
    rots:     np.ndarray,    # (N, n_cams, 3, 3)
    trans:    np.ndarray,    # (N, n_cams, 3)
    rel_pose: np.ndarray,    # (N, 3)  [dx, dy, dyaw]
    gt_boxes: np.ndarray,    # (M, 7)  [cx,cy,cz,w,l,h,yaw]
    max_angle_deg: float = 180.0,
    rng=None,
) -> tuple:
    """
    Rotate the entire scene around the ego origin by a random angle.
    Returns rotated (rots, trans, rel_pose, gt_boxes).

    The rotation is applied to:
    - Camera extrinsics (rots, trans): camera origins and orientations in ego frame
    - Relative poses: collaborator positions/headings in ego frame
    - GT box centres and headings

    Images are NOT modified -- they remain in each camera's own frame.
    The BEV grid implicitly rotates because features are lifted via the
    (now-rotated) camera extrinsics, so they land at rotated positions in BEV.
    """
    if rng is None:
        rng = np.random.default_rng()

    angle = rng.uniform(-max_angle_deg, max_angle_deg) * (math.pi / 180.0)
    R2 = _rot2d(angle)

    # --- rotation matrix for 3D (rotate around Z / vertical axis) ---
    R3 = np.eye(3, dtype=np.float32)
    R3[:2, :2] = R2

    rots_out = rots.copy()
    trans_out = trans.copy()
    rel_pose_out = rel_pose.copy()
    gt_boxes_out = gt_boxes.copy() if gt_boxes is not None and len(gt_boxes) > 0 else gt_boxes

    N, n_cams = rots.shape[:2]

    # --- rotate camera extrinsics ---
    # rots[i,c] = R_cam2ego: rotate the ego frame by R3 means
    # new R_cam2ego = R3 @ R_cam2ego
    # new trans[i,c] = R3 @ trans[i,c]  (cam origin in new ego frame)
    for i in range(N):
        for c in range(n_cams):
            rots_out[i, c] = R3 @ rots[i, c]
            trans_out[i, c] = R3 @ trans[i, c]

    # --- rotate relative poses ---
    for i in range(N):
        dx, dy, dyaw = rel_pose[i]
        new_xy = R2 @ np.array([dx, dy])
        rel_pose_out[i, 0] = new_xy[0]
        rel_pose_out[i, 1] = new_xy[1]
        rel_pose_out[i, 2] = dyaw + angle  # heading also rotates

    # --- rotate GT boxes ---
    if gt_boxes_out is not None and len(gt_boxes_out) > 0:
        cx = gt_boxes_out[:, 0].copy()
        cy = gt_boxes_out[:, 1].copy()
        new_xy = R2 @ np.stack([cx, cy])           # (2, M)
        gt_boxes_out[:, 0] = new_xy[0]
        gt_boxes_out[:, 1] = new_xy[1]
        gt_boxes_out[:, 6] = gt_boxes_out[:, 6] + angle
        # wrap yaw to [-pi, pi] so metric wrapping and sin/cos targets are consistent
        gt_boxes_out[:, 6] = (gt_boxes_out[:, 6] + np.pi) % (2 * np.pi) - np.pi   # yaw

    return rots_out, trans_out, rel_pose_out, gt_boxes_out


# ===========================================================================
# Size-adaptive Gaussian radius (replaces fixed sigma=2.0)
# ===========================================================================
def gaussian_radius_cells(w_m: float, l_m: float, cell_m: float,
                           min_overlap: float = 0.7) -> float:
    """
    CenterPoint-style Gaussian radius in CELLS, computed from object size.

    For a box of width w and length l, the radius r is the largest value such
    that a box shifted by r cells still has >= min_overlap IoU.
    Ensures larger objects get wider Gaussian supervision, preventing the
    model from treating all objects as single-pixel targets.
    """
    w = w_m / cell_m
    l = l_m / cell_m

    # Three geometric constraints (from CenterNet paper appendix)
    def r_from_constraint(a, b, c):
        disc = max(b*b - 4*a*c, 0)
        return (-b + math.sqrt(disc)) / (2*a)

    r1 = r_from_constraint(1, w + l, w * l * (1 - min_overlap) / (1 + min_overlap))
    r2 = r_from_constraint(4, 2 * (w + l), (1 - min_overlap) * w * l)
    r3 = r_from_constraint(4 * min_overlap,
                           -2 * min_overlap * (w + l),
                           (min_overlap - 1) * w * l)
    radius = min(r1, r2, r3)
    return max(1.0, radius)   # at least 1 cell


def draw_gaussian_sized(heat: np.ndarray, row: float, col: float,
                        w_m: float, l_m: float, cell_m: float,
                        centerpoint: bool = False) -> np.ndarray:
    """
    Draw a size-adaptive Gaussian onto `heat` (H,W float32).
    Replaces the fixed sigma=2.0 call in _build_bev_gt.

    centerpoint=False keeps the original target (min_overlap=0.7, r>=1,
    sigma=r/3: a near-delta at 0.4 m cells). centerpoint=True uses the
    CenterPoint target (min_overlap=0.1, r>=2, sigma=(2r+1)/6).
    """
    H, W = heat.shape
    if centerpoint:
        r = max(2.0, gaussian_radius_cells(w_m, l_m, cell_m, min_overlap=0.1))
        sigma = (2 * int(r) + 1) / 6.0
    else:
        r = gaussian_radius_cells(w_m, l_m, cell_m)
        sigma = r / 3.0
    diameter = 2 * int(r) + 1

    # Build kernel
    size = diameter
    y_range = np.arange(size, dtype=np.float32) - size // 2
    x_range = np.arange(size, dtype=np.float32) - size // 2
    X, Y = np.meshgrid(x_range, y_range)
    g = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0

    # Paste into heatmap
    ri, ci = int(round(row)), int(round(col))
    radius = int(r)
    top = max(0, ri - radius);  bot = min(H, ri + radius + 1)
    left = max(0, ci - radius); right = min(W, ci + radius + 1)
    gtop = radius - (ri - top);  gbot = gtop + (bot - top)
    gleft = radius - (ci - left); gright = gleft + (right - left)

    if bot > top and right > left and gbot > gtop and gright > gleft:
        patch = heat[top:bot, left:right]
        gkernel = g[gtop:gbot, gleft:gright]
        np.maximum(patch, gkernel, out=patch)

    # Force exact 1.0 at peak (prevent float32 rounding causing pos_inds=0)
    if 0 <= ri < H and 0 <= ci < W:
        heat[ri, ci] = 1.0

    return heat