"""Geometric augmentation for LSS with exact intrinsics/extrinsics bookkeeping.

Two independent augmentations, as in BEVDet (Huang et al., arXiv:2112.11790):

IDA (image-view): resize / crop / horizontal flip / small rotation of each
camera image. Expressed as one 3x3 pixel affine M. The new intrinsics are
K' = M @ K; the image and any dense depth map are warped with the SAME M via
cv2.warpAffine, so consistency holds by construction (no half-pixel guesses).

BDA (BEV-space): random rotation / flip / scale of the whole ego frame,
expressed as one 4x4 matrix A applied to (a) the 3D geometry LSS splats into
(frustum points after cam->ego), (b) the GT boxes, and (c) for cooperation,
every collaborator->ego transform (T' = A @ T). BEVDet reports that image-view
augmentation does not regularise the BEV encoder/head because the view
transform decouples the two spaces, and that the BEV side overfits without
BDA. Your symptom (colour jitter did not help; yaw memorised) matches that.

IMPORTANT: if your code inverts extrinsics assuming rigidity (R^T), applying a
scaled A to cam2ego breaks it. Safer: apply A to the frustum points AFTER the
cam->ego transform (`bda_apply_points`), which is what BEVDet does.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


# ----------------------------------------------------------------------------
# BDA
# ----------------------------------------------------------------------------
@dataclass
class BDAParams:
    rot_deg: float = 0.0
    flip_x: bool = False   # x -> -x (front/back)
    flip_y: bool = False   # y -> -y (left/right)
    scale: float = 1.0


def sample_bda(rng: np.random.Generator, rot_range_deg=22.5, scale_range=(0.95, 1.05),
               flip_p=0.5) -> BDAParams:
    """BEVDet defaults: rot U[-22.5, 22.5] deg, scale U[0.95, 1.05], flips."""
    return BDAParams(rot_deg=float(rng.uniform(-rot_range_deg, rot_range_deg)),
                     flip_x=bool(rng.random() < flip_p),
                     flip_y=bool(rng.random() < flip_p),
                     scale=float(rng.uniform(*scale_range)))


def bda_matrix(p: BDAParams) -> np.ndarray:
    """4x4 A = S @ F @ R acting on homogeneous ego-frame points."""
    t = np.radians(p.rot_deg)
    R = np.eye(4); R[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    F = np.diag([-1.0 if p.flip_x else 1.0, -1.0 if p.flip_y else 1.0, 1.0, 1.0])
    S = np.diag([p.scale, p.scale, p.scale, 1.0])
    return S @ F @ R


def bda_apply_points(A: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply A to (..., 3) points (numpy or torch-compatible via @)."""
    return pts @ A[:3, :3].T + A[:3, 3]


def bda_apply_boxes(A: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Apply A to (N, 7) boxes. Heading is transformed as a direction vector,
    which handles flips correctly (flip_y: yaw -> -yaw; flip_x: yaw -> pi-yaw)."""
    b = np.asarray(boxes, float).reshape(-1, 7).copy()
    if len(b) == 0:
        return b
    L = A[:3, :3]
    s = np.cbrt(abs(np.linalg.det(L)))
    b[:, :3] = b[:, :3] @ L.T + A[:3, 3]
    d = np.stack([np.cos(b[:, 6]), np.sin(b[:, 6]), np.zeros(len(b))], 1) @ L.T
    b[:, 6] = np.arctan2(d[:, 1], d[:, 0])
    b[:, 3:6] *= s
    return b


# ----------------------------------------------------------------------------
# IDA
# ----------------------------------------------------------------------------
@dataclass
class IDAParams:
    resize: float = 1.0
    crop_xy: tuple = (0, 0)       # top-left of crop in the RESIZED image
    out_hw: tuple = (384, 512)
    flip: bool = False
    rot_deg: float = 0.0          # about the output image centre


def sample_ida(rng, src_hw=(600, 800), out_hw=(384, 512), resize_jitter=(-0.06, 0.11),
               flip_p=0.5, rot_range_deg=5.4) -> IDAParams:
    """BEVDet-style sampling (ranges are BEVDet's nuScenes defaults; treat them
    as a starting point, not tuned for OPV2V). Base resize keeps the 4:3
    aspect (800->512 gives 0.64, 600->384). Larger resize -> random crop;
    smaller -> zero padding (negative crop offset)."""
    r = out_hw[1] / src_hw[1] + float(rng.uniform(*resize_jitter))
    rh, rw = int(round(src_hw[0] * r)), int(round(src_hw[1] * r))

    def off(full, out):
        return int(rng.integers(0, full - out + 1)) if full >= out else -int(rng.integers(0, out - full + 1))

    return IDAParams(resize=r, crop_xy=(off(rw, out_hw[1]), off(rh, out_hw[0])), out_hw=out_hw,
                     flip=bool(rng.random() < flip_p),
                     rot_deg=float(rng.uniform(-rot_range_deg, rot_range_deg)))


def ida_eval_params(src_hw=(600, 800), out_hw=(384, 512)) -> IDAParams:
    """Deterministic test-time transform: aspect-preserving resize + centre crop."""
    r = out_hw[1] / src_hw[1]
    rh = int(round(src_hw[0] * r))
    return IDAParams(resize=r, crop_xy=(0, max(rh - out_hw[0], 0) // 2), out_hw=out_hw)


def ida_matrix(p: IDAParams) -> np.ndarray:
    """3x3 pixel affine (integer-pixel-centre convention, same as OpenCV)."""
    H, W = p.out_hw
    # resize about pixel centres: x' = r*(x+0.5)-0.5
    M = np.array([[p.resize, 0, 0.5 * p.resize - 0.5],
                  [0, p.resize, 0.5 * p.resize - 0.5],
                  [0, 0, 1.0]])
    M = np.array([[1, 0, -p.crop_xy[0]], [0, 1, -p.crop_xy[1]], [0, 0, 1.0]]) @ M
    if p.flip:
        M = np.array([[-1, 0, W - 1], [0, 1, 0], [0, 0, 1.0]]) @ M
    if p.rot_deg:
        c = np.array([(W - 1) / 2, (H - 1) / 2])
        t = np.radians(p.rot_deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        T = np.eye(3); T[:2, :2] = R; T[:2, 2] = c - R @ c
        M = T @ M
    return M


def ida_apply_intrinsics(M: np.ndarray, K: np.ndarray) -> np.ndarray:
    """K' = M @ K. With rotation the result is not upper-triangular; LSS-style
    code that only uses inverse(K') to lift pixels is still exact. If your
    lift code assumes a pinhole K (reads fx, cx), keep rot_deg = 0 or pass
    LSS-style post_rot/post_tran = M[:2,:2], M[:2,2] instead."""
    return M @ K


def ida_apply_image(M, img, out_hw):
    assert cv2 is not None, "opencv-python required"
    return cv2.warpAffine(img, M[:2], (out_hw[1], out_hw[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def ida_apply_dense_depth(M, depth, out_hw):
    """Dense depth (e.g. ZoeDepth prior): nearest-neighbour, 0 = invalid."""
    assert cv2 is not None, "opencv-python required"
    return cv2.warpAffine(depth.astype(np.float32), M[:2], (out_hw[1], out_hw[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def ida_apply_sparse_depth(M, uv, depth, out_hw):
    """Sparse depth (projected LiDAR): transform pixel coords, drop out-of-image."""
    uvh = np.concatenate([uv, np.ones((len(uv), 1))], 1) @ M.T
    H, W = out_hw
    ok = (uvh[:, 0] >= -0.5) & (uvh[:, 0] < W - 0.5) & (uvh[:, 1] >= -0.5) & (uvh[:, 1] < H - 0.5)
    return uvh[ok, :2], depth[ok]


def downsample_depth_min(depth, factor):
    """Downsample a depth map to feature resolution taking the MIN valid depth
    per block (BEVDepth convention: nearest surface wins). 0 = invalid."""
    H, W = depth.shape
    h, w = H // factor, W // factor
    d = depth[:h * factor, :w * factor].reshape(h, factor, w, factor).transpose(0, 2, 1, 3).reshape(h, w, -1)
    d = np.where(d > 0, d, np.inf).min(-1)
    return np.where(np.isfinite(d), d, 0.0)
