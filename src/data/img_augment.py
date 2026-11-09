"""BEVDet-style image-view augmentation.

Follows HuangJunJie2017/BEVDet `PrepareImageInputs.sample_augmentation` / `img_transform`
(configs/bevdet/bevdet-r50.py, `data_config`) exactly: resize -> crop -> horizontal flip -> rotate,
with PIL doing the pixel work so the sign conventions match the reference implementation.

Camera intrinsics are NOT modified. The whole 2D transform is carried in `post_rot` (3x3) and
`post_tran` (3,), which `lss_backbone.frustum_to_ego` inverts before un-projecting, so the lifted
BEV geometry is unchanged by image augmentation. Geometry only -- no photometric jitter, matching
BEVDet, whose pipeline applies only normalisation to pixel values.

`image_aug_consistent()` is the gate: it checks that the returned (post_rot, post_tran) really do
predict where the pixels moved, which is what catches a sign or ordering error.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# BEVDet bevdet-r50.py data_config, training ranges
RESIZE_RANGE = (-0.06, 0.11)   # added to the base scale fW / srcW
ROT_RANGE = (-5.4, 5.4)        # degrees
CROP_H_RANGE = (0.0, 0.0)      # (0, 0) => fixed bottom-aligned vertical crop
FLIP_P = 0.5


def _rot2(deg: float) -> np.ndarray:
    """BEVDet's get_rot: [[cos, sin], [-sin, cos]]."""
    h = np.deg2rad(deg)
    c, s = np.cos(h), np.sin(h)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def sample_params(src_hw, out_hw, rng, train: bool = True) -> dict:
    """Draw one set of augmentation parameters (BEVDet sample_augmentation)."""
    srcH, srcW = src_hw
    fH, fW = out_hw
    if train:
        resize = fW / srcW + rng.uniform(*RESIZE_RANGE)
        newW, newH = int(srcW * resize), int(srcH * resize)
        crop_h = int((1 - rng.uniform(*CROP_H_RANGE)) * newH) - fH
        crop_w = int(rng.uniform(0, max(0, newW - fW)))
        flip = bool(rng.uniform() < FLIP_P)
        rotate = float(rng.uniform(*ROT_RANGE))
    else:
        resize = fW / srcW
        newW, newH = int(srcW * resize), int(srcH * resize)
        crop_h = int((1 - np.mean(CROP_H_RANGE)) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        flip, rotate = False, 0.0
    return {"resize": resize, "resize_dims": (newW, newH),
            "crop": (crop_w, crop_h, crop_w + fW, crop_h + fH),
            "flip": flip, "rotate": rotate}


def apply(img: np.ndarray, p: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply one parameter set to an HxWx3 uint8 RGB image.

    Returns (augmented image, post_rot (3,3), post_tran (3,)) where, for an original pixel
    (u, v), the augmented pixel is post_rot[:2,:2] @ (u, v) + post_tran[:2].
    """
    crop = p["crop"]
    pil = Image.fromarray(img).resize(p["resize_dims"], Image.BILINEAR).crop(crop)
    if p["flip"]:
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
    pil = pil.rotate(p["rotate"])

    post_rot = np.eye(2) * p["resize"]
    post_tran = -np.array(crop[:2], dtype=np.float64)
    if p["flip"]:
        A = np.array([[-1.0, 0.0], [0.0, 1.0]])
        b = np.array([crop[2] - crop[0], 0.0])
        post_rot, post_tran = A @ post_rot, A @ post_tran + b
    A = _rot2(p["rotate"])
    b = np.array([crop[2] - crop[0], crop[3] - crop[1]], dtype=np.float64) / 2
    b = A @ (-b) + b
    post_rot, post_tran = A @ post_rot, A @ post_tran + b

    pr = np.eye(3, dtype=np.float32)
    pr[:2, :2] = post_rot
    pt = np.zeros(3, dtype=np.float32)
    pt[:2] = post_tran
    return np.asarray(pil), pr, pt


def image_aug_consistent(src_hw=(600, 800), out_hw=(384, 512), n: int = 40,
                         downsample: int = 8, tol_cells: float = 0.25,
                         seed: int = 0, verbose: bool = True) -> bool:
    """Gate: do (post_rot, post_tran) actually predict where pixels move?

    Puts single bright markers at known pixels in a black image, runs the real transform, and checks
    each surviving marker lands where post_rot @ (u, v) + post_tran says it should. A sign error or a
    wrong composition order shows up as an error of tens of pixels, with no model weights involved.

    The error is judged in FEATURE CELLS (`downsample` pixels each), because that is the unit the
    lift actually samples. A residual of about half a pixel is expected and is not a bug: the marker
    centroid measures pixel *centres* while post_rot/post_tran are corner-based, and bilinear
    resampling spreads the marker. BEVDet shares both conventions.
    """
    rng = np.random.default_rng(seed)
    srcH, srcW = src_hw
    errs, checked, dropped = [], 0, 0
    for _ in range(n):
        p = sample_params(src_hw, out_hw, rng, train=True)
        us = rng.integers(0, srcW, 12)
        vs = rng.integers(0, srcH, 12)
        for u, v in zip(us, vs):
            img = np.zeros((srcH, srcW, 3), np.uint8)
            img[v, u] = 255
            out, pr, pt = apply(img, p)
            pred = pr[:2, :2] @ np.array([u, v], float) + pt[:2]
            ys, xs = np.nonzero(out.max(axis=2) > 25)       # bilinear spreads the marker
            if len(xs) == 0:
                dropped += 1                                 # fell outside the crop: expected
                continue
            w = out.max(axis=2)[ys, xs].astype(float)
            got = np.array([(xs * w).sum() / w.sum(), (ys * w).sum() / w.sum()])
            errs.append(float(np.hypot(*(got - pred))))
            checked += 1
    errs = np.array(errs) / float(downsample)       # pixels -> feature cells
    ok = checked > 0 and float(np.quantile(errs, 0.99)) <= tol_cells
    if verbose:
        print(f"  {'PASS' if ok else 'FAIL'}  image aug post_rot/post_tran vs actual pixel motion: "
              f"n={checked} (dropped by crop {dropped})  median {np.median(errs):.4f} cells  "
              f"p99 {np.quantile(errs, 0.99):.4f}  max {errs.max():.4f}  (tol {tol_cells} cells "
              f"= {tol_cells * downsample:.1f} px)")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if image_aug_consistent() else 1)
