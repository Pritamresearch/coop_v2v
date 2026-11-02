"""
detection_fixed.py -- drop-in replacement for src/loss/detection.py plus the
target-drawing helpers your loader should use.

Two defects are fixed here, both of which independently produce
"loss -> 0, AP = 0, heatmap decays to zero":

  (1) `pos_inds = gt.eq(1)` silently selects nothing when the Gaussian peak is
      0.99999994 in float32. Fixed with a tolerance AND by forcing the peak.
  (2) Silent num_pos == 0. Previously this produced a small, plausible-looking
      loss. Now it is LOUD -- it cannot pass unnoticed again.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

PEAK_TOL = 1e-4          # a cell is "positive" if gt_heat >= 1 - PEAK_TOL
_warned = {"nopos": 0}


# ===========================================================================
# 1. Focal loss
# ===========================================================================
def gaussian_focal_loss(pred, gt, alpha=2.0, beta=4.0, eps=1e-4,
                        strict=True, return_parts=False):
    """
    Penalty-reduced focal loss (CenterNet / CornerNet).

    pred : (B,1,H,W) or (B,C,H,W) POST-sigmoid probabilities in (0,1)
    gt   : same shape, Gaussian-splatted targets with peaks at exactly 1.0

    strict=True raises if a batch contains no positive cell. Keep it True.
    A silent num_pos==0 is precisely the bug that cost you these training runs:
    the loss stays small and finite, so nothing looks wrong, while the model is
    being trained to predict "empty everywhere".
    """
    pred = pred.clamp(min=eps, max=1.0 - eps)

    # TOLERANCE, not exact equality. This is fix (1).
    pos_inds = gt.ge(1.0 - PEAK_TOL).float()
    neg_inds = 1.0 - pos_inds

    neg_weights = torch.pow(1.0 - gt, beta)

    pos_loss = -torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos_inds
    neg_loss = -torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    pos_sum = pos_loss.sum()
    neg_sum = neg_loss.sum()

    if num_pos < 1:
        _warned["nopos"] += 1
        msg = (f"gaussian_focal_loss: NO POSITIVE CELLS in this batch "
               f"(gt.max()={float(gt.max()):.8f}, "
               f"cells>0={int((gt > 0).sum())}). Only the negative term is "
               f"active -- the heatmap head is being trained to output zero "
               f"everywhere. Check the box->grid conversion and the Gaussian "
               f"peak amplitude.")
        if strict:
            raise RuntimeError(msg)
        if _warned["nopos"] <= 5:
            print("WARNING: " + msg)
        loss = neg_sum
    else:
        loss = (pos_sum + neg_sum) / num_pos

    if return_parts:
        return loss, {
            "num_pos": float(num_pos),
            "pos": float(pos_sum / num_pos.clamp(min=1)),
            "neg": float(neg_sum / num_pos.clamp(min=1)),
            "pred_at_pos": float((pred * pos_inds).sum() / num_pos.clamp(min=1)),
            "pred_at_neg": float((pred * neg_inds).sum() / neg_inds.sum().clamp(min=1)),
        }
    return loss


# ===========================================================================
# 2. Detection loss with per-component reporting
# ===========================================================================
def detection_loss(heat, reg, gt_heat, gt_reg, gt_mask,
                   w_hm=1.0, w_reg=1.0, strict=True):
    """
    Returns (total, parts). `parts` now always carries num_pos and the
    separated hm/reg magnitudes, so a collapse is visible in train_log.csv
    on the epoch it happens rather than 100 epochs later.
    """
    hm_loss, hm_parts = gaussian_focal_loss(
        heat, gt_heat, strict=strict, return_parts=True)

    m = gt_mask.unsqueeze(1) if gt_mask.dim() == reg.dim() - 1 else gt_mask
    m = m.expand_as(reg).float()
    n = m.sum().clamp(min=1.0)
    reg_loss = (F.smooth_l1_loss(reg * m, gt_reg * m, reduction="sum") / n)

    total = w_hm * hm_loss + w_reg * reg_loss
    parts = {"hm": float(hm_loss), "reg": float(reg_loss), **hm_parts}
    return total, parts


# ===========================================================================
# 3. Target construction -- guarantees a peak of exactly 1.0
# ===========================================================================
def gaussian_radius(det_size, min_overlap=0.7):
    """CenterNet radius: smallest r such that a box shifted by r still has
    >= min_overlap IoU with the true box. Prevents over-wide Gaussians on
    large objects, which flatten the target and weaken positive supervision."""
    h, w = det_size
    a1 = 1
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 - math.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0))) / (2 * a1)

    a2 = 4
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 - math.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0))) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0))) / (2 * a3)
    return max(0.0, min(r1, r2, r3))


def _gaussian2d(shape, sigma=1.0):
    m, n = [(s - 1.) / 2. for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    g = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    g[g < np.finfo(g.dtype).eps * g.max()] = 0
    return g


def draw_gaussian(heatmap, center, radius, k=1.0):
    """
    Splat a Gaussian onto `heatmap` (H,W numpy float32) at integer `center`=(cx,cy)
    i.e. (col, row). Returns the heatmap.

    The final line is fix (2): the centre cell is FORCED to exactly 1.0 after
    the splat. Without it, float32 accumulation can leave the peak at
    0.99999994 and `gt.eq(1)` finds nothing.
    """
    diameter = 2 * int(radius) + 1
    gaussian = _gaussian2d((diameter, diameter), sigma=diameter / 6.0)

    cx, cy = int(center[0]), int(center[1])
    H, W = heatmap.shape[:2]
    radius = int(radius)

    left, right = min(cx, radius), min(W - cx, radius + 1)
    top, bottom = min(cy, radius), min(H - cy, radius + 1)
    if right <= 0 or bottom <= 0 or left < 0 or top < 0:
        return heatmap

    masked_heatmap = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                               radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    heatmap[cy, cx] = 1.0          # <-- FORCE the peak. Non-negotiable.
    return heatmap


def build_bev_targets(boxes, cfg, n_reg=4):
    """
    boxes : (M,>=5) array of (cx, cy, w, l, yaw) in EGO METRIC coordinates.
    cfg   : must expose x_min,x_max,y_min,y_max,bev_H,bev_W.

    Returns gt_heat (1,H,W), gt_reg (n_reg,H,W), gt_mask (H,W), n_kept.

    CONVENTION, stated explicitly so it can be checked against decode():
        col = (cx - x_min) / (x_max - x_min) * W      -> x maps to COLUMN
        row = (cy - y_min) / (y_max - y_min) * H      -> y maps to ROW
    decode() in src/metrics.py MUST invert exactly this. If the two disagree,
    every box is placed at a transposed/reflected location and AP is 0 while
    the loss looks perfectly healthy.
    """
    H, W = int(cfg.bev_H), int(cfg.bev_W)
    x_min, x_max = float(cfg.x_min), float(cfg.x_max)
    y_min, y_max = float(cfg.y_min), float(cfg.y_max)
    res_x = (x_max - x_min) / W
    res_y = (y_max - y_min) / H

    heat = np.zeros((1, H, W), dtype=np.float32)
    reg = np.zeros((n_reg, H, W), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, np.asarray(boxes).shape[-1])
    n_kept = 0
    for b in boxes:
        cx, cy, bw, bl = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        fcol = (cx - x_min) / res_x
        frow = (cy - y_min) / res_y
        col, row = int(fcol), int(frow)
        if not (0 <= row < H and 0 <= col < W):
            continue                      # genuinely outside the BEV range

        r = gaussian_radius((max(bl / res_y, 1.0), max(bw / res_x, 1.0)))
        r = max(int(r), 1)                # radius 0 => a single cell, still fine,
                                          # but 1 gives the head a gradient to follow
        draw_gaussian(heat[0], (col, row), r)

        reg[0, row, col] = fcol - col     # sub-cell offset x
        reg[1, row, col] = frow - row     # sub-cell offset y
        reg[2, row, col] = math.log(max(bw, 1e-3))
        reg[3, row, col] = math.log(max(bl, 1e-3))
        mask[row, col] = 1.0              # mask indexes the SAME cell as the peak
        n_kept += 1

    assert n_kept == 0 or heat.max() == 1.0, \
        f"target builder produced peak {heat.max()!r}, expected exactly 1.0"
    return heat, reg, mask, n_kept