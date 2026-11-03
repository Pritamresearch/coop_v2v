"""
Detection loss (role of the IFTR / OpenCOOD head loss).

    L_det = focal_heatmap + lambda_off * smooth_l1(offset)
                          + lambda_sz  * smooth_l1(log-size)
                          + lambda_yaw * smooth_l1(sin,cos)
                          + lambda_zh  * smooth_l1(z,h)

Only cells at object centres (gt_mask) contribute to the regression terms.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def gaussian_focal_loss(pred: torch.Tensor, gt: torch.Tensor,
                        alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-4,
                        pos_weight: float = 1.0, norm: str = "sepnorm") -> torch.Tensor:
    """
    CornerNet / CenterNet focal loss for a per-cell heatmap.
    pred, gt : (H, W)   both in [0, 1]

    pos_weight: extra multiplier on the positive-cell loss term before
    normalisation by n_pos. At this grid resolution (200x200, tens of
    positive cells per frame vs ~40,000 total), the naive CenterNet
    formula lets the negative term dominate the effective gradient even
    after the (1-gt)^beta down-weighting. A modest positive upweight
    (2-4x is standard in the literature) counteracts this without
    changing the loss's overall scale materially.

    norm: "sepnorm" divides the negative term by n_neg (original runs);
    "centernet" divides both terms by n_pos (standard CenterNet). With
    sepnorm a flat ~0.6 heatmap is loss-optimal at this grid size (see
    diagnostics/2026-11_heatmap_loss_and_depth).
    """
    pred = pred.float()   # force fp32: log() derivatives near the clamp
    gt = gt.float()       # boundaries overflow fp16 range under autocast
    pred = pred.clamp(eps, 1.0 - eps)
    pos = gt.eq(1.0).float()
    neg = (1.0 - pos)
    pos_loss = -((1.0 - pred) ** alpha) * torch.log(pred) * pos * pos_weight
    neg_loss = -((1.0 - gt) ** beta) * (pred ** alpha) * torch.log(1.0 - pred) * neg
    n_pos = pos.sum().clamp(min=1.0)
    if norm == "centernet":
        return (pos_loss.sum() + neg_loss.sum()) / n_pos
    n_neg = neg.sum().clamp(min=1.0)
    return pos_loss.sum() / n_pos + neg_loss.sum() / n_neg


def detection_loss(out: dict, gt: dict,
                   w_off: float = 1.0, w_sz: float = 1.0,
                   w_yaw: float = 1.0, w_zh: float = 0.5,
                   hm_norm: str = "sepnorm", hm_pos_weight: float = 4.0) -> tuple[torch.Tensor, dict]:
    """
    out: model outputs (heat, offset, size, yaw?, zh?)
    gt : dict with:
        heat   : (H, W)         Gaussian-target heatmap
        offset : (2, H, W)      sub-cell dx, dy
        size   : (2, H, W)      log w, log l
        yaw    : (2, H, W)      sin, cos          (optional)
        zh     : (2, H, W)      z_centre, height  (optional)
        mask   : (H, W)         1 at object centres, 0 elsewhere
    """
    L_hm = gaussian_focal_loss(out["heat"], gt["heat"], pos_weight=hm_pos_weight, norm=hm_norm)
    mask = gt["mask"].bool()
    L_reg = out["heat"].sum() * 0.0                         # zero tensor sharing device/dtype
    parts = {"hm": L_hm.detach().float().item()}

    if mask.any():
        n = mask.sum().float().clamp(min=1.0)
        L_off = F.smooth_l1_loss(out["offset"][:, mask], gt["offset"][:, mask], reduction="sum") / n
        L_sz  = F.smooth_l1_loss(out["size"][:, mask],   gt["size"][:, mask],   reduction="sum") / n
        L_reg = L_reg + w_off * L_off + w_sz * L_sz
        parts.update({"off": float(L_off), "sz": float(L_sz)})

        if "yaw" in out and "yaw" in gt:
            L_yaw = F.smooth_l1_loss(out["yaw"][:, mask], gt["yaw"][:, mask], reduction="sum") / n
            L_reg = L_reg + w_yaw * L_yaw
            parts["yaw"] = float(L_yaw)
        if "zh" in out and "zh" in gt:
            L_zh = F.smooth_l1_loss(out["zh"][:, mask], gt["zh"][:, mask], reduction="sum") / n
            L_reg = L_reg + w_zh * L_zh
            parts["zh"] = float(L_zh)

    total = L_hm + L_reg
    parts["total"] = float(total.detach())
    return total, parts
