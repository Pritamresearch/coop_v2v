"""
detection_sepnorm.py -- separate pos/neg normalization variant of the
heatmap focal loss, per:
  "Three-dimensional Backbone Network for 3D Object Detection in Traffic
  Scenes" (arxiv 1901.08373), Eq. 2:
      L_cls = -(1/N_pos) * sum_pos (...)  -  (1/N_neg) * sum_neg (...)
  and independently reported as a fix for the same collapse-to-zero failure
  mode in xingyizhou/CenterNet issue #670.

Standard CenterNet divides BOTH pos and neg terms by n_pos alone, which at
this grid's imbalance (~10 positive cells vs ~40,000 total) inflates the
negative term's effective per-cell weight by ~4000x. This variant removes
that asymmetry by normalizing each term by its own population instead of
a hand-tuned pos_weight multiplier.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def gaussian_focal_loss(pred: torch.Tensor, gt: torch.Tensor,
                        alpha: float = 2.0, beta: float = 4.0,
                        eps: float = 1e-4) -> torch.Tensor:
    pred = pred.float().clamp(eps, 1.0 - eps)
    gt = gt.float()
    pos = gt.eq(1.0).float()
    neg = (1.0 - pos)
    pos_loss = -((1.0 - pred) ** alpha) * torch.log(pred) * pos
    neg_loss = -((1.0 - gt) ** beta) * (pred ** alpha) * torch.log(1.0 - pred) * neg
    n_pos = pos.sum().clamp(min=1.0)
    n_neg = neg.sum().clamp(min=1.0)
    return pos_loss.sum() / n_pos + neg_loss.sum() / n_neg


def detection_loss(out: dict, gt: dict,
                   w_off: float = 1.0, w_sz: float = 1.0,
                   w_yaw: float = 1.0, w_zh: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Identical structure to src/loss/detection.py's detection_loss,
    just calling this file's separate-normalization gaussian_focal_loss."""
    L_hm = gaussian_focal_loss(out["heat"], gt["heat"])
    mask = gt["mask"].bool()
    L_reg = out["heat"].sum() * 0.0
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
