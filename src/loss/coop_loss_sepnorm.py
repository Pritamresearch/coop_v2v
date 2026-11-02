"""
coop_loss_sepnorm.py -- identical to src/loss/coop_loss.py, but wired to
the separate-normalization detection_loss (arm 3 of the pos/neg imbalance
experiment). See src/loss/detection_sepnorm.py for the change.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .detection_sepnorm import detection_loss


def _depth_ce_from_img(depth_img: torch.Tensor, depth_gt_img: torch.Tensor) -> torch.Tensor:
    N, n_cams, K, fH, fW = depth_img.shape
    logits = depth_img.reshape(-1, K, fH, fW)
    target = depth_gt_img.reshape(-1, fH, fW).long()
    return F.cross_entropy(logits, target, ignore_index=-1)


def coop_loss(out: dict, scene: dict, cfg, lam: float) -> tuple[torch.Tensor, dict]:
    L_det, det_parts = detection_loss(out, scene["gt"])
    parts = {"det": det_parts["total"], **{f"det_{k}": v for k, v in det_parts.items()}}

    E_B = out["soft_B"]
    E_B_norm = E_B / (cfg.grid_H * cfg.grid_W * cfg.per_cell_bytes)
    L_bw = E_B_norm
    parts["bw_norm"] = float(L_bw.detach())
    parts["bw_bytes"] = float(E_B.detach())

    L_depth = torch.zeros((), device=L_det.device, dtype=L_det.dtype)
    if out.get("depth_img") is not None and "depth_gt_img" in scene:
        if cfg.gamma_depth > 0.0:
            L_depth = _depth_ce_from_img(out["depth_img"], scene["depth_gt_img"])
    parts["depth"] = float(L_depth.detach())

    total = L_det + lam * L_bw + cfg.gamma_depth * L_depth
    parts["total"] = float(total.detach())
    return total, parts
