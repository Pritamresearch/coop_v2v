"""
Joint co-design objective (Novelty N4):

    L = L_det  +  lambda * E[bytes]  +  gamma * L_depth

    * L_det    : detection loss (focal + smooth-L1).
    * E[bytes] : differentiable expected message size in bytes
                 (matches inference-time byte counting, so training and
                 inference agree on the axis of the accuracy-bandwidth curve).
    * L_depth  : per-pixel cross-entropy on the LSS depth bins; only applied
                 when depth ground truth is available. This is what teaches
                 the backbone to concentrate mass on the correct bin so
                 that D becomes meaningful.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F

from .detection import detection_loss


_BIN_WEIGHTS = {}


def _depth_bin_weights(cfg, device) -> torch.Tensor:
    """1/sqrt(bin frequency) per depth bin, normalised so the average weight over the training
    label distribution is 1. Frequencies come from scripts/depth_bin_frequency.py."""
    key = (cfg.depth_bins, cfg.depth_K, str(device))
    if key not in _BIN_WEIGHTS:
        import numpy as np, os
        path = os.path.join(os.path.dirname(__file__), f"depth_bin_freq_{cfg.depth_bins}{cfg.depth_K}.npy")
        f = np.load(path)
        w = 1.0 / np.sqrt(np.maximum(f, 1e-6))
        w = w / (f * w).sum()
        _BIN_WEIGHTS[key] = torch.tensor(w, dtype=torch.float32, device=device)
    return _BIN_WEIGHTS[key]


def _depth_ce_from_img(depth_img: torch.Tensor, depth_gt_img: torch.Tensor,
                       bin_weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    depth_img    : (N, n_cams, K, fH, fW)   -- per-pixel depth logits from LSS encoder
    depth_gt_img : (N, n_cams, fH, fW)      -- per-pixel index in {0..K-1}, -1 = ignore
    """
    N, n_cams, K, fH, fW = depth_img.shape
    logits = depth_img.reshape(-1, K, fH, fW)               # (N*n_cams, K, fH, fW)
    target = depth_gt_img.reshape(-1, fH, fW).long()         # (N*n_cams, fH, fW)
    if bin_weights is None:
        return F.cross_entropy(logits, target, ignore_index=-1)
    ce = F.cross_entropy(logits, target, ignore_index=-1, reduction="none")
    w = bin_weights[target.clamp(min=0)] * (target >= 0)
    return (w * ce).sum() / w.sum()


def coop_loss(out: dict, scene: dict, cfg, lam: float) -> tuple[torch.Tensor, dict]:
    """
    Combined objective. Returns (total_loss, parts_dict).
    'lam' is applied by the training schedule (warm-up sets it to 0).
    """
    L_det, det_parts = detection_loss(out, scene["gt"],
                                      hm_norm=getattr(cfg, "hm_norm", "sepnorm"),
                                      hm_pos_weight=getattr(cfg, "hm_pos_weight", 4.0))
    parts = {"det": det_parts["total"], **{f"det_{k}": v for k, v in det_parts.items()}}

    # bandwidth term (differentiable through Gumbel)
    E_B = out["soft_B"]                                     # scalar in bytes
    # Normalise by grid size so lambda has a comparable scale across configs
    E_B_norm = E_B / (cfg.grid_H * cfg.grid_W * cfg.per_cell_bytes)
    L_bw = E_B_norm
    parts["bw_norm"] = float(L_bw.detach())
    parts["bw_bytes"] = float(E_B.detach())

    # depth supervision
    L_depth = torch.zeros((), device=L_det.device, dtype=L_det.dtype)
    if out.get("depth_img") is not None and "depth_gt_img" in scene:
        if cfg.gamma_depth > 0.0:
            bw = (_depth_bin_weights(cfg, out["depth_img"].device)
                  if getattr(cfg, "depth_loss_weight", "none") == "inv_sqrt_freq" else None)
            L_depth = _depth_ce_from_img(out["depth_img"], scene["depth_gt_img"], bw)
    parts["depth"] = float(L_depth.detach())

    total = L_det + lam * L_bw + cfg.gamma_depth * L_depth
    parts["total"] = float(total.detach())
    return total, parts
