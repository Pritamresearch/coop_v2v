"""
Full co-designed model.

Data flow (one scene, N agents, ego = agent 0):

    (N, 4 cams, 3, H, W) images
        -> LSSBackbone
    -> F  (N, C, Hb, Wb)   depth_logits (N, K, Hb, Wb)
        -> UtilityHead
    -> U_i (N, Hb, Wb)                 D (N, Hb, Wb)
        -> warp collaborators to ego frame + receiver 'need' weighting
    -> U_{i->r} (N, Hb, Wb)   F_i^ego (N, C, Hb, Wb)
        -> top-k scheduler (soft in train, hard+budget in inference)
    -> M (N, Hb, Wb)  in {0,1}
        -> 4-bit quantised sparse encoding (Module B)
    -> F' (received on the ego side; bytes counted)
        -> Utility-gated fusion
    -> F* (C, Hb, Wb)
        -> Multi-branch detection head
    -> heat/off/size/yaw/height  =>  3D boxes

Ego is agent 0 BY CONVENTION.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .depth_aware_utility import UtilityHead
from .gated_fusion import UtilityGatedFusion
from . import topk_scheduler as sched


# ------------------------------------------------------------------
# 4-bit quantiser with straight-through estimator (Module B)
# ------------------------------------------------------------------
def quantise_ste(x: torch.Tensor, n_bits: int = 4) -> torch.Tensor:
    """
    Symmetric per-tensor quantisation with STE gradient.
    Values are clipped to [-s, s] where s = max |x|, then rounded to 2^n_bits levels.
    """
    if x.numel() == 0:
        return x
    s = x.abs().max().clamp(min=1e-6)
    levels = (1 << (n_bits - 1)) - 1                # 4-bit symmetric -> [-7, 7]
    y = torch.round(x / s * levels) / levels * s
    # STE: forward = quantised, backward = identity
    return x + (y - x).detach()


# ------------------------------------------------------------------
# Detection head (CenterPoint-style, multi-branch)
# ------------------------------------------------------------------
class DetHead(nn.Module):
    """
    Shared 3x3 conv tower + 5 branches:
        heat   : 1 channel  -- objectness heatmap (sigmoid)
        offset : 2 channels -- sub-cell (dx, dy)
        size   : 2 channels -- log(w), log(l)
        yaw    : 2 channels -- (sin, cos) [optional]
        z_h    : 2 channels -- z_centre, height [optional]
    """
    def __init__(self, C: int, predict_yaw: bool = True, predict_z: bool = True):
        super().__init__()
        self.predict_yaw = predict_yaw
        self.predict_z = predict_z
        self.tower = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),
            nn.GroupNorm(32, C),
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, padding=1),
            nn.GroupNorm(32, C),
            nn.ReLU(inplace=True),
        )
        self.heat = nn.Conv2d(C, 1, 1)
        self.offset = nn.Conv2d(C, 2, 1)
        self.size = nn.Conv2d(C, 2, 1)
        if predict_yaw:
            self.yaw = nn.Conv2d(C, 2, 1)
        if predict_z:
            self.zh = nn.Conv2d(C, 2, 1)
        # bias-init the heatmap head so early sigmoid is small (focal loss stability)
        prior = 0.1
        self.heat.bias.data.fill_(-math.log((1 - prior) / prior))
        # init size head to predict typical vehicle: w~2m, l~4.5m
        self.size.bias.data[0] = math.log(2.0)   # log(w)
        self.size.bias.data[1] = math.log(4.5)   # log(l)

    def forward(self, F_star: torch.Tensor) -> dict:
        h = self.tower(F_star.unsqueeze(0))                  # (1, C, H, W)
        out = {
            "heat":   torch.sigmoid(self.heat(h)).squeeze(0).squeeze(0),   # (H, W)
            "offset": self.offset(h).squeeze(0),                            # (2, H, W)
            "size":   self.size(h).squeeze(0),                              # (2, H, W)  log w, log l
        }
        if self.predict_yaw:
            out["yaw"] = self.yaw(h).squeeze(0)                             # (2, H, W)  sin, cos
        if self.predict_z:
            out["zh"] = self.zh(h).squeeze(0)                               # (2, H, W)  z, h
        return out


# ------------------------------------------------------------------
# Full model
# ------------------------------------------------------------------
class CoopDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.utility = UtilityHead(C=cfg.feat_C, utility_type=cfg.utility_type, alpha=cfg.alpha)
        self.fusion = UtilityGatedFusion(C=cfg.feat_C, use_utility_gate=cfg.use_utility_gate,
                                         fusion_type=getattr(cfg, 'fusion_type', 'attention'))
        self.head = DetHead(C=cfg.feat_C, predict_yaw=cfg.predict_yaw, predict_z=cfg.predict_z)

        # Optionally freeze backbone + utility for ablation A6 (separate training)
        if not cfg.joint_train:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for p in self.utility.parameters():
                p.requires_grad_(False)

    # --------------------------------------------------------------
    # helpers
    # --------------------------------------------------------------
    def _ego_visibility(self, D_ego: torch.Tensor) -> torch.Tensor:
        """
        Ego 'see' map based on depth certainty: cells where the ego has
        confident depth estimates are considered 'seen' (ego_vis=1),
        cells with uncertain depth need collaborator input (ego_vis=0).
        D_ego: (H, W) depth uncertainty in [0,1], 0=certain, 1=uniform
        Returns ego_vis in [0,1]: 1=ego sees well, 0=ego needs help
        """
        # ego_vis = 1 - D (certain depth -> ego sees well)
        # Threshold: cells with D < 0.5 are considered visible
        return (1.0 - D_ego).clamp(0.0, 1.0)


    def _request_map(self, F_all: torch.Tensor, D_all: torch.Tensor) -> torch.Tensor:
        """Receiver 'need' map: high where the ego wants a collaborator's help.

        cfg.need_source:
          "depth"      -- 1 - ego_vis, from the ego's depth uncertainty (ours).
          "confidence" -- 1 - ego confidence (Where2comm-faithful). Deliberately independent of
                          depth, so a confidence-vs-depth comparison is not contaminated by our
                          depth-based need leaking through the request map.

        forward() and scripts/check_baseline_components.py both call THIS method, so the gate
        tests the real code path rather than a reimplementation of it.
        """
        if getattr(self.cfg, "need_source", "depth") == "confidence":
            return 1.0 - torch.sigmoid(self.utility.conf_head(F_all[0:1])).squeeze(1).squeeze(0)
        return 1.0 - self._ego_visibility(D_all[0])

    def _budget_cells(self) -> int:
        return int(self.cfg.budget_frac * self.cfg.grid_H * self.cfg.grid_W)

    # --------------------------------------------------------------
    # forward
    # --------------------------------------------------------------
    def forward(self, scene: dict, hard: bool = False, gumbel_T: float = 1.0,
                tau: float | None = None, budget_frac: float | None = None):
        """
        scene: dict as produced by the OPV2V loader (see data/opv2v.py contract).
               Ego is scene entry 0; collaborators are 1..N-1.
        hard : if True use deterministic hard top-k + budget (inference);
               if False use differentiable soft selection (training).
        """
        cfg = self.cfg

        # 1) Per-agent backbone
        F_all, depth_logits, extra = self.backbone(scene)    # F: (N, C, H, W)
        N = F_all.shape[0]

        # 2) Per-agent utility + depth uncertainty
        U_all, D_all = self.utility(F_all, depth_logits)     # (N, H, W), (N, H, W)
        # If no fusion baseline: ego-only. Send zero bytes.
        if cfg.no_fusion or N == 1:
            F_star = F_all[0]
            out = self.head(F_star)
            out.update({
                "F": F_all, "U": U_all, "D": D_all,
                "depth_logits": depth_logits,
                "depth_img": extra.get("depth_img"),
                "M": torch.zeros_like(U_all),
                "soft_B": torch.tensor(0.0, device=F_all.device),
                "n_cells": 0,
                "F_star": F_star,
            })
            return out

        # 3) Warp all agents (incl. ego) into ego BEV frame
        rel_poses = scene["rel_pose"].to(F_all.device)       # (N, 3)  ego row is zero
        F_ego = F_all[0]                                     # already in ego frame
        F_recv_list, U_warp_list = [], []
        for i in range(N):
            F_recv_list.append(sched.warp_to_ego(F_all[i], rel_poses[i], cfg))
            U_warp_list.append(sched.warp_to_ego(U_all[i], rel_poses[i], cfg))
        F_warped = torch.stack(F_recv_list, dim=0)           # (N, C, H, W)
        U_warped = torch.stack(U_warp_list, dim=0)           # (N, H, W)
        # 4) Receiver-centric 'need' from the ego
        # Depth-uncertainty-based ego visibility:
        # cells where ego has confident depth are "seen"; uncertain cells need help
        need = self._request_map(F_all, D_all)
        U_r = sched.receiver_utility(U_warped, need, cfg.receiver_centric)
        # 5) Selection
        #    Ego is agent 0; it does not "transmit" to itself. Exclude row 0 from selection
        #    but keep it in the F_warped stack (fusion queries against it).
        U_r_others = U_r[1:]                                 # (N-1, H, W)
        if cfg.dense_transmit:
            M_others = torch.ones_like(U_r_others)
        elif getattr(cfg, "selection_mode", "topk") == "threshold":
            # CoCa3D-faithful: every cell above a fixed utility threshold is sent, so the byte
            # cost VARIES per frame instead of being capped. The threshold is calibrated on val
            # (scripts/calibrate_threshold.py) to hit a target MEAN rate.
            M_others = (U_r_others > cfg.utility_thresh).float()
        elif hard:
            n_keep = self._budget_cells() if budget_frac is None else \
                     int((budget_frac if budget_frac is not None else cfg.budget_frac)
                         * cfg.grid_H * cfg.grid_W)
            k_eff = min(cfg.k, N - 1) if N > 1 else 1
            M_others = sched.hard_topk(U_r_others, k=k_eff,
                                       budget_cells=n_keep,
                                       tau=(tau if tau is not None else cfg.tau))
        else:
            k_eff = min(cfg.k, N - 1) if N > 1 else 1
            if cfg.selection_relax == "true_topk":
                M_others = sched.gumbel_topk_soft(U_r_others, k=k_eff, T=gumbel_T)
            else:
                M_others = sched.gumbel_sigmoid(U_r_others, tau=cfg.tau, T=gumbel_T)

        # Full mask (ego row = 0: ego does not transmit to itself)
        M = torch.zeros_like(U_r)
        M[1:] = M_others
        # 6) Encoder (Module B): sparsify + 4-bit quantise
        F_send = F_warped * M.unsqueeze(1)                   # (N, C, H, W)
        if cfg.use_encoder:
            F_send = quantise_ste(F_send, n_bits=4)
        # 7) Utility-gated fusion (ego queries against warped collaborators)
        F_star = self.fusion(F_ego, F_send, U_r, M)
        # 8) Detection head
        out = self.head(F_star)
        # 9) Bandwidth accounting (differentiable via M)
        #    Expected bytes = per_cell_bytes * sum(M_others)  (soft in train, exact in infer)
        # Bytes = selected cells + the dense control map each collaborator sends (its utility /
        # confidence map). The control map was previously counted only in metrics.bytes_kb_from_out
        # and not in this differentiable term, and it was not scaled by agent count.
        n_senders = max(N - 1, 0)
        control_B = (float(cfg.grid_H * cfg.grid_W * cfg.b_util) * n_senders
                     if getattr(cfg, "count_control_bytes", True) else 0.0)
        soft_B = M_others.float().sum() * float(cfg.per_cell_bytes) + control_B
        n_cells = int(M_others.sum().item()) if hard else int(M_others.detach().float().sum().item())

        out.update({
            "F": F_all, "U": U_all, "D": D_all,
            "depth_logits": depth_logits,
            "depth_img": extra.get("depth_img"),
            "M": M,
            "soft_B": soft_B,
            "n_cells": n_cells,
            "n_senders": n_senders,
            "F_star": F_star,
        })
        return out
