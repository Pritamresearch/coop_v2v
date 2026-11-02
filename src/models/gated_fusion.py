"""
Module C -- Utility-Gated Attention Fusion (Novelty N3, fusion half).

The SAME utility that drove selection now biases the attention:
    logits = (q . k) / sqrt(C)  +  beta * U_r
    a      = softmax over agents of logits
    F*     = F_ego + sum_i a_i * v_i

    * beta is a learnable scalar (gate strength).
    * cells an agent did NOT transmit are masked to -inf so softmax gives 0.
    * cells where NO agent transmitted fall back to F_ego (residual skip).

Ablation A1 control arm: use_utility_gate = False drops the beta * U term,
reducing to plain content-only attention. This is the decisive test for
whether utility gating reverses "more-is-worse".
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class UtilityGatedFusion(nn.Module):
    def __init__(self, C: int, use_utility_gate: bool = True, beta_init: float = 1.0):
        super().__init__()
        self.C = C
        self.use_utility_gate = use_utility_gate
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.q = nn.Conv2d(C, C, 1)
        self.k = nn.Conv2d(C, C, 1)
        self.v = nn.Conv2d(C, C, 1)
        self.out = nn.Conv2d(C, C, 1)

    def forward(self, F_ego: torch.Tensor, F_recv: torch.Tensor,
                U_r: torch.Tensor, mask: torch.Tensor):
        """
        F_ego  : (C, H, W)              ego BEV feature
        F_recv : (N, C, H, W)           collaborators warped into ego frame
        U_r    : (N, H, W)              receiver-conditioned utility per cell
        mask   : (N, H, W)              selection mask in {0, 1}

        returns:
            F_star : (C, H, W)          fused ego-frame feature
        """
        N, C, H, W = F_recv.shape
        # queries from ego (1, C, H, W), keys/values from collaborators (N, C, H, W)
        q = self.q(F_ego.unsqueeze(0))                        # (1, C, H, W)
        k = self.k(F_recv)                                    # (N, C, H, W)
        v = self.v(F_recv)                                    # (N, C, H, W)

        # content score: sum over channels, then / sqrt(C)
        score = (q * k).sum(dim=1) / math.sqrt(C)             # (N, H, W)

        if self.use_utility_gate:
            score = score + self.beta * U_r

        # mask non-transmitted cells to -inf so softmax ignores them
        neg_inf = torch.finfo(score.dtype).min
        score = score.masked_fill(mask < 0.5, neg_inf)

        # softmax over agents (dim=0). Cells with no agent get all -inf; handle below.
        att = torch.softmax(score, dim=0)                     # (N, H, W)
        att = torch.nan_to_num(att, nan=0.0, posinf=0.0, neginf=0.0)

        # empty-cell detection: no agent transmitted this cell
        no_agent = (mask.sum(dim=0, keepdim=True) < 0.5)      # (1, H, W)

        # weighted sum of values
        fused_contrib = (att.unsqueeze(1) * v).sum(dim=0)     # (C, H, W)
        fused_contrib = self.out(fused_contrib.unsqueeze(0)).squeeze(0)

        # residual + fallback: no-agent cells inherit ego feature only
        F_star = F_ego + fused_contrib
        F_star = torch.where(no_agent, F_ego, F_star)
        return F_star
