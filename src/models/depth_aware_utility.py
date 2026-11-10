"""
Module A, Step 1 -- Depth-Aware Utility (Novelty N1).

Per-cell utility:
    U(h,w) = ||F(h,w)||_2  *  exp(-alpha * D(h,w))

    * ||F||_2 collapses the C-length feature vector to one magnitude number
      ("something is strongly here").
    * D in [0,1] is per-cell normalised entropy of the camera's own
      categorical depth distribution -- the SIGNAL nobody has used before.
      0 = certain depth (keep full strength), 1 = flat/uncertain (suppress).
    * alpha is a LEARNABLE scalar; a large alpha means "trust only very
      certain cells" and a small alpha means "ignore depth".

Four variants are provided for ablation A2:
    depth_aware      : ||F||_2 * exp(-alpha * D)         (ours, N1)
    depth_entropy    : 1 - D                             (CoCa3D-style, sender-side)
    confidence_only  : sigmoid(conv(F))                  (Where2comm style)
    magnitude_only   : ||F||_2
    learned_fue      : relu(conv([F, D]))                (learned feature-utility)

All variants return U in shape (N, H, W).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def depth_uncertainty(depth_logits: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Normalised categorical entropy per BEV cell.

    depth_logits: (N, K, H, W)   -- log-probabilities (or unnormalised logits).
    returns    D: (N, H, W)      -- in [0,1], 0 = certain, 1 = uniform.
    """
    # If input is already log-probs (LSSBackbone hands us that), softmax is
    # numerically fine either way -- torch handles the shift.
    p = F.softmax(depth_logits, dim=1)                  # (N, K, H, W)
    K = p.size(1)
    logK = torch.log(torch.tensor(float(K), device=p.device))
    H = -(p * (p + eps).log()).sum(dim=1) / logK        # (N, H, W)
    return H.clamp(0.0, 1.0)


class UtilityHead(nn.Module):
    def __init__(self, C: int, utility_type: str = "depth_aware", alpha: float = 4.0):
        super().__init__()
        self.utility_type = utility_type
        # alpha is learnable regardless of the variant (harmless if unused)
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        # Convs used by the confidence / learned variants (small, cheap)
        self.conf_head = nn.Conv2d(C, 1, 1)
        self.fue_head = nn.Sequential(
            nn.Conv2d(C + 1, C // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, 1, 1),
        )

    def forward(self, F_bev: torch.Tensor, depth_logits: torch.Tensor):
        """
        F_bev        : (N, C, H, W)
        depth_logits : (N, K, H, W)
        returns:
            U : (N, H, W)  per-cell utility, non-negative
            D : (N, H, W)  depth uncertainty (always returned so downstream
                           modules can use it for logging / evaluation)
        """
        D = depth_uncertainty(depth_logits)
        mag = F_bev.norm(dim=1)                              # (N, H, W)

        if self.utility_type == "magnitude_only":
            U = mag
        elif self.utility_type == "confidence_only":
            U = torch.sigmoid(self.conf_head(F_bev)).squeeze(1)          # (N, H, W)
        elif self.utility_type == "learned_fue":
            cat = torch.cat([F_bev, D.unsqueeze(1)], dim=1)              # (N, C+1, H, W)
            U = torch.relu(self.fue_head(cat)).squeeze(1)
        elif self.utility_type == "depth_entropy":
            # CoCa3D-style: utility is the depth distribution's CONFIDENCE alone, with no
            # feature-magnitude term. CoCa3D selects LOW entropy, so higher utility = lower D.
            U = 1.0 - D
        elif self.utility_type == "depth_aware":
            # softplus(alpha) so alpha stays non-negative during training
            a = torch.nn.functional.softplus(self.alpha)
            U = mag * torch.exp(-a * D)
        else:
            raise ValueError(self.utility_type)

        return U, D
