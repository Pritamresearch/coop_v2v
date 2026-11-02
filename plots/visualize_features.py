"""
Sparse-feature visualization (mirrors JigsawComm Fig. 4: dense vs. sparsified
feature maps before/after the semantic threshold).

Shows, for one collaborator and the ego:

    ||F_collab||        : dense per-cell feature magnitude BEFORE selection
    ||F_collab * M||     : same, masked by the selection -- the cells actually
                           transmitted (everything else is zeroed, i.e. never sent)
    ||F_ego||            : ego's own dense feature (for reference)
    ||F_star||           : fused feature actually used by the detection head

The point of this figure is qualitative evidence that the utility gate
concentrates transmitted mass on informative regions rather than sending
a uniformly-sampled subset.
"""
from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models import topk_scheduler as sched


@torch.no_grad()
def _recompute_feature_chain(model, scene, device):
    cfg = model.cfg
    F_all, depth_logits, extra = model.backbone(scene)
    U_all, D_all = model.utility(F_all, depth_logits)
    N = F_all.shape[0]

    rel_poses = scene["rel_pose"].to(device)
    F_ego = F_all[0]
    F_warp_list, U_warp_list = [], []
    for i in range(N):
        F_warp_list.append(sched.warp_to_ego(F_all[i], rel_poses[i], cfg))
        U_warp_list.append(sched.warp_to_ego(U_all[i], rel_poses[i], cfg))
    F_warped = torch.stack(F_warp_list, dim=0)
    U_warped = torch.stack(U_warp_list, dim=0)

    ego_vis = model._ego_visibility(F_ego)
    need = 1.0 - ego_vis
    U_r = sched.receiver_utility(U_warped, need, cfg.receiver_centric)

    U_r_others = U_r[1:]
    k_eff = min(cfg.k, N - 1) if N > 1 else 1
    n_keep = int(cfg.budget_frac * cfg.grid_H * cfg.grid_W)
    M_others = sched.hard_topk(U_r_others, k=k_eff, budget_cells=n_keep, tau=cfg.tau)
    M = torch.zeros_like(U_r)
    M[1:] = M_others

    F_send = F_warped * M.unsqueeze(1)
    F_star = model.fusion(F_ego, F_send, U_r, M)

    return {
        "F_ego_mag": F_ego.norm(dim=0).cpu().numpy(),
        "F_collab_mag": F_warped[1].norm(dim=0).cpu().numpy() if N > 1 else None,
        "F_collab_sparse_mag": F_send[1].norm(dim=0).cpu().numpy() if N > 1 else None,
        "F_star_mag": F_star.norm(dim=0).cpu().numpy(),
        "mask": M[1].cpu().numpy() if N > 1 else None,
    }


def render_feature_sparsity(model, scene, save_path: str):
    device = next(model.parameters()).device
    ch = _recompute_feature_chain(model, scene, device)

    has_collab = ch["F_collab_mag"] is not None
    n_panels = 4 if has_collab else 2
    fig, axs = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.2))

    panels = [("||F_ego|| (dense)", ch["F_ego_mag"])]
    if has_collab:
        panels += [
            ("||F_collab|| (dense, pre-selection)", ch["F_collab_mag"]),
            ("||F_collab * M|| (transmitted only)", ch["F_collab_sparse_mag"]),
        ]
    panels += [("||F*|| (fused, used by head)", ch["F_star_mag"])]

    vmax = max(p[1].max() for p in panels)
    for ax, (title, mag) in zip(axs, panels):
        im = ax.imshow(mag, cmap="inferno", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if has_collab:
        n_sent = int((ch["mask"] > 0.5).sum())
        n_total = ch["mask"].size
        fig.suptitle(f"Sparsification: {n_sent}/{n_total} cells transmitted "
                    f"({n_sent/n_total:.1%})", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
