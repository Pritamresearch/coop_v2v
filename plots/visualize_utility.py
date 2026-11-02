"""
Utility-map visualization (mirrors JigsawComm Fig. 3: "meta utility map").

For one scene, shows -- per collaborator -- the chain that N1/N2/N3 compute:

    U_i          : raw per-agent utility (before warp)              [N1]
    U_i (warped) : same, warped into the ego BEV frame               [geometry]
    U_r_i        : receiver-conditioned utility (x ego 'need')       [N2]
    M_i          : final selection mask (which cells actually ship)  [N3]

Plus one aggregate panel: how many agents were selected per cell
(sum_i M_i), which is the direct visual evidence for whether top-k > 1
is doing anything (if this panel is uniformly <=1, top-k collapses to
top-1 in practice).

This requires re-running the model's internals rather than just its
public forward() output, because forward() does not expose the
PRE-warp utility. We recompute the warp/utility chain here explicitly
(same functions the model itself calls), so what you see is exactly
what the model computed -- not an approximation.
"""
from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.models import topk_scheduler as sched


@torch.no_grad()
def _recompute_utility_chain(model, scene, device):
    """
    Re-run backbone + utility + warp + receiver-conditioning, exactly as
    CoopDetector.forward does, but return every intermediate map.
    """
    cfg = model.cfg
    F_all, depth_logits, extra = model.backbone(scene)
    U_all, D_all = model.utility(F_all, depth_logits)          # (N,H,W) each
    N = F_all.shape[0]

    rel_poses = scene["rel_pose"].to(device)
    F_ego = F_all[0]
    U_warp_list = []
    for i in range(N):
        U_warp_list.append(sched.warp_to_ego(U_all[i], rel_poses[i], cfg))
    U_warped = torch.stack(U_warp_list, dim=0)

    ego_vis = model._ego_visibility(F_ego)
    need = 1.0 - ego_vis
    U_r = sched.receiver_utility(U_warped, need, cfg.receiver_centric)

    # hard selection (inference-style) for a clean, binary mask panel
    U_r_others = U_r[1:]
    k_eff = min(cfg.k, N - 1) if N > 1 else 1
    n_keep = int(cfg.budget_frac * cfg.grid_H * cfg.grid_W)
    M_others = sched.hard_topk(U_r_others, k=k_eff, budget_cells=n_keep, tau=cfg.tau)
    M = torch.zeros_like(U_r)
    M[1:] = M_others

    return {
        "U_raw": U_all.cpu().numpy(),        # (N,H,W) before warp
        "U_warped": U_warped.cpu().numpy(),   # (N,H,W) after warp, before receiver cond.
        "U_r": U_r.cpu().numpy(),             # (N,H,W) receiver-conditioned
        "D": D_all.cpu().numpy(),             # (N,H,W) depth uncertainty
        "M": M.cpu().numpy(),                 # (N,H,W) final selection
        "need": need.cpu().numpy(),           # (H,W) ego need map
    }


def render_utility_maps(model, scene, save_path: str, max_collab: int = 3):
    """
    One figure, rows = collaborators (excluding ego, up to max_collab),
    columns = [raw U, warped U, receiver-conditioned U_r, selected mask M].
    A final row shows the ego 'need' map and the aggregated per-cell agent count.
    """
    device = next(model.parameters()).device
    chain = _recompute_utility_chain(model, scene, device)
    N = chain["U_raw"].shape[0]
    n_collab = min(N - 1, max_collab)

    cols = ["raw utility U (own frame)", "warped to ego frame",
            "receiver-conditioned U_r", "selected cells M"]
    n_rows = n_collab + 1
    fig, axs = plt.subplots(n_rows, 4, figsize=(4 * 4, 3.2 * n_rows))
    if n_rows == 1:
        axs = axs.reshape(1, -1)

    for r in range(n_collab):
        i = r + 1   # collaborator index (skip ego=0)
        panels = [chain["U_raw"][i], chain["U_warped"][i], chain["U_r"][i], chain["M"][i]]
        cmaps = ["viridis", "viridis", "magma", "gray"]
        for c in range(4):
            im = axs[r, c].imshow(panels[c], cmap=cmaps[c], origin="lower")
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
            if r == 0:
                axs[r, c].set_title(cols[c], fontsize=10)
            if c == 0:
                axs[r, c].set_ylabel(f"agent {i}", fontsize=10)
            plt.colorbar(im, ax=axs[r, c], fraction=0.046, pad=0.04)

    # aggregate row: ego need map + per-cell agent count (sum of M over collaborators)
    agent_count = chain["M"][1:].sum(axis=0)     # (H,W), how many agents cover each cell
    last = n_collab
    im0 = axs[last, 0].imshow(chain["need"], cmap="cividis", origin="lower")
    axs[last, 0].set_title("ego 'need' map (1 - ego_vis)", fontsize=10)
    axs[last, 0].set_xticks([]); axs[last, 0].set_yticks([])
    plt.colorbar(im0, ax=axs[last, 0], fraction=0.046, pad=0.04)

    im1 = axs[last, 1].imshow(agent_count, cmap="plasma", origin="lower",
                              vmin=0, vmax=max(1, agent_count.max()))
    axs[last, 1].set_title("agents selected per cell (sum M)", fontsize=10)
    axs[last, 1].set_xticks([]); axs[last, 1].set_yticks([])
    plt.colorbar(im1, ax=axs[last, 1], fraction=0.046, pad=0.04)

    axs[last, 2].axis("off")
    axs[last, 3].axis("off")
    frac_gt1 = float((agent_count > 1).mean())
    axs[last, 2].text(0.0, 0.5,
                      f"fraction of cells with\n>1 agent selected:\n{frac_gt1:.1%}",
                      fontsize=11, va="center")

    fig.suptitle("Utility chain (N1 -> N2 -> N3 selection)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
