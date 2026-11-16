"""
Module A, Step 2 -- Receiver-Centric Top-K Scheduler.

Implements two novelties:
    N2  Receiver-centric conditioning:
        Each ego weights every collaborator's utility by its own 'need'
        map (where the ego is blind), so a cell is valuable only if it is
        strong, depth-certain, AND unseen by the ego.

    N3 (selection half)  Top-k per cell:
        Generalises JigsawComm's top-1 to top-k so up to k complementary
        partial views of a cell can be combined. Whether top-k > top-1
        is an empirical question decided by ablation A1 -- we do NOT
        assume it.

Selection is differentiable via TWO options (config flag):
    'true_topk' : perturbed softmax over agents per cell with straight-through
                  (matches diagram promise: exactly k per cell in the forward pass)
    'sigmoid'   : independent per-cell Gumbel-sigmoid (does NOT enforce exactly k)

Then a receiver-side greedy 'budget' step keeps only the highest-utility
cells until  per_cell_bytes * |M| <= budget_bytes.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# Warping
# ------------------------------------------------------------------
def warp_to_ego(x: torch.Tensor, rel_pose: torch.Tensor, cfg) -> torch.Tensor:
    """
    Warp a per-agent (H, W) or (C, H, W) map into the ego BEV frame.

    rel_pose: (3,) tensor  [dx, dy, dyaw]   -- translation (metres) + heading (rad)
                                              of agent i measured in the ego frame.
    Uses affine_grid + grid_sample. When rel_pose=0 (ego), this reduces to identity (correct).

    NOTE: for OPV2V the loader must produce rel_pose in ego-frame metres/radians.
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)             # (1,1,H,W)
        squeeze = "hw"
    elif x.dim() == 3:
        x = x.unsqueeze(0)                          # (1,C,H,W)
        squeeze = "chw"
    else:
        raise ValueError(f"unexpected shape {x.shape}")

    H, W = x.shape[-2], x.shape[-1]
    dx, dy, dyaw = rel_pose[0], rel_pose[1], rel_pose[2]
    # scale metres -> normalised grid coords ([-1, 1] spans the full grid)
    span_x = W * cfg.cell_m
    span_y = H * cfg.cell_m
    tx = 2.0 * dx / span_x
    ty = 2.0 * dy / span_y
    c, s = torch.cos(dyaw), torch.sin(dyaw)
    theta = torch.stack([
        torch.stack([ c,  s, -(c * tx + s * ty)]),
        torch.stack([-s,  c,  (s * tx - c * ty)]),
    ]).unsqueeze(0).to(x.dtype).to(x.device)         # (1, 2, 3)  -- inverse transform

    grid = F.affine_grid(theta, size=x.shape, align_corners=False)
    warped = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    if squeeze == "hw":
        return warped.squeeze(0).squeeze(0)
    return warped.squeeze(0)


# ------------------------------------------------------------------
# Receiver conditioning
# ------------------------------------------------------------------
def receiver_utility(U_warped: torch.Tensor, need: torch.Tensor, receiver_centric: bool) -> torch.Tensor:
    """
    U_warped : (N, H, W)  utilities already warped into ego frame
    need     : (H, W)     ego 'need' map (high where ego is blind)
    """
    if receiver_centric:
        return U_warped * need.unsqueeze(0)          # (N, H, W)
    return U_warped


# ------------------------------------------------------------------
# Selection
# ------------------------------------------------------------------
def _gumbel_noise(shape, device, dtype):
    u = torch.rand(shape, device=device, dtype=dtype).clamp(1e-9, 1.0 - 1e-9)
    return -torch.log(-torch.log(u))


def gumbel_topk_soft(U: torch.Tensor, k: int, T: float) -> torch.Tensor:
    """
    Differentiable per-cell top-k over the agent axis (dim=0).

    Uses perturbed softmax + straight-through:
        - forward pass: exactly k highest-utility agents per cell = 1, rest = 0
        - backward pass: gradient of a temperature-softmax over agents

    U : (N, H, W)
    returns M : (N, H, W)  in {0,1} forward, differentiable backward.
    """
    N = U.shape[0]
    k = min(int(k), N)

    g = _gumbel_noise(U.shape, device=U.device, dtype=U.dtype)
    scores = (U + g) / max(T, 1e-3)                  # (N, H, W)

    # soft: relative importance across agents at each cell
    soft = torch.softmax(scores, dim=0)              # sums to 1 across N per cell
    # hard: exactly k winners per cell (top-k along agent axis)
    _, topk_idx = scores.topk(k=k, dim=0)            # (k, H, W)
    hard = torch.zeros_like(scores)
    hard.scatter_(0, topk_idx, 1.0)                  # (N, H, W)

    # Straight-through: hard forward, soft backward.
    # Scale hard by k so soft and hard have similar magnitude (both sum ~ k across N).
    return hard + (soft * k - (soft * k).detach())


def gumbel_sigmoid(U: torch.Tensor, tau: float, T: float) -> torch.Tensor:
    """
    Independent per-cell Gumbel-sigmoid gate. Does NOT strictly enforce
    exactly k per cell -- provided as an ablation for comparison.
    """
    g1 = _gumbel_noise(U.shape, U.device, U.dtype)
    g2 = _gumbel_noise(U.shape, U.device, U.dtype)
    soft = torch.sigmoid(((U - tau) + (g1 - g2)) / max(T, 1e-3))
    hard = (soft > 0.5).float()
    return hard + (soft - soft.detach())


def hard_topk(U: torch.Tensor, k: int, budget_cells: int, tau: float = 0.0) -> torch.Tensor:
    """
    Deterministic inference-time selection.
        1. keep at most k agents per cell
        2. optional threshold tau on utility
        3. greedy budget: keep the highest-utility cells until |M| <= budget_cells

    U : (N, H, W)
    returns M : (N, H, W) in {0, 1}
    """
    N, H, W = U.shape
    k = min(int(k), N)

    _, topk_idx = U.topk(k=k, dim=0)                 # (k, H, W)
    M = torch.zeros_like(U)
    M.scatter_(0, topk_idx, 1.0)

    if tau > 0.0:
        M = M * (U > tau).float()

    # greedy budget: keep EXACTLY the n_keep highest-utility cells.
    #
    # This previously thresholded on the n_keep-th largest value:
    #     thresh = cell_util.topk(k=n_keep).values.min()
    #     cell_mask = (cell_util >= thresh)
    # which keeps every cell TIED with the threshold, so the budget was advisory. It
    # failed hardest exactly where it mattered: utility is ~0 on most covered cells
    # (BEV depth entropy is degenerate below 0.01 in 97% of them), so when fewer than
    # n_keep cells have non-zero utility the n_keep-th largest IS zero and
    # `cell_util >= 0` is true for the whole grid -- the full map was transmitted while
    # the accounting still reported a budgeted send.
    #
    # Selecting by index instead makes the count exact. argsort(stable=True) on the
    # negated utility breaks ties by ascending cell index, so the choice is deterministic
    # and reproducible rather than dependent on topk's tie order.
    cell_util = (U * M).sum(dim=0).flatten()         # (H*W,)
    n_total = H * W
    n_keep = max(0, min(budget_cells, n_total))
    if n_keep < n_total:
        if n_keep == 0:
            return torch.zeros_like(U)
        order = torch.argsort(-cell_util, stable=True)        # ties -> ascending index
        cell_mask = torch.zeros_like(cell_util)
        cell_mask[order[:n_keep]] = 1.0
        M = M * cell_mask.view(1, H, W)
    return M
