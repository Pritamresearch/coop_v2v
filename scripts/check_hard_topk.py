"""Gate: hard_topk must respect the byte budget exactly, including under ties (CPU).

Calls the PRODUCTION src.models.topk_scheduler.hard_topk. The negative control is the
previous `>=` threshold implementation, written out here only so it can be shown to fail.

Cases:
  1. all utilities tied at 0      -- the pathological case: budget was ignored entirely
  2. all utilities tied at a positive value
  3. mostly zeros, a few positive -- fewer non-zero cells than the budget
  4. all utilities distinct       -- the fixed and old versions must AGREE (no regression)
  5. real ours_nodepth60 val frames -- kept cells <= budget on every frame
  6. determinism                  -- identical mask when called twice
"""
from __future__ import annotations
import os, sys
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config, RESULTS_ROOT
from src.models.topk_scheduler import hard_topk

H = W = 64
HW = H * W
BUDGET = int(0.4 * HW)          # 1638 of 4096


def old_hard_topk(U, k, budget_cells, tau=0.0):
    """The previous implementation: threshold on the n_keep-th largest value."""
    N, h, w = U.shape
    k = min(int(k), N)
    _, idx = U.topk(k=k, dim=0)
    M = torch.zeros_like(U); M.scatter_(0, idx, 1.0)
    if tau > 0.0:
        M = M * (U > tau).float()
    cell_util = (U * M).sum(dim=0).flatten()
    n_total = h * w
    n_keep = max(0, min(budget_cells, n_total))
    if n_keep < n_total:
        if n_keep == 0:
            return torch.zeros_like(U)
        thresh = cell_util.topk(k=n_keep).values.min()
        M = M * (cell_util >= thresh).float().view(1, h, w)
    return M


def kept_cells(M):
    return int((M.sum(dim=0) > 0).sum().item())


def case(name, U, fn, k=2, budget=BUDGET, expect_exact=True):
    M = fn(U, k=k, budget_cells=budget)
    kc = kept_cells(M)
    ok = (kc == budget) if expect_exact else (kc <= budget)
    print(f"    {name:34s} kept {kc:5d} / budget {budget:5d}  "
          f"{'exactly' if kc == budget else ('within' if kc <= budget else f'OVER by {kc-budget}')}")
    return ok


def synthetic(fn, label):
    print(f"  [{label}]")
    r = {}
    r["all tied at 0"] = case("all utilities tied at 0", torch.zeros(3, H, W), fn)
    r["all tied positive"] = case("all utilities tied at 0.5", torch.full((3, H, W), 0.5), fn)
    U = torch.zeros(3, H, W); U[1, :4, :10] = torch.rand(4, 10) + 0.1   # 40 positive cells
    r["few non-zero"] = case("40 non-zero cells, budget 1638", U, fn)
    g = torch.rand(3, H, W)                                            # distinct values
    r["distinct"] = case("all utilities distinct", g, fn)
    return r, g


def main() -> int:
    print("[real code: production hard_topk]")
    real, g = synthetic(hard_topk, "synthetic tie cases")

    # no-regression: with distinct utilities the fixed and old versions pick the same cells
    a = hard_topk(g, k=2, budget_cells=BUDGET)
    b = old_hard_topk(g, k=2, budget_cells=BUDGET)
    same = bool(torch.equal(a, b))
    print(f"    {'distinct utilities: fixed == old':34s} {same}")
    real["no regression on distinct"] = same

    # determinism
    real["deterministic"] = bool(torch.equal(hard_topk(g, k=2, budget_cells=BUDGET),
                                             hard_topk(g, k=2, budget_cells=BUDGET)))
    print(f"    {'called twice: identical':34s} {real['deterministic']}")

    # real data
    print("  [real ours_nodepth60 val frames]")
    try:
        from src.data.opv2v import OPV2VDataset, opv2v_collate
        from src.models.coop_detector import CoopDetector
        from src.metrics import _to_device
        cfg = replace(Config.load(f"{RESULTS_ROOT}/ours_nodepth60/config.json"),
                      device="cpu", augment=False)
        ck = torch.load(f"{RESULTS_ROOT}/checkpoints/ours_nodepth60/last.pth",
                        map_location="cpu", weights_only=False)
        m = CoopDetector(cfg); m.load_state_dict(ck["model"]); m.eval()
        ds = OPV2VDataset(cfg.val_dir, replace(cfg, depth_source="none"), max_agents=cfg.max_agents)
        bc = int(cfg.budget_frac * cfg.grid_H * cfg.grid_W)
        over = []
        with torch.no_grad():
            for i in np.linspace(0, len(ds) - 1, 4).astype(int):
                scene = _to_device(opv2v_collate([ds[int(i)]]), torch.device("cpu"))
                out = m(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau,
                        budget_frac=cfg.budget_frac)
                kc = kept_cells(out["M"][1:])
                over.append(kc - bc)
                print(f"    frame {int(i):5d}: kept {kc:6d} / budget {bc:6d}  "
                      f"{'OK' if kc <= bc else 'OVER'}")
        real["real frames within budget"] = all(o <= 0 for o in over)
    except Exception as e:                                    # noqa: BLE001
        print(f"    real-frame check unavailable: {type(e).__name__}: {e}")
        real["real frames within budget"] = False

    for k, v in real.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    print("\n[negative control: the previous `>=` threshold version -- MUST fail the tie cases]")
    bad, _ = synthetic(old_hard_topk, "old implementation")
    tie_cases = ["all tied at 0", "all tied positive", "few non-zero"]
    controls_fail = not any(bad[c] for c in tie_cases)
    for c in tie_cases:
        print(f"  broken {c}: {'correctly FAILS' if not bad[c] else 'WRONGLY PASSES'}")

    overall = all(real.values()) and controls_fail
    print(f"\nreal: {real}")
    print(f"overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
