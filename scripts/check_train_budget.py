"""Gate: training-time selection must send exactly n_keep cells for the sampled budget.

Real ours_nodepth60 val frames, training path (hard=False).

  real code      cfg.train_budget=True  -> cells sent == n_keep for each sampled budget_frac,
                 mask binary in the forward pass, and gradients still flow
  neg control    cfg.train_budget=False -> the current no-budget path sends k*H*W, which
                 must FAIL the same check
"""
from __future__ import annotations
import os, sys
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config, RESULTS_ROOT
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.metrics import _to_device

FRACS = (0.1, 0.25, 0.4, 0.6)


def run(model, cfg, ds, idxs, label):
    HW = cfg.grid_H * cfg.grid_W
    ok = True
    print(f"  [{label}]")
    for bf in FRACS:
        n_keep = int(bf * HW)
        sums, cells, nonbin = [], [], []
        for i in idxs:
            scene = _to_device(opv2v_collate([ds[int(i)]]), torch.device("cpu"))
            with torch.no_grad():
                out = model(scene, hard=False, gumbel_T=cfg.gumbel_T1, budget_frac=bf)
            Mo = out["M"][1:]
            sums.append(float(Mo.sum()))
            cells.append(int((Mo.sum(dim=0) > 0).sum().item()))
            nonbin.append(float(((Mo != 0) & (Mo != 1)).float().mean()))
        c = np.array(cells)
        exact = bool((c == n_keep).all())
        print(f"    budget_frac={bf:<5} n_keep={n_keep:6d}  cells sent {c.min()}-{c.max()}  "
              f"pairs mean {np.mean(sums):9.1f}  non-binary {np.mean(nonbin):.6f}  "
              f"{'EXACT' if exact else 'MISMATCH'}")
        ok = ok and exact and np.mean(nonbin) == 0.0
    return ok


def grad_flows(cfg, model, ds, i) -> bool:
    scene = _to_device(opv2v_collate([ds[int(i)]]), torch.device("cpu"))
    out = model(scene, hard=False, gumbel_T=cfg.gumbel_T1, budget_frac=0.25)
    loss = out["heat"].sum() + out["soft_B"]
    loss.backward()
    g = [p.grad for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    print(f"    parameters receiving non-zero gradient: {len(g)}")
    return len(g) > 0


def main() -> int:
    base = Config.load(f"{RESULTS_ROOT}/ours_nodepth60/config.json")
    ck = torch.load(f"{RESULTS_ROOT}/checkpoints/ours_nodepth60/last.pth",
                    map_location="cpu", weights_only=False)

    cfg_on = replace(base, device="cpu", augment=False, train_budget=True)
    cfg_off = replace(base, device="cpu", augment=False, train_budget=False)
    ds = OPV2VDataset(cfg_on.val_dir, replace(cfg_on, depth_source="none"),
                      max_agents=cfg_on.max_agents)
    idxs = np.linspace(0, len(ds) - 1, 3).astype(int).tolist()

    m_on = CoopDetector(cfg_on); m_on.load_state_dict(ck["model"]); m_on.eval()
    m_off = CoopDetector(cfg_off); m_off.load_state_dict(ck["model"]); m_off.eval()

    print("[real code: train_budget=True]")
    real = run(m_on, cfg_on, ds, idxs, "budget enforced in training")
    print("  gradient check:")
    grads = grad_flows(cfg_on, m_on, ds, idxs[0])

    print("\n[negative control: train_budget=False, the current no-budget path -- MUST fail]")
    bad = run(m_off, cfg_off, ds, idxs, "no budget in training")

    overall = real and grads and not bad
    print(f"\nreal exact-budget: {real}   gradients flow: {grads}   "
          f"control {'correctly FAILS' if not bad else 'WRONGLY PASSES'}")
    print(f"overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
