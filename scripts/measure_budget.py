"""Does hard_topk actually respect the byte budget? (CPU)

hard_topk's final step is

    thresh = cell_util.topk(k=n_keep).values.min()
    cell_mask = (cell_util >= thresh)

With ties at the cut, `>=` keeps every cell sharing that utility. If fewer than n_keep
cells have non-zero utility the n_keep-th largest IS zero, so `cell_util >= 0` is true
everywhere and the budget is ignored completely.

Reports per frame, for a trained cooperative run:
  kept spatial cells   vs  budget_frac * H * W
  kept (agent, cell)   vs  budget_frac * H * W * k   (n_cells, what bytes are charged on)
  the threshold value, and how many cells tie with it

  python scripts/measure_budget.py --run ours_nodepth60 --frames 60
"""
from __future__ import annotations
import argparse, json, os, sys
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config, RESULTS_ROOT
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.metrics import bytes_kb_from_out, _to_device

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ours_nodepth60")
    ap.add_argument("--ckpt", default="last.pth")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--impl", default="prod", choices=["prod", "old"],
                    help="'old' monkeypatches the pre-fix `>=` threshold selection, to measure the "
                         "behaviour the H1/H2 analyses actually ran under")
    ap.add_argument("--out", default="/mnt/scratch2/users/2641309/tmp_claude_diag/budget_ties.json")
    a = ap.parse_args()

    if a.impl == "old":
        # The pre-fix selection, restored verbatim so the two can be compared on the same
        # frames. Patched before the model is built so the forward path picks it up.
        import src.models.topk_scheduler as sched

        def old_hard_topk(U, k, budget_cells, tau=0.0):
            N, h, w = U.shape
            k = min(int(k), N)
            _, idx = U.topk(k=k, dim=0)
            M = torch.zeros_like(U); M.scatter_(0, idx, 1.0)
            if tau > 0.0:
                M = M * (U > tau).float()
            cu = (U * M).sum(dim=0).flatten()
            n_total = h * w
            n_keep = max(0, min(budget_cells, n_total))
            if n_keep < n_total:
                if n_keep == 0:
                    return torch.zeros_like(U)
                thresh = cu.topk(k=n_keep).values.min()
                M = M * (cu >= thresh).float().view(1, h, w)
            return M

        sched.hard_topk = old_hard_topk
        print("USING THE PRE-FIX `>=` THRESHOLD SELECTION")

    cfg = replace(Config.load(f"{RESULTS_ROOT}/{a.run}/config.json"), device="cpu", augment=False)
    ck = torch.load(f"{RESULTS_ROOT}/checkpoints/{a.run}/{a.ckpt}", map_location="cpu", weights_only=False)
    model = CoopDetector(cfg); model.load_state_dict(ck["model"]); model.eval()
    ds = OPV2VDataset(cfg.val_dir, replace(cfg, depth_source="none"), max_agents=cfg.max_agents)
    idxs = np.linspace(0, len(ds) - 1, a.frames).astype(int).tolist()

    HW = cfg.grid_H * cfg.grid_W
    budget_cells = int(cfg.budget_frac * HW)
    print(f"{a.run} epoch {ck['epoch']}  grid {cfg.grid_H}x{cfg.grid_W}={HW}  "
          f"budget_frac={cfg.budget_frac} -> budget_cells={budget_cells}  k={cfg.k}")
    print(f"charged budget (agent,cell) pairs = budget_cells * k = {budget_cells * cfg.k}\n")

    rows = []
    # capture the utility the scheduler actually sees, via the same forward
    with torch.no_grad():
        for n, i in enumerate(idxs):
            scene = _to_device(opv2v_collate([ds[int(i)]]), torch.device("cpu"))
            out = model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau,
                        budget_frac=cfg.budget_frac)
            M = out["M"]                                     # (N, H, W), ego row zeroed
            M_others = M[1:]
            kept_pairs = int(M_others.sum().item())
            kept_cells = int((M_others.sum(dim=0) > 0).sum().item())
            U = out["U"]
            N = U.shape[0]
            # reconstruct cell_util exactly as hard_topk does, from the same tensors
            U_others = U[1:]
            cu = (U_others * M_others).sum(dim=0).flatten()
            n_keep = min(budget_cells, HW)
            thresh = float(cu.topk(k=n_keep).values.min()) if n_keep > 0 else 0.0
            n_tied = int((cu == thresh).sum().item())
            n_nonzero = int((cu > 0).sum().item())
            kb = bytes_kb_from_out(out, cfg)
            rows.append({"frame": int(i), "agents": N, "kept_cells": kept_cells,
                         "kept_pairs": kept_pairs, "thresh": thresh, "n_tied": n_tied,
                         "n_nonzero_util": n_nonzero, "kb": kb})
            if (n + 1) % 10 == 0:
                print(f"  {n+1}/{len(idxs)} frames", flush=True)

    kc = np.array([r["kept_cells"] for r in rows])
    kp = np.array([r["kept_pairs"] for r in rows])
    th = np.array([r["thresh"] for r in rows])
    tied = np.array([r["n_tied"] for r in rows])
    nz = np.array([r["n_nonzero_util"] for r in rows])
    kb = np.array([r["kb"] for r in rows])

    print(f"\n{len(rows)} frames")
    print(f"kept spatial cells     mean {kc.mean():9.1f}  max {kc.max():7d}   budget {budget_cells}   "
          f"over budget in {(kc > budget_cells).mean()*100:.0f}% of frames  "
          f"(max overshoot {kc.max()/budget_cells:.2f}x)")
    print(f"kept (agent,cell)      mean {kp.mean():9.1f}  max {kp.max():7d}   charged budget "
          f"{budget_cells*cfg.k}   over in {(kp > budget_cells*cfg.k).mean()*100:.0f}% of frames "
          f"(max {kp.max()/(budget_cells*cfg.k):.2f}x)")
    print(f"threshold value        mean {th.mean():.3e}  zero in {(th == 0).mean()*100:.0f}% of frames")
    print(f"cells tied at thresh   mean {tied.mean():9.1f}  max {tied.max():7d}  (of {HW})")
    print(f"cells with util > 0    mean {nz.mean():9.1f}  min {nz.min():7d}  "
          f"-- budget is unenforceable whenever this is < {budget_cells}: "
          f"{(nz < budget_cells).mean()*100:.0f}% of frames")
    print(f"charged bytes          mean {kb.mean():9.1f} KB  max {kb.max():.1f} KB")

    R = {"run": a.run, "epoch": int(ck["epoch"]), "frames": len(rows),
         "grid": HW, "budget_frac": cfg.budget_frac, "budget_cells": budget_cells, "k": cfg.k,
         "kept_cells": {"mean": float(kc.mean()), "max": int(kc.max())},
         "kept_pairs": {"mean": float(kp.mean()), "max": int(kp.max())},
         "frac_frames_over_cell_budget": float((kc > budget_cells).mean()),
         "frac_frames_thresh_zero": float((th == 0).mean()),
         "frac_frames_budget_unenforceable": float((nz < budget_cells).mean()),
         "kb": {"mean": float(kb.mean()), "max": float(kb.max())},
         "per_frame": rows}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
