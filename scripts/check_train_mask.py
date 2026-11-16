"""Is the TRAINING-path transmit mask soft (leaky) or straight-through? (CPU)

Two separate questions, often confused:

  (a) Does forward multiply features by a SOFT mask, letting every cell leak in with a
      small weight?  gumbel_topk_soft returns `hard + (soft*k - (soft*k).detach())`,
      whose forward value is exactly `hard` because the bracket cancels. Checked here by
      measuring the mask's values on real frames, not by reading the source.

  (b) Is the BYTE BUDGET applied during training at all?  gumbel_topk_soft does top-k
      over the AGENT axis only ("at most k agents per cell"). The cell budget lives in
      hard_topk, which runs only when hard=True.

Reports per frame: sum(M_others), the inference budget n_keep*k, and the fraction of
mask entries that are neither 0 nor 1.
"""
from __future__ import annotations
import argparse, os, sys
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config, RESULTS_ROOT
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.metrics import _to_device

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ours_nodepth60")
    ap.add_argument("--frames", type=int, default=8)
    a = ap.parse_args()

    cfg = replace(Config.load(f"{RESULTS_ROOT}/{a.run}/config.json"), device="cpu", augment=False)
    ck = torch.load(f"{RESULTS_ROOT}/checkpoints/{a.run}/last.pth", map_location="cpu",
                    weights_only=False)
    m = CoopDetector(cfg); m.load_state_dict(ck["model"]); m.eval()
    ds = OPV2VDataset(cfg.val_dir, replace(cfg, depth_source="none"), max_agents=cfg.max_agents)
    idxs = np.linspace(0, len(ds) - 1, a.frames).astype(int).tolist()

    HW = cfg.grid_H * cfg.grid_W
    n_keep = int(cfg.budget_frac * HW)
    charged = n_keep * cfg.k
    print(f"{a.run} epoch {ck['epoch']}  grid {HW}  k={cfg.k}  budget_frac={cfg.budget_frac}")
    print(f"inference budget: n_keep={n_keep} cells, n_keep*k={charged} (agent,cell) pairs\n")
    print(f"{'frame':>6s} {'agents':>6s} | {'TRAIN sum(M)':>13s} {'frac not 0/1':>13s} "
          f"{'vs budget':>10s} | {'EVAL sum(M)':>12s} {'vs budget':>10s}")

    tr_sums, ev_sums, frac_soft = [], [], []
    with torch.no_grad():
        for i in idxs:
            scene = _to_device(opv2v_collate([ds[int(i)]]), torch.device("cpu"))
            # training path: hard=False, Gumbel temperature as used at the END of training
            out_tr = m(scene, hard=False, gumbel_T=cfg.gumbel_T1, tau=cfg.tau,
                       budget_frac=cfg.budget_frac)
            out_ev = m(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau,
                       budget_frac=cfg.budget_frac)
            Mtr, Mev = out_tr["M"][1:], out_ev["M"][1:]
            s_tr, s_ev = float(Mtr.sum()), float(Mev.sum())
            soft = float(((Mtr != 0) & (Mtr != 1)).float().mean())
            tr_sums.append(s_tr); ev_sums.append(s_ev); frac_soft.append(soft)
            print(f"{int(i):>6d} {out_tr['U'].shape[0]:>6d} | {s_tr:>13.1f} {soft:>13.4f} "
                  f"{s_tr/charged:>9.2f}x | {s_ev:>12.1f} {s_ev/charged:>9.2f}x")

    tr, ev, fs = np.array(tr_sums), np.array(ev_sums), np.array(frac_soft)
    print(f"\n(a) mask values: fraction neither 0 nor 1 = {fs.mean():.6f} "
          f"-> {'SOFT, features leak' if fs.mean() > 1e-9 else 'STRAIGHT-THROUGH, binary forward'}")
    print(f"(b) training sum(M) mean {tr.mean():.1f} = {tr.mean()/charged:.2f}x the inference "
          f"budget of {charged}; k*H*W would be {cfg.k*HW}")
    print(f"    eval     sum(M) mean {ev.mean():.1f} = {ev.mean()/charged:.2f}x")
    print("\n    The training path applies top-k over the AGENT axis only; the cell budget is"
          "\n    enforced solely in hard_topk (hard=True). Bandwidth during training is therefore"
          "\n    shaped only by the lambda*bytes penalty, not capped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
