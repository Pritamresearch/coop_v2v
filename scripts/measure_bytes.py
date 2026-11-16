"""Measured transmitted bytes under the FIXED accounting (CPU).

Reports, over N unaugmented val frames, what a trained cooperative model actually
transmits: total KB per frame, split into feature-cell payload and the dense
per-sender control map, plus the sender count it scales with.

Bytes come from the production src.metrics.bytes_kb_from_out and the model's own
differentiable soft_B -- never re-derived here. The two are cross-checked per frame;
any disagreement is reported, because they agreeing is what the byte gate asserts.

Pre-2026-11-10 bandwidth numbers under-report: the control map was counted once
regardless of agent count and was missing from soft_B entirely.

  python scripts/measure_bytes.py --run ours_nodepth60 --frames 200
"""
from __future__ import annotations
import argparse, json, os, sys, time
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
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--budget-min", type=float, default=120.0)
    ap.add_argument("--out", default="/mnt/scratch2/users/2641309/tmp_claude_diag/bytes_measured.json")
    a = ap.parse_args()

    t0 = time.time()
    cfg = replace(Config.load(f"{RESULTS_ROOT}/{a.run}/config.json"), device="cpu", augment=False)
    ck = torch.load(f"{RESULTS_ROOT}/checkpoints/{a.run}/{a.ckpt}", map_location="cpu", weights_only=False)
    model = CoopDetector(cfg); model.load_state_dict(ck["model"]); model.eval()
    ds = OPV2VDataset(cfg.val_dir, replace(cfg, depth_source="none"), max_agents=cfg.max_agents)
    idx = np.linspace(0, len(ds) - 1, a.frames).astype(int).tolist()

    ctrl_per_sender_kb = cfg.grid_H * cfg.grid_W * cfg.b_util / 1024.0
    print(f"{a.run} epoch {ck['epoch']}  budget_frac={cfg.budget_frac}  per_cell_bytes={cfg.per_cell_bytes} "
          f"(idx {cfg.b_idx} + feat {cfg.b_feat} + util {cfg.b_util})")
    print(f"control map per sender = {cfg.grid_H}x{cfg.grid_W}x{cfg.b_util} B = {ctrl_per_sender_kb:.1f} KB; "
          f"count_control_bytes={getattr(cfg,'count_control_bytes',True)}\n")

    kb, cells, senders, mismatch = [], [], [], 0
    with torch.no_grad():
        for k, i in enumerate(idx):
            if time.time() - t0 > a.budget_min * 60:
                print(f"  TIME BUDGET hit at frame {k}"); break
            scene = _to_device(opv2v_collate([ds[int(i)]]), torch.device("cpu"))
            out = model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)
            k_kb = bytes_kb_from_out(out, cfg)                     # production accounting
            s_kb = float(out["soft_B"]) / 1024.0                   # model's own differentiable figure
            if abs(k_kb - s_kb) > 1e-3:
                mismatch += 1
                if mismatch <= 3:
                    print(f"  MISMATCH frame {i}: bytes_kb={k_kb:.3f} vs soft_B/1024={s_kb:.3f}")
            kb.append(k_kb); cells.append(int(out.get("n_cells", 0)))
            senders.append(int(out.get("n_senders", 0)))
            if (k + 1) % 25 == 0:
                print(f"  {k+1} frames, {(time.time()-t0)/(k+1):.1f}s each", flush=True)

    kb, cells, senders = np.array(kb), np.array(cells), np.array(senders)
    feat_kb = cells * cfg.per_cell_bytes / 1024.0
    ctrl_kb = senders * ctrl_per_sender_kb
    R = {"run": a.run, "epoch": int(ck["epoch"]), "frames": int(len(kb)),
         "budget_frac": cfg.budget_frac, "per_cell_bytes": cfg.per_cell_bytes,
         "soft_B_vs_bytes_kb_mismatches": int(mismatch),
         "total_kb": {"mean": float(kb.mean()), "median": float(np.median(kb)),
                      "p95": float(np.percentile(kb, 95)), "min": float(kb.min()), "max": float(kb.max())},
         "feature_kb_mean": float(feat_kb.mean()), "control_kb_mean": float(ctrl_kb.mean()),
         "control_share": float(ctrl_kb.mean() / max(kb.mean(), 1e-9)),
         "cells_mean": float(cells.mean()),
         "senders": {int(s): int((senders == s).sum()) for s in np.unique(senders)}}

    print(f"\nframes={len(kb)}  soft_B vs bytes_kb mismatches: {mismatch}")
    print(f"total        mean {kb.mean():8.1f} KB   median {np.median(kb):8.1f}   p95 {np.percentile(kb,95):8.1f}   "
          f"min {kb.min():.1f}  max {kb.max():.1f}")
    print(f"  feature payload mean {feat_kb.mean():8.1f} KB  ({cells.mean():.0f} cells x {cfg.per_cell_bytes} B)")
    print(f"  control maps    mean {ctrl_kb.mean():8.1f} KB  ({R['control_share']*100:.1f}% of the total)")
    print(f"senders per frame: {R['senders']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
