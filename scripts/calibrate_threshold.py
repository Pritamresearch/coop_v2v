"""Calibrate the CoCa3D-style utility threshold on VAL to hit target MEAN byte rates.

    python scripts/calibrate_threshold.py <ckpt> <preset> [--frames 60] [--targets 0.1,0.2,0.4]

`selection_mode="threshold"` sends every cell whose receiver utility exceeds a fixed threshold, so
the byte cost VARIES per frame rather than being capped. That is faithful to CoCa3D, which sweeps a
threshold and has no byte budget. It is also the honest comparison: our fixed per-frame byte
guarantee is one of our own claims, so budget-matching the baseline would hand it our advantage.

For each target (expressed as a fraction of the grid, to match budget_frac), the threshold is found
by bisection on the MEAN selected-cell count over val frames. Reported per target:
  mean bytes with a frame-bootstrap CI, 95th-percentile bytes, and the fraction of frames whose
  byte count exceeds the equivalent fixed budget.
Those last two are the cost of giving up a hard guarantee, and belong in the paper.
"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace
import numpy as np, torch
from src.config import Config
from src.baselines import make
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.metrics import _to_device, bytes_kb_from_out


def utilities(model, cfg, scenes):
    """Receiver utilities for the collaborator rows, exactly as selection sees them."""
    out = []
    with torch.no_grad():
        for sc in scenes:
            o = model(sc, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau,
                      budget_frac=cfg.budget_frac)
            u = o["U_r_others"] if "U_r_others" in o else None
            if u is None:                      # not exported: recompute from the public pieces
                import src.models.topk_scheduler as sched
                F_all, depth_logits, _ = model.backbone(sc)
                U_all, D_all = model.utility(F_all, depth_logits)
                rel = sc["rel_pose"].to(F_all.device)
                U_w = torch.stack([sched.warp_to_ego(U_all[i], rel[i], cfg)
                                   for i in range(F_all.shape[0])], 0)
                if getattr(cfg, "need_source", "depth") == "confidence":
                    need = 1.0 - torch.sigmoid(model.utility.conf_head(F_all[0:1])).squeeze(1).squeeze(0)
                else:
                    need = 1.0 - (1.0 - D_all[0]).clamp(0, 1)
                u = sched.receiver_utility(U_w, need, cfg.receiver_centric)[1:]
            out.append(u.flatten().cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("preset")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--targets", default="0.1,0.2,0.4")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    base = replace(make(a.preset, Config()), device="cpu", augment=False, depth_source="none")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = CoopDetector(base); model.load_state_dict(ck["model"]); model.eval()
    ds = OPV2VDataset(base.val_dir, base, max_agents=base.max_agents, use_gt_pose=base.use_gt_pose)
    scenes, n_sel = [], []
    for i in np.linspace(0, len(ds) - 1, a.frames * 3).astype(int):
        s = opv2v_collate([ds[int(i)]])
        if int(s["n_agents"]) >= 2:
            scenes.append(_to_device(s, torch.device("cpu")))
        if len(scenes) == a.frames:
            break
    print(f"{len(scenes)} multi-agent val frames, checkpoint epoch {ck.get('epoch')}")

    U = utilities(model, base, scenes)
    senders = [max(int(s["n_agents"]) - 1, 0) for s in scenes]
    cells = base.grid_H * base.grid_W
    rng = np.random.default_rng(0)
    R = {"frames": len(scenes), "targets": {}}

    for tgt in [float(x) for x in a.targets.split(",")]:
        want = tgt * cells                       # mean selected cells per sender-row, per frame
        lo, hi = min(u.min() for u in U), max(u.max() for u in U)
        for _ in range(60):                      # bisection on the MEAN count
            mid = (lo + hi) / 2
            got = np.mean([np.mean((u > mid).sum() / max(len(u) / cells, 1)) for u in U])
            if got > want:
                lo = mid
            else:
                hi = mid
        thr = (lo + hi) / 2
        per_frame_kb, over = [], 0
        budget_kb = (tgt * cells * base.per_cell_bytes) / 1024.0
        for u, ns in zip(U, senders):
            n = int((u > thr).sum())
            kb = bytes_kb_from_out({"n_cells": n, "n_senders": ns}, base)
            per_frame_kb.append(kb)
            fixed = bytes_kb_from_out({"n_cells": int(tgt * cells), "n_senders": ns}, base)
            over += kb > fixed
        kb = np.array(per_frame_kb)
        b = [kb[rng.integers(0, len(kb), len(kb))].mean() for _ in range(2000)]
        R["targets"][f"{tgt}"] = {
            "threshold": float(thr), "mean_kb": float(kb.mean()),
            "ci": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
            "p95_kb": float(np.percentile(kb, 95)),
            "frac_frames_over_fixed_budget": float(over / len(kb)),
            "equivalent_fixed_budget_kb": float(budget_kb)}
        r = R["targets"][f"{tgt}"]
        print(f"  budget_frac {tgt:>5}: thresh {thr:+.5f}  mean {r['mean_kb']:7.1f} KB "
              f"[{r['ci'][0]:.1f}, {r['ci'][1]:.1f}]  p95 {r['p95_kb']:7.1f} KB  "
              f"frames over the equivalent fixed budget: {r['frac_frames_over_fixed_budget']:.3f}")
    out = a.out or f"threshold_calibration_{a.preset}.json"
    json.dump(R, open(out, "w"), indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
