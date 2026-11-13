"""Pre-registered epoch-20 image-augmentation readout (CPU).

Compares three arms on the SAME, unaugmented val frames (paired), all at epoch 20:

  ab_depth_reweight   baseline: LiDAR depth, arm-B depth loss, NO image augmentation
  ab_imgaug_rot180    + BEVDet image aug (resize/crop/flip/rotate), BEV rotation +-180 deg
  ab_imgaug_rot22     + BEVDet image aug, BEV rotation narrowed to +-22.5 deg (BEVDet's range)

PRE-REGISTERED DECISION RULE (fixed before any arm's epoch-20 numbers were seen).
An arm WINS only if ALL THREE hold against the baseline:
  (1) val 0-20 m AP50 improves AND the frame-bootstrap 95% CI of the paired difference excludes 0
  (2) val 0-20 m centre AP@2m improves AND its CI excludes 0
  (3) the train/val AP50 gap at 0-20 m is reduced (overfitting goes down, not up)
Anything else is a FAIL -- including "better but CI touches 0", which is the outcome this
rule exists to catch.

Matching is done by the production Evaluator (src.metrics), one frame at a time, so the
per-frame (score, tp) records that feed the bootstrap are exactly what the training-time
metric would produce.  Centre-distance AP uses the same greedy nearest-centre rule as the
earlier per-band readouts.

Evaluation is always unaugmented: load_model sets augment=False, and the dataset gates image
augmentation on `self._img_aug and self.augment`, so the aug arms are scored on clean images.

  python scripts/imgaug_readout.py --frames 200 --bootstrap 2000 --out readout.json
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
from src.metrics import decode, Evaluator, _ap_from_matches, _to_device

BASELINE = "ab_depth_reweight"
ARMS     = ["ab_imgaug_rot180", "ab_imgaug_rot22"]
BAND     = (0.0, 20.0)       # the pre-registered primary band
CD_RADII = (1.0, 2.0, 4.0)

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))


def load_model(run: str, epoch: int = 20):
    ckpt = f"{RESULTS_ROOT}/checkpoints/{run}/epoch_{epoch}.pth"
    cfg = replace(Config.load(f"{RESULTS_ROOT}/{run}/config.json"), device="cpu", augment=False)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = CoopDetector(cfg); m.load_state_dict(ck["model"]); m.eval()
    return cfg, m, int(ck["epoch"])


def in_band(x: np.ndarray) -> np.ndarray:
    r = np.hypot(x[:, 0], x[:, 1])
    return (r >= BAND[0]) & (r < BAND[1])


def frame_records(cfg, model, ds, i):
    """Per-frame match records inside the primary band, via the production Evaluator."""
    scene = _to_device(opv2v_collate([ds[int(i)]]), torch.device("cpu"))
    with torch.no_grad():
        out = model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)
    b, s = decode(out, cfg, score_thresh=cfg.eval_score_thresh, topk=200)
    g, oof = scene["gt"]["boxes"].numpy(), scene["gt"]["oof"].numpy()

    kg = in_band(g); g, oof = g[kg], oof[kg]
    if len(b):
        kd = in_band(b); b, s = b[kd], s[kd]

    ev = Evaluator(cfg)                      # production matching, one frame
    ev.update(b, s, g, oof, kb=0.0)
    rec = {"n_gt": int(len(g)),
           "iou50": (list(map(float, ev.scores[0.5])), list(map(int, ev.tp[0.5]))),
           "iou30": (list(map(float, ev.scores[0.3])), list(map(int, ev.tp[0.3])))}

    for rad in CD_RADII:                     # greedy nearest-centre, as in the per-band readouts
        sc, tp = [], []
        used = np.zeros(len(g), bool)
        for k in np.argsort(-s) if len(s) else []:
            j = -1
            if len(g):
                dd = np.where(used, np.inf, np.linalg.norm(g[:, :2] - b[k, :2], axis=1))
                j = int(dd.argmin())
                if dd[j] > rad: j = -1
            if j >= 0: used[j] = True
            sc.append(float(s[k])); tp.append(int(j >= 0))
        rec[f"cd{rad:g}"] = (sc, tp)
    return rec


def ap_over(frames: list, key: str) -> float:
    sc, tp, n = [], [], 0
    for f in frames:
        s_, t_ = f[key]; sc += s_; tp += t_; n += f["n_gt"]
    return _ap_from_matches(sc, tp, n)


def bootstrap_diff(arm: list, base: list, key: str, B: int, rng) -> dict:
    """Paired frame-level bootstrap of (arm - baseline) for one metric."""
    n = len(arm)
    point = ap_over(arm, key) - ap_over(base, key)
    diffs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)                       # same frames for both arms: paired
        diffs[b] = ap_over([arm[i] for i in idx], key) - ap_over([base[i] for i in idx], key)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0), "p_gt_0": float((diffs > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--epoch", type=int, default=20)
    ap.add_argument("--budget-min", type=float, default=150.0)
    ap.add_argument("--out", default="/mnt/scratch2/users/2641309/tmp_claude_diag/imgaug_readout_ep20.json")
    a = ap.parse_args()

    t0 = time.time()
    runs = [BASELINE] + ARMS
    rec: dict = {r: {"val": [], "train": []} for r in runs}
    epochs = {}

    for run in runs:
        cfg, model, ep = load_model(run, a.epoch)
        epochs[run] = ep
        ds = {"val":   OPV2VDataset(cfg.val_dir,   replace(cfg, depth_source="none"), max_agents=1),
              "train": OPV2VDataset(cfg.train_dir, replace(cfg, depth_source="none"), max_agents=1)}
        idx = {s: np.linspace(0, len(d) - 1, a.frames).astype(int).tolist() for s, d in ds.items()}
        print(f"\n=== {run}  epoch {ep}  img_aug={getattr(cfg,'img_aug',False)} "
              f"bev_rot={getattr(cfg,'bev_rot_deg',None)} (eval augment={cfg.augment}) ===", flush=True)
        for k in range(a.frames):
            if time.time() - t0 > a.budget_min * 60:
                print(f"  TIME BUDGET hit at frame {k}", flush=True); break
            for s in ("val", "train"):
                rec[run][s].append(frame_records(cfg, model, ds[s], idx[s][k]))
            if (k + 1) % 25 == 0:
                print(f"  {k+1}/{a.frames} frames, {(time.time()-t0)/(k+1):.1f}s per frame-pair", flush=True)

    n = min(len(rec[r]["val"]) for r in runs)
    for r in runs:
        rec[r]["val"] = rec[r]["val"][:n]; rec[r]["train"] = rec[r]["train"][:n]
    print(f"\nPaired on {n} val frames and {n} train frames, unaugmented.\n")

    if epochs[BASELINE] != a.epoch or any(epochs[r] != a.epoch for r in ARMS):
        print(f"WARNING: checkpoint epochs are {epochs}, expected {a.epoch} for all arms")

    rng = np.random.default_rng(0)
    R = {"frames": n, "bootstrap": a.bootstrap, "epochs": epochs, "band": "0-20 m", "arms": {}}

    base_v, base_t = rec[BASELINE]["val"], rec[BASELINE]["train"]
    b_ap50, b_cd2 = ap_over(base_v, "iou50"), ap_over(base_v, "cd2")
    b_gap = ap_over(base_t, "iou50") - b_ap50
    R["baseline"] = {"run": BASELINE, "val_ap50": b_ap50, "val_cdAP2m": b_cd2,
                     "train_ap50": ap_over(base_t, "iou50"), "gap": b_gap,
                     "val_ap30": ap_over(base_v, "iou30"), "val_cdAP1m": ap_over(base_v, "cd1"),
                     "val_cdAP4m": ap_over(base_v, "cd4")}
    print(f"[baseline] {BASELINE} ep{epochs[BASELINE]}: val AP50={b_ap50:.4f}  centre AP@2m={b_cd2:.4f}  "
          f"train AP50={R['baseline']['train_ap50']:.4f}  gap={b_gap:+.4f}")

    for arm in ARMS:
        av, at = rec[arm]["val"], rec[arm]["train"]
        a_ap50, a_cd2 = ap_over(av, "iou50"), ap_over(av, "cd2")
        a_gap = ap_over(at, "iou50") - a_ap50
        c1 = bootstrap_diff(av, base_v, "iou50", a.bootstrap, rng)
        c2 = bootstrap_diff(av, base_v, "cd2",   a.bootstrap, rng)
        crit = {"ap50_up_ci_excl_0":  bool(c1["diff"] > 0 and c1["excludes_zero"]),
                "cdAP2m_up_ci_excl_0": bool(c2["diff"] > 0 and c2["excludes_zero"]),
                "gap_reduced":         bool(a_gap < b_gap)}
        verdict = "WIN" if all(crit.values()) else "FAIL"
        R["arms"][arm] = {"val_ap50": a_ap50, "val_cdAP2m": a_cd2,
                          "train_ap50": ap_over(at, "iou50"), "gap": a_gap,
                          "val_ap30": ap_over(av, "iou30"), "val_cdAP1m": ap_over(av, "cd1"),
                          "val_cdAP4m": ap_over(av, "cd4"),
                          "ap50_vs_baseline": c1, "cdAP2m_vs_baseline": c2,
                          "criteria": crit, "verdict": verdict}
        print(f"\n[{arm}] ep{epochs[arm]}")
        print(f"   val AP50      {a_ap50:.4f}  (base {b_ap50:.4f})  diff {c1['diff']:+.4f}  "
              f"95% CI [{c1['ci_lo']:+.4f}, {c1['ci_hi']:+.4f}]  excludes 0: {c1['excludes_zero']}")
        print(f"   centre AP@2m  {a_cd2:.4f}  (base {b_cd2:.4f})  diff {c2['diff']:+.4f}  "
              f"95% CI [{c2['ci_lo']:+.4f}, {c2['ci_hi']:+.4f}]  excludes 0: {c2['excludes_zero']}")
        print(f"   train/val gap {a_gap:+.4f}  (base {b_gap:+.4f})  reduced: {crit['gap_reduced']}")
        print(f"   --> {verdict}   {crit}")

    wins = [r for r, v in R["arms"].items() if v["verdict"] == "WIN"]
    R["winner"] = wins[0] if len(wins) == 1 else (None if not wins else "AMBIGUOUS")
    print(f"\nPRE-REGISTERED OUTCOME: " +
          ("no arm passes -- keep the no-image-aug baseline" if not wins else
           f"winner = {R['winner']}" if len(wins) == 1 else f"both arms pass: {wins}"))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(R, f, indent=2)
    print(f"\nwrote {a.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
