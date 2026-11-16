"""Evaluate a LIST of checkpoints on one split, through a single code path.

Registered in diagnostics/2026-11_cooperation/PREREG_FINAL_ROUND.md (Amendment 1):
every method reports the mean +/- sd over the checkpoints of epochs 55-59, for
validation and for the one frozen test evaluation. All of them are reported; none is
chosen. The same function computes both splits so the numbers are comparable.

  python scripts/eval_checkpoints.py --run ego_final --epochs 55 56 57 58 59 --split val
  COOP_TEST_FREEZE=1 python scripts/eval_checkpoints.py --run ego_final \
      --epochs 55 56 57 58 59 --split test

The test split is refused unless COOP_TEST_FREEZE=1, matching slurm/test.sh: one
evaluation, after every choice is frozen.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys, time
from dataclasses import replace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config, RESULTS_ROOT
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.metrics import decode, Evaluator, _ap_from_matches, bytes_kb_from_out, _to_device

METRICS = ("ap50", "ap30", "recall50", "cdAP2m", "kb")
FLOOR_DEFAULT = 0.013          # PREREG_FINAL_ROUND Amendment 1: broken-run detector,
                               # NOT a quality bar. Quality is judged against the
                               # reference band, mean ~0.026.


def split_dir(cfg: Config, split: str) -> str:
    """The one place a split maps to a directory. Test is gated here, not by the caller."""
    if split == "val":
        return cfg.val_dir
    if split == "test":
        if os.environ.get("COOP_TEST_FREEZE") != "1":
            raise SystemExit(
                "REFUSING to evaluate on the held-out test split.\n"
                "All method choices must be frozen first, then rerun with COOP_TEST_FREEZE=1.")
        return cfg.test_dir
    raise SystemExit(f"unknown split {split!r} (use val or test)")


def load(run: str, epoch: int, device: str = "cpu"):
    cfg = Config.load(f"{RESULTS_ROOT}/{run}/config.json")
    # Evaluation is never augmented, for any split or checkpoint.
    cfg = replace(cfg, device=device, augment=False, img_aug=False)
    ck = torch.load(f"{RESULTS_ROOT}/checkpoints/{run}/epoch_{epoch}.pth",
                    map_location=device, weights_only=False)
    m = CoopDetector(cfg); m.load_state_dict(ck["model"]); m.eval()
    return cfg, m, int(ck["epoch"])


def evaluate_one(cfg: Config, model, ds, idxs) -> dict:
    """The single evaluation path. Identical for val and test by construction."""
    ev = Evaluator(cfg)
    cd_sc, cd_tp, n_gt, kbs = [], [], 0, []
    with torch.no_grad():
        for i in idxs:
            scene = _to_device(opv2v_collate([ds[int(i)]]), torch.device(cfg.device))
            out = model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau,
                        budget_frac=cfg.budget_frac)
            b, s = decode(out, cfg, score_thresh=cfg.eval_score_thresh, topk=200)
            g, oof = scene["gt"]["boxes"].cpu().numpy(), scene["gt"]["oof"].cpu().numpy()
            ev.update(b, s, g, oof, kb=bytes_kb_from_out(out, cfg))
            kbs.append(bytes_kb_from_out(out, cfg))
            n_gt += len(g)
            used = np.zeros(len(g), bool)
            for k in np.argsort(-s) if len(s) else []:
                j = -1
                if len(g):
                    dd = np.where(used, np.inf, np.linalg.norm(g[:, :2] - b[k, :2], axis=1))
                    j = int(dd.argmin())
                    if dd[j] > 2.0: j = -1
                if j >= 0: used[j] = True
                cd_sc.append(float(s[k])); cd_tp.append(int(j >= 0))
    e = ev.compute()
    return {"ap50": e["ap50"], "ap30": e["ap30"], "recall50": e["recall50"],
            "cdAP2m": _ap_from_matches(cd_sc, cd_tp, n_gt),
            "kb": float(np.mean(kbs)) if kbs else 0.0}


def aggregate(per_ckpt: dict[int, dict]) -> dict:
    """Mean and sample sd across checkpoints. Reports every one; selects none."""
    out = {}
    for m in METRICS:
        vals = [per_ckpt[e][m] for e in sorted(per_ckpt)]
        out[m] = {"mean": float(statistics.fmean(vals)),
                  "sd": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                  "n": len(vals), "values": vals}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--epochs", type=int, nargs="+", default=[55, 56, 57, 58, 59])
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--frames", type=int, default=0, help="0 = the whole split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--floor", type=float, default=FLOOR_DEFAULT)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    t0 = time.time()
    cfg0, _, _ = load(a.run, a.epochs[0], a.device)
    d = split_dir(cfg0, a.split)
    ds = OPV2VDataset(d, replace(cfg0, depth_source="none"), max_agents=cfg0.max_agents)
    idxs = (np.linspace(0, len(ds) - 1, a.frames).astype(int).tolist()
            if a.frames else list(range(len(ds))))
    print(f"{a.run}  split={a.split}  {len(idxs)} frames  device={a.device}  "
          f"epochs={a.epochs}  (augment forced off)")

    per = {}
    for e in a.epochs:
        cfg, model, ep = load(a.run, e, a.device)
        assert ep == e, f"epoch_{e}.pth holds epoch {ep}"
        per[e] = evaluate_one(cfg, model, ds, idxs)
        print(f"  epoch {e}: " + "  ".join(f"{m}={per[e][m]:.4f}" for m in METRICS), flush=True)

    agg = aggregate(per)
    print(f"\n{a.run} {a.split}: mean +/- sd over {agg['ap50']['n']} checkpoints "
          f"(epochs {min(per)}-{max(per)}), no selection")
    for m in METRICS:
        print(f"  {m:9s} {agg[m]['mean']:.4f} +/- {agg[m]['sd']:.4f}")

    R = {"run": a.run, "split": a.split, "frames": len(idxs), "epochs": a.epochs,
         "per_checkpoint": per, "aggregate": agg}

    if a.split == "val":
        lo = agg["ap50"]["mean"]
        R["floor"] = {"value": a.floor, "mean_ap50": lo, "passed": bool(lo >= a.floor)}
        if lo < a.floor:
            print(f"\nVOID: val AP50 mean {lo:.4f} < floor {a.floor:.3f} -- the training regime was "
                  f"not reproduced. Investigate before reporting anything from this round.")
        else:
            print(f"\nfloor {a.floor:.3f} passed (mean AP50 {lo:.4f}). The floor is a broken-run "
                  f"detector, not a quality bar; quality is judged against the reference band "
                  f"(mean ~0.026).")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(R, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")
    print(f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
