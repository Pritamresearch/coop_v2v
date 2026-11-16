"""Gate: scripts/eval_checkpoints.py must behave as pre-registered.

Checks the PRODUCTION functions in eval_checkpoints (never re-derived copies), each
with a negative control that must fail:

  1. aggregate() reports mean +/- sample sd over EVERY checkpoint and selects none
  2. split_dir() maps val -> cfg.val_dir and test -> cfg.test_dir
  3. split_dir("test") REFUSES without COOP_TEST_FREEZE=1
  4. load() forces augmentation off, whatever the run's config says
  5. evaluate_one() is deterministic on real frames: the same checkpoint twice,
     byte-identical metrics (no augmentation or sampling leaking into evaluation)

Check 5 runs on real OPV2V val frames with a real trained checkpoint.
"""
from __future__ import annotations
import os, statistics, sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_checkpoints as E
from src.config import Config


def _fake(vals):
    return {e: {m: v for m in E.METRICS} for e, v in vals.items()}


def check_aggregate(agg_fn) -> bool:
    per = _fake({55: 0.010, 56: 0.020, 57: 0.030, 58: 0.040, 59: 0.050})
    a = agg_fn(per)["ap50"]
    want_mean = statistics.fmean([0.01, 0.02, 0.03, 0.04, 0.05])
    want_sd = statistics.stdev([0.01, 0.02, 0.03, 0.04, 0.05])
    ok = (abs(a["mean"] - want_mean) < 1e-12 and abs(a["sd"] - want_sd) < 1e-12
          and a["n"] == 5 and sorted(a["values"]) == [0.01, 0.02, 0.03, 0.04, 0.05])
    # a selecting aggregator would report the max as the headline
    ok = ok and abs(a["mean"] - max(a["values"])) > 1e-9
    print(f"    mean={a['mean']:.5f} (want {want_mean:.5f})  sd={a['sd']:.5f} (want {want_sd:.5f})  "
          f"n={a['n']}  all values kept={len(a['values'])==5}")
    return ok


def check_split(split_fn, guard: bool) -> bool:
    cfg = Config()
    ok = split_fn(cfg, "val") == cfg.val_dir
    os.environ.pop("COOP_TEST_FREEZE", None)
    refused = False
    try:
        split_fn(cfg, "test")
    except SystemExit:
        refused = True
    os.environ["COOP_TEST_FREEZE"] = "1"
    try:
        allowed = split_fn(cfg, "test") == cfg.test_dir
    except SystemExit:
        allowed = False
    os.environ.pop("COOP_TEST_FREEZE", None)
    print(f"    val->val_dir={ok}  test refused without the flag={refused}  "
          f"test allowed with it={allowed}")
    return bool(ok and allowed and (refused if guard else not refused))


def check_aug_off(load_fn, run: str, epoch: int) -> bool:
    cfg, _, _ = load_fn(run, epoch, "cpu")
    print(f"    loaded cfg: augment={cfg.augment}  img_aug={cfg.img_aug}")
    return (cfg.augment is False) and (cfg.img_aug is False)


def check_determinism(run: str, epoch: int, frames: int) -> bool:
    cfg, model, _ = E.load(run, epoch, "cpu")
    from src.data.opv2v import OPV2VDataset
    ds = OPV2VDataset(E.split_dir(cfg, "val"), replace(cfg, depth_source="none"),
                      max_agents=cfg.max_agents)
    idxs = np.linspace(0, len(ds) - 1, frames).astype(int).tolist()
    a = E.evaluate_one(cfg, model, ds, idxs)
    b = E.evaluate_one(cfg, model, ds, idxs)
    same = all(abs(a[m] - b[m]) < 1e-12 for m in E.METRICS)
    print(f"    run1 ap50={a['ap50']:.6f} cdAP2m={a['cdAP2m']:.6f} | "
          f"run2 ap50={b['ap50']:.6f} cdAP2m={b['cdAP2m']:.6f} | identical={same}")
    return same


def check_determinism_broken(run: str, epoch: int, frames: int) -> bool:
    """Negative control: leave BEV augmentation on, as a careless eval would."""
    cfg, model, _ = E.load(run, epoch, "cpu")
    cfg = replace(cfg, augment=True)
    from src.data.opv2v import OPV2VDataset
    ds = OPV2VDataset(cfg.val_dir, replace(cfg, depth_source="none"), max_agents=cfg.max_agents)
    idxs = np.linspace(0, len(ds) - 1, frames).astype(int).tolist()
    a = E.evaluate_one(cfg, model, ds, idxs)
    b = E.evaluate_one(cfg, model, ds, idxs)
    same = all(abs(a[m] - b[m]) < 1e-12 for m in E.METRICS)
    print(f"    augmented: run1 ap50={a['ap50']:.6f} | run2 ap50={b['ap50']:.6f} | identical={same}")
    return same


def main() -> int:
    run, epoch, frames = "ours_nodepth60", 50, 4
    print(f"real data: {run} epoch_{epoch}.pth, {frames} val frames\n[real code]")
    real = {}
    print("  aggregate (mean +/- sd, no selection):");      real["aggregate"] = check_aggregate(E.aggregate)
    print("  split_dir (val/test mapping + freeze guard):"); real["split_dir"] = check_split(E.split_dir, guard=True)
    print("  load (augmentation forced off):");              real["aug_off"] = check_aug_off(E.load, run, epoch)
    print("  evaluate_one (deterministic on real frames):"); real["determinism"] = check_determinism(run, epoch, frames)
    for k, v in real.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    print("\n[negative controls -- each MUST fail]")
    def selecting(per):                      # picks the best checkpoint
        out = {}
        for m in E.METRICS:
            vals = [per[e][m] for e in sorted(per)]
            out[m] = {"mean": max(vals), "sd": 0.0, "n": 1, "values": [max(vals)]}
        return out
    def unguarded(cfg, split):               # test with no freeze check
        return cfg.val_dir if split == "val" else cfg.test_dir
    def load_aug_on(r, e, dev="cpu"):        # forgets to disable augmentation
        cfg = Config.load(f"{E.RESULTS_ROOT}/{r}/config.json")
        return replace(cfg, device=dev, augment=True), None, e

    bad = {}
    print("  selecting aggregator:");    bad["aggregate"] = check_aggregate(selecting)
    print("  unguarded split_dir:");     bad["split_dir"] = check_split(unguarded, guard=True)
    print("  load leaving aug on:");     bad["aug_off"] = check_aug_off(load_aug_on, run, epoch)
    print("  eval with augment=True:");  bad["determinism"] = check_determinism_broken(run, epoch, frames)
    for k, v in bad.items():
        print(f"  broken {k}: {'correctly FAILS' if not v else 'WRONGLY PASSES'}")

    overall = all(real.values()) and not any(bad.values())
    print(f"\nreal: {real}\noverall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
