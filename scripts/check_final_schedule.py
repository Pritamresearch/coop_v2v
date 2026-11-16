"""Gate: the final-round schedule must behave as pre-registered (CPU, instant).

Pre-registered final round (diagnostics/2026-11_cooperation/PREREG_FINAL_ROUND.md):
cosine LR with T_max=120, training stopped after epoch 59, every auxiliary schedule
finishing inside the 60 trained epochs.

Checks, all against the PRODUCTION schedule functions in src.train:
  1. exactly 60 epochs are trained (0..59) under epochs=120, max_epoch=59
  2. the LR at each epoch matches the 120-epoch cosine, NOT a 60-epoch cosine
  3. Gumbel T reaches T1 at the final trained epoch
  4. the lambda bandwidth term switches on strictly before the final trained epoch
  5. LR is monotonically decreasing after warmup, and T is monotone

Each has a negative control that MUST fail, per the standing rule that a gate which
cannot fail is decoration.
"""
from __future__ import annotations
import math, os, sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.train import lr_at_epoch, gumbel_T, lambda_schedule

FINAL = dict(epochs=120, max_epoch=59, gumbel_epochs=60, warmup_epochs=20,
             lr_warmup_epochs=5, save_every=10)
LAST = 59


def trained_epochs(cfg):
    out = []
    for e in range(cfg.epochs):
        if getattr(cfg, "max_epoch", -1) >= 0 and e > cfg.max_epoch:
            break
        out.append(e)
    return out


def cosine(e, epochs, warm, base, eta=1e-6):
    if e < warm:
        return base * (e + 1) / max(warm, 1)
    p = min(max((e - warm) / max(1, epochs - warm), 0.0), 1.0)
    return eta + 0.5 * (base - eta) * (1 + math.cos(math.pi * p))


def run(cfg, label):
    eps = trained_epochs(cfg)
    lrs = [lr_at_epoch(e, cfg, cfg.lr_warmup_epochs) for e in eps]
    Ts = [gumbel_T(e, cfg) for e in eps]
    lams = [lambda_schedule(e, cfg) for e in eps]
    w = cfg.lr_warmup_epochs

    c = {}
    c["60 epochs trained (0..59)"] = (len(eps) == 60 and eps[0] == 0 and eps[-1] == LAST)
    c["LR follows the 120-epoch cosine"] = all(
        abs(lr - cosine(e, 120, w, cfg.lr)) < 1e-12 for e, lr in zip(eps, lrs))
    c["LR is NOT a 60-epoch cosine"] = any(
        abs(lr - cosine(e, 60, w, cfg.lr)) > 1e-9 for e, lr in zip(eps, lrs))
    c["Gumbel reaches T1 at epoch 59"] = abs(Ts[-1] - cfg.gumbel_T1) < 1e-9
    c["lambda on before the last epoch"] = (lams[-1] == cfg.lambda_bw and lams[0] == 0.0
                                            and cfg.warmup_epochs < LAST)
    c["LR decreasing after warmup"] = all(b <= a + 1e-15 for a, b in zip(lrs[w:], lrs[w + 1:]))
    c["T decreasing"] = all(b <= a + 1e-12 for a, b in zip(Ts, Ts[1:]))

    print(f"\n[{label}]  trained {len(eps)} epochs  "
          f"LR {lrs[0]:.2e} -> {lrs[-1]:.2e} ({lrs[-1]/cfg.lr:.1%} of base)  "
          f"T {Ts[0]:.2f} -> {Ts[-1]:.2f}  lambda on from epoch {cfg.warmup_epochs}")
    for k, v in c.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")
    return all(c.values())


def main() -> int:
    base = Config()
    good = replace(base, **FINAL)
    ok = run(good, "pre-registered final schedule")

    print("\n[negative controls -- each MUST fail]")
    controls = {
        "gumbel tied to cfg.epochs (gumbel_epochs=-1)": replace(good, gumbel_epochs=-1),
        "schedule compressed to 60 (the regime that cost accuracy)": replace(good, epochs=60, max_epoch=-1, gumbel_epochs=60),
        "lambda warmup past the final trained epoch": replace(good, warmup_epochs=80),
        "no early stop (trains all 120)": replace(good, max_epoch=-1),
    }
    bad = {}
    for label, cfg in controls.items():
        bad[label] = run(cfg, "broken: " + label)

    print("\n" + "=" * 78)
    print(f"final schedule: {'PASS' if ok else 'FAIL'}")
    for label, passed in bad.items():
        print(f"  control {'correctly FAILS' if not passed else 'WRONGLY PASSES'}: {label}")
    overall = ok and not any(bad.values())
    print(f"overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
