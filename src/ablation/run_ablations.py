"""
Ablation matrix.

Each ablation toggles exactly ONE flag on the base config. A1 and A2 are
load-bearing (they decide whether N1 and N3 hold).

A1  k x use_utility_gate       -- does gating make top-k beat top-1?  (N3)
A2  utility_type               -- value of depth-awareness             (N1)
A3  receiver_centric           -- value of receiver conditioning       (N2)
A4  alpha (init value)         -- depth sensitivity                    (N1)
A6  joint_train                -- joint vs. separate training          (N4)

Each variant is trained, then evaluated at cfg.budget_frac on the test loader;
the row (variant_id, ap50, ap70, oof_recall, kb) is appended to
results/ablation/<ablation_id>.csv so `make_plots.py` can render a
one-shot decisive figure.
"""
from __future__ import annotations
import csv
import os
from dataclasses import replace

from ..config import Config
from ..train import train, build_loaders
from ..metrics import evaluate


def _append(csv_path: str, header: list[str], row: list):
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def _train_and_eval(cfg: Config):
    tr, va, te = build_loaders(cfg)
    model = train(cfg, tr, va, te)
    r = evaluate(model, te, cfg, hard=True, tau=cfg.tau, budget_frac=cfg.budget_frac)
    return r


def run_A1(base: Config):
    """Decisive test: does gating reverse 'more-is-worse'?"""
    root = f"{base.root}/ablation"
    os.makedirs(root, exist_ok=True)
    csv_path = f"{root}/A1_topk_gate.csv"
    for k in [1, 2, 3]:
        for gate in [True, False]:
            cfg = replace(base, exp_name=f"A1_k{k}_gate{int(gate)}",
                          k=k, use_utility_gate=gate)
            r = _train_and_eval(cfg)
            _append(csv_path,
                    ["k", "gate", "ap30", "ap50", "ap70", "oof_recall", "kb"],
                    [k, int(gate), r["ap30"], r["ap50"], r["ap70"], r["oof_recall"], r["kb"]])


def run_A2(base: Config):
    """Utility variants at matched bandwidth."""
    root = f"{base.root}/ablation"; os.makedirs(root, exist_ok=True)
    csv_path = f"{root}/A2_utility.csv"
    for ut in ["depth_aware", "confidence_only", "magnitude_only", "learned_fue"]:
        cfg = replace(base, exp_name=f"A2_{ut}", utility_type=ut)
        r = _train_and_eval(cfg)
        _append(csv_path,
                ["utility_type", "ap30", "ap50", "ap70", "oof_recall", "kb"],
                [ut, r["ap30"], r["ap50"], r["ap70"], r["oof_recall"], r["kb"]])


def run_A3(base: Config):
    root = f"{base.root}/ablation"; os.makedirs(root, exist_ok=True)
    csv_path = f"{root}/A3_receiver.csv"
    for rc in [True, False]:
        cfg = replace(base, exp_name=f"A3_rc{int(rc)}", receiver_centric=rc)
        r = _train_and_eval(cfg)
        _append(csv_path,
                ["receiver_centric", "ap30", "ap50", "ap70", "oof_recall", "kb"],
                [int(rc), r["ap30"], r["ap50"], r["ap70"], r["oof_recall"], r["kb"]])


def run_A4(base: Config):
    root = f"{base.root}/ablation"; os.makedirs(root, exist_ok=True)
    csv_path = f"{root}/A4_alpha.csv"
    for a in [0.0, 2.0, 4.0, 8.0]:
        cfg = replace(base, exp_name=f"A4_alpha{a}", alpha=a)
        r = _train_and_eval(cfg)
        _append(csv_path,
                ["alpha_init", "ap30", "ap50", "ap70", "oof_recall", "kb"],
                [a, r["ap30"], r["ap50"], r["ap70"], r["oof_recall"], r["kb"]])
