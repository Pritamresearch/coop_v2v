"""
Ablation study runner — real OPV2V data only.

    python run_ablations.py [--only A1,A2]

Writes:
    results/ablation/A1_topk_gate.csv
    results/ablation/A2_utility.csv
    results/ablation/A3_receiver.csv
    results/ablation/A4_alpha.csv
"""
from __future__ import annotations
import argparse
from src.config import Config
from src.ablation.run_ablations import run_A1, run_A2, run_A3, run_A4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default="A1,A2,A3,A4",
                   help="Comma-separated ablations to run (default: all)")
    args = p.parse_args()
    run_ids = {x.strip() for x in args.only.split(",")}

    base = Config()   # full OPV2V config
    base.validate()

    if "A1" in run_ids:
        print("\n===== A1: top-k x gating =====")
        run_A1(base)
    if "A2" in run_ids:
        print("\n===== A2: utility type =====")
        run_A2(base)
    if "A3" in run_ids:
        print("\n===== A3: receiver-centric =====")
        run_A3(base)
    if "A4" in run_ids:
        print("\n===== A4: alpha depth sensitivity =====")
        run_A4(base)

    print("\nAblations complete. Figures: python -m plots.make_plots")


if __name__ == "__main__":
    main()
