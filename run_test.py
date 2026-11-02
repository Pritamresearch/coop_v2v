"""
Test-set evaluation using best.pth checkpoints.

    python run_test.py [--exp ours_full] [--all]

Loads best.pth for the specified experiment (or all presets with --all),
runs the full test set, generates:
    - 3D bounding box qualitative figures (BEV + camera overlay)
    - per-scene boxes_*.json files
    - accuracy-bandwidth curve
    - robustness plots

This script never retrains — it only uses saved best.pth weights.
"""
from __future__ import annotations
import argparse
import os
import csv

import torch

from src.config import Config
from src.baselines import make, ALL
from src.train import build_loaders
from src.inference import load_best_model, sweep, test_with_best
from src.robustness import run_robustness
from plots.make_plots import make_all
from plots.visualize_boxes import make_qualitative_figures, render_multi_method_grid
from plots.visualize_utility import render_utility_maps
from plots.visualize_features import render_feature_sparsity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp",  type=str, default="ours_full",
                   help="Experiment name matching a trained preset")
    p.add_argument("--all",  action="store_true",
                   help="Run test eval for ALL presets")
    p.add_argument("--n_qual", type=int, default=5,
                   help="Number of scenes for qualitative figures")
    args = p.parse_args()

    os.makedirs("plots/figures", exist_ok=True)

    base = Config()
    base.validate()

    # Build test loader once
    print("\n===== Building OPV2V test loader =====")
    _, _, te = build_loaders(base)
    print(f"  Test scenes: {len(te.dataset)}")

    presets = ALL if args.all else [args.exp]
    budgets  = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

    results_summary = []
    loaded_models   = {}

    # ----------------------------------------------------------------
    # 1) Load best.pth and evaluate each preset
    # ----------------------------------------------------------------
    for name in presets:
        cfg = make(name, base)
        print(f"\n{'='*60}")
        print(f"  Evaluating: {name}  (loading best.pth)")
        print(f"{'='*60}")

        try:
            model = load_best_model(cfg)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        loaded_models[name] = (cfg, model)

        # Fixed operating-point test
        r = test_with_best(cfg, te)
        results_summary.append({
            "method": name,
            "ap30": r["ap30"], "ap50": r["ap50"], "ap70": r["ap70"],
            "oof_recall": r["oof_recall"], "kb": r["kb"],
        })

        # Budget sweep
        print(f"\n  Sweeping budget for {name}...")
        sweep(cfg, model, te, budgets)

        # Qualitative 3D-box figures
        out_dir = f"plots/figures/{name}"
        print(f"\n  Qualitative figures -> {out_dir}/")
        make_qualitative_figures(cfg, model, te.dataset,
                                 n_scenes=args.n_qual, out_dir=out_dir)

    # ----------------------------------------------------------------
    # 2) Utility-chain + feature-sparsity for ours_full
    # ----------------------------------------------------------------
    if "ours_full" in loaded_models:
        cfg_o, model_o = loaded_models["ours_full"]
        scene0 = te.dataset[0]
        print("\n===== Utility chain + feature sparsity figures =====")
        render_utility_maps(model_o, scene0,
                            "plots/figures/qual_utility_chain.png", max_collab=4)
        render_feature_sparsity(model_o, scene0,
                                "plots/figures/qual_feature_sparsity.png")

    # ----------------------------------------------------------------
    # 3) Multi-method qualitative grid
    # ----------------------------------------------------------------
    if len(loaded_models) > 1:
        cfg_ref = loaded_models.get("ours_full",
                                    list(loaded_models.values())[0])[0]
        print("\n===== Multi-method qualitative grid =====")
        render_multi_method_grid(
            cfg_ref,
            {name: loaded_models[name] for name in loaded_models},
            te.dataset,
            "plots/figures/qual_multi_method_grid.png",
            n_scenes=args.n_qual,
        )

    # ----------------------------------------------------------------
    # 4) Robustness sweeps (ours_full only)
    # ----------------------------------------------------------------
    if "ours_full" in loaded_models:
        cfg_o, model_o = loaded_models["ours_full"]
        print("\n===== Robustness sweeps =====")
        run_robustness(cfg_o, model_o, te)

    # ----------------------------------------------------------------
    # 5) All aggregate figures from CSVs
    # ----------------------------------------------------------------
    print("\n===== Generating all aggregate figures =====")
    make_all(root=str(base.root))

    # ----------------------------------------------------------------
    # 6) Print summary table
    # ----------------------------------------------------------------
    if results_summary:
        print("\n" + "=" * 72)
        print(f"{'Method':<20} {'AP@0.3':>8} {'AP@0.5':>8} {'AP@0.7':>8} "
              f"{'OOF-R':>8} {'KB':>8}")
        print("-" * 72)
        for r in results_summary:
            print(f"{r['method']:<20} {r['ap30']:>8.3f} {r['ap50']:>8.3f} "
                  f"{r['ap70']:>8.3f} {r['oof_recall']:>8.3f} {r['kb']:>8.1f}")
        print("=" * 72)

        # Save summary CSV
        summary_path = f"{base.root}/test_summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method","ap30","ap50","ap70","oof_recall","kb"])
            w.writeheader()
            w.writerows(results_summary)
        print(f"\nSummary saved: {summary_path}")

    print("\n===== DONE =====")
    print(f"  Figures    : plots/figures/")
    print(f"  3D boxes   : plots/figures/<method>/boxes_*.json")
    print(f"  Curves     : {base.root}/curves/")
    print(f"  Robustness : {base.root}/robustness/")


if __name__ == "__main__":
    main()
