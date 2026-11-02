"""
Train all baseline presets on real OPV2V data and generate all paper figures.

    python run_baselines.py

Each preset trains for cfg.epochs (default 60), saves checkpoints to
results/checkpoints/<exp_name>/best.pth, then sweeps the accuracy-bandwidth
curve and generates qualitative figures.

To resume interrupted training:
    Edit src/config.py and set resume=True, then re-run.

Output structure:
    results/
        checkpoints/<exp_name>/best.pth   <- used for all test eval
        checkpoints/<exp_name>/last.pth
        logs/<exp_name>/train_log.csv
        curves/<exp_name>.csv
        ablation/A*.csv
        robustness/<exp_name>.csv
    plots/figures/
        fig_accuracy_bandwidth.pdf/png
        fig_oof_recall.pdf/png
        fig_ablation_topk_gate.pdf/png
        fig_robustness.pdf/png
        qual_bev_*.png               <- BEV 3D boxes (GT green / pred colored)
        qual_cam_*.png               <- camera 3D wireframe projection
        qual_utility_chain.png       <- N1->N2->N3 per-agent maps
        qual_feature_sparsity.png    <- dense vs transmitted vs fused BEV
        qual_multi_method_grid.png   <- rows=scenes, cols=methods
"""
from __future__ import annotations
import argparse
import os

from src.config import Config
from src.baselines import make, ALL
from src.train import train, build_loaders
from src.inference import sweep, load_best_model
from src.robustness import run_robustness
from plots.make_plots import make_all
from plots.visualize_boxes import make_qualitative_figures, render_multi_method_grid
from plots.visualize_utility import render_utility_maps
from plots.visualize_features import render_feature_sparsity


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated subset of presets to run, e.g. ours_full,no_fusion")
    p.add_argument("--epochs", type=int, default=None,
               help="Override cfg.epochs (e.g. 30 for faster runs)")
    p.add_argument("--warmup", type=int, default=None,
               help="Override cfg.warmup_epochs")

    args = p.parse_args()

    os.makedirs("plots/figures", exist_ok=True)

    # Base config — full OPV2V paper-grade settings
    base = Config()

    if args.epochs is not None:
        base.epochs = args.epochs
    if args.warmup is not None:
        base.warmup_epochs = args.warmup

    base.validate()

    presets = [x.strip() for x in args.only.split(",") if x.strip()] or ["ours_full", "no_fusion", "dense", "where2comm_cam", "jigsawcomm_cam"]
    budgets = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

    # Build loaders once (shared across all presets — same dataset)
    print("\n========== Building OPV2V data loaders ==========")
    tr, va, te = build_loaders(base)
    print(f"  Train: {len(tr.dataset)}  Val: {len(va.dataset)}  Test: {len(te.dataset)}")

    trained = {}
    for name in presets:
        cfg = make(name, base)
        print(f"\n{'='*60}")
        print(f"  Training: {name}")
        print(f"{'='*60}")
        model = train(cfg, tr, va, te)
        trained[name] = (cfg, model)

        print(f"\n---------- Budget sweep: {name} ----------")
        sweep(cfg, model, te, budgets)

    # ----------------------------------------------------------------
    # Qualitative figures — BEV + camera 3D boxes
    # ----------------------------------------------------------------
    print("\n========== Qualitative 3D-box figures ==========")
    for name in ("ours_full", "no_fusion"):
        if name not in trained:
            continue
        cfg, model = trained[name]
        make_qualitative_figures(cfg, model, te.dataset, n_scenes=5,
                                 out_dir=f"plots/figures/{name}")

    # Utility-chain + feature-sparsity (JigsawComm Fig. 3 / 4 equivalent)
    if "ours_full" in trained:
        cfg_o, model_o = trained["ours_full"]
        print("\n========== Utility chain + feature sparsity figures ==========")
        scene0 = te.dataset[0]
        render_utility_maps(model_o, scene0,
                            "plots/figures/qual_utility_chain.png", max_collab=4)
        render_feature_sparsity(model_o, scene0,
                                "plots/figures/qual_feature_sparsity.png")

    # Multi-method qualitative grid
    if len(trained) > 1:
        cfg_o = trained.get("ours_full", list(trained.values())[0])[0]
        print("\n========== Multi-method qualitative grid ==========")
        render_multi_method_grid(
            cfg_o,
            {name: (trained[name][0], trained[name][1]) for name in trained},
            te.dataset,
            "plots/figures/qual_multi_method_grid.png",
            n_scenes=5,
        )

    # ----------------------------------------------------------------
    # Robustness sweeps on ours_full (uses best.pth loaded in train())
    # ----------------------------------------------------------------
    if "ours_full" in trained:
        cfg_o, model_o = trained["ours_full"]
        print("\n========== Robustness sweeps ==========")
        run_robustness(cfg_o, model_o, te)

    # ----------------------------------------------------------------
    # All aggregate figures from CSVs
    # ----------------------------------------------------------------
    print("\n========== Generating all figures ==========")
    make_all(root=str(base.root))

    print("\n========== COMPLETE ==========")
    print(f"  Results     : {base.root}")
    print(f"  Checkpoints : {base.root}/checkpoints/")
    print(f"  Figures     : plots/figures/")
    print(f"  Logs        : {base.root}/logs/")


if __name__ == "__main__":
    main()
