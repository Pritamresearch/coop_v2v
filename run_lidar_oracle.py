"""
run_lidar_oracle.py

Upper-bound / oracle experiment: trains the exact same framework as
ours_full, but with real LiDAR depth supervision (depth_source="lidar")
instead of ZoeDepth monocular priors.

Purpose: establish the best-case performance ceiling for the framework
when depth is known exactly, so the camera-only (ZoeDepth) result can be
reported as "X% of the LiDAR-supervised upper bound."

Dataset note: only ~11% of OPV2V frames have a synchronised .pcd file
(the loader auto-filters to these via cfg.depth_source == "lidar"), so
this trains on a smaller (~1,063 frame) but genuinely LiDAR-supervised
subset. All other hyperparameters match the main ours_full configuration
for a fair comparison.

Run:
    python run_lidar_oracle.py --epochs 120
"""
import argparse
from src.config import Config
from src.train import train, build_loaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=40)
    args = ap.parse_args()

    cfg = Config(
        exp_name="lidar_oracle",
        depth_source="lidar",
        gamma_depth=1.0,
        epochs=args.epochs,
        warmup_epochs=args.warmup,
        resume=False,
    )

    print(f"====== LiDAR-oracle training: {cfg.exp_name} ======")
    print(f"depth_source={cfg.depth_source}  gamma_depth={cfg.gamma_depth}  "
          f"epochs={cfg.epochs}  warmup_epochs={cfg.warmup_epochs}")

    tr, va, te = build_loaders(cfg)
    print(f"Train: {len(tr.dataset)}  Val: {len(va.dataset)}  Test: {len(te.dataset)}")

    model = train(cfg, tr, va, te)
    print(f"====== Done: {cfg.exp_name} ======")


if __name__ == "__main__":
    main()