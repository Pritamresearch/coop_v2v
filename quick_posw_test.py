"""
quick_posw_test.py -- 10-epoch validation run with the new pos_weight,
writing to its own exp_name so it never touches ours_full's history.
"""
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders

def main():
    base = Config()
    base.validate()

    print("\n========== Building OPV2V data loaders ==========")
    tr, va, te = build_loaders(base)
    print(f"  Train: {len(tr.dataset)}  Val: {len(va.dataset)}  Test: {len(te.dataset)}")

    cfg = make("ours_full", base)
    cfg = replace(cfg, exp_name="ours_full_posw20", epochs=10, resume=True)

    print(f"\n{'='*60}")
    print(f"  Training: {cfg.exp_name}  (pos_weight=20.0, epochs={cfg.epochs})")
    print(f"{'='*60}")
    train(cfg, tr, va, te)

if __name__ == "__main__":
    main()
