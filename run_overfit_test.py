"""
run_overfit_test.py

Decisive diagnostic: can the (warp-fixed) pipeline overfit 8 real frames?
If yes -> the warp_to_ego bug was the root cause, go back to the full run.
If no  -> the camera-axis geometry issue is still stacked underneath; paste
          one real camera0 yaml block and we fix that next.

Run from the project root:
    python3 run_overfit_test.py 2>&1 | tee overfit_test.log
"""
import torch
from torch.utils.data import Subset, DataLoader

from src.config import Config
from src.train import train
from src.data.opv2v import OPV2VDataset, opv2v_collate


def main():
    cfg = Config(
        exp_name="overfit_test_warpfix",
        epochs=150,
        warmup_epochs=0,     # bandwidth term is zeroed via lambda_bw below anyway;
        lambda_bw=0.0,       # keep loss = pure detection loss for this test
        val_every=5,
    )

    print(f"[overfit_test] train_dir={cfg.train_dir}")
    print(f"[overfit_test] val_dir  ={cfg.val_dir}")

    full_train = OPV2VDataset(cfg.train_dir, cfg,
                               max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
    full_val = full_train  # TRUE overfit test: validate on the SAME frames trained on

    n_train = min(8, len(full_train))
    n_val = min(8, len(full_val))
    if n_train < 8 or n_val < 8:
        print(f"[overfit_test] WARNING: only {n_train} train / {n_val} val frames "
              f"available (wanted 8 each). Continuing with what's there.")

    tr_subset = Subset(full_train, list(range(n_train)))
    va_subset = Subset(full_val, list(range(n_val)))

    kw = dict(
        batch_size=cfg.batch_size,
        num_workers=0,          # keep it simple/deterministic for a diagnostic run
        collate_fn=opv2v_collate,
        pin_memory=(cfg.device == "cuda"),
    )
    tr = DataLoader(tr_subset, shuffle=True, **kw)
    va = DataLoader(va_subset, shuffle=False, **kw)

    print(f"[overfit_test] Overfitting on {n_train} train / {n_val} val frames "
          f"for {cfg.epochs} epochs, lambda_bw=0.")
    print(f"[overfit_test] WATCH: Heat max should climb past ~0.7-0.8, "
          f"AP50 should climb past ~0.3-0.5, within these 150 epochs.")

    model = train(cfg, tr, va, va)   # test_loader arg unused here, reuse va


if __name__ == "__main__":
    main()
