"""
Inference — operating-point sweep and single-run evaluation.

load_best_model(cfg)  : loads best.pth checkpoint for a trained experiment.
sweep(cfg, model, loader, budgets) : traces the accuracy-bandwidth curve.
test_with_best(cfg)   : convenience — load best weights + run full test-set eval.
"""
from __future__ import annotations
import csv
import os
import torch

from .config import Config
from .models.coop_detector import CoopDetector
from .metrics import evaluate


def load_best_model(cfg: Config) -> CoopDetector:
    """
    Rebuild the model from cfg and load best.pth weights.
    Always uses best.pth (keyed on val AP@0.7) so test numbers
    match the best checkpoint, not the last epoch.
    """
    device = torch.device("cuda" if (cfg.device == "cuda" and torch.cuda.is_available())
                          else "cpu")
    model = CoopDetector(cfg).to(device)
    best_path = f"{cfg.root}/checkpoints/{cfg.exp_name}/best.pth"
    if not os.path.exists(best_path):
        raise FileNotFoundError(
            f"No best.pth found at {best_path}. "
            f"Run training first: python run_baselines.py"
        )
    ck = torch.load(best_path, map_location=device)
    # checkpoint saved as dict with 'model' key; handle both formats
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[{cfg.exp_name}] Loaded best.pth — epoch {ck.get('epoch', '?')}  "
          f"best_ap70={ck.get('best_ap70', '?'):.3f}" if isinstance(ck, dict) else
          f"[{cfg.exp_name}] Loaded best.pth")
    return model


def sweep(cfg: Config, model, test_loader, budgets):
    """
    Sweep budget_frac over 'budgets' list using the SAME trained model.
    One row per budget point in results/curves/<exp_name>.csv.
    """
    out_dir  = f"{cfg.root}/curves"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{cfg.exp_name}.csv"

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "budget_frac", "kb", "ap30", "ap50", "ap70", "oof_recall"])
        for b in budgets:
            r = evaluate(model, test_loader, cfg,
                         hard=True, tau=cfg.tau, budget_frac=float(b))
            w.writerow([cfg.exp_name, float(b), r["kb"],
                        r["ap30"], r["ap50"], r["ap70"], r["oof_recall"]])
            print(f"  [sweep {cfg.exp_name}] budget={b:.2f}  KB={r['kb']:.1f}  "
                  f"AP30/50/70={r['ap30']:.3f}/{r['ap50']:.3f}/{r['ap70']:.3f}  "
                  f"OOF={r['oof_recall']:.3f}")
    return out_path


def test_with_best(cfg: Config, test_loader) -> dict:
    """
    Load best.pth and run test-set evaluation at the paper's operating point.
    Returns the metrics dict.
    """
    model = load_best_model(cfg)
    r = evaluate(model, test_loader, cfg,
                 hard=True, tau=cfg.tau, budget_frac=cfg.budget_frac)
    print(f"\n[TEST {cfg.exp_name}]  "
          f"AP30={r['ap30']:.3f}  AP50={r['ap50']:.3f}  AP70={r['ap70']:.3f}  "
          f"OOF={r['oof_recall']:.3f}  KB={r['kb']:.1f}")
    return r
