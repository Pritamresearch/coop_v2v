"""
Bug-vs-generalisation split: evaluate ONE checkpoint on training scenes
(augment off) and on validation scenes with the identical evaluate() code.

Usage:
    python eval_train_vs_val.py <checkpoint.pth> [n_scenes]
"""
import sys
import itertools
from dataclasses import replace

import torch

from src.config import Config
from src.baselines import make
from src.train import build_loaders
from src.metrics import evaluate
from src.models.coop_detector import CoopDetector

CKPT = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300

base = Config()
base.validate()
# augment=False so training scenes are seen exactly as during eval
base = replace(base, augment=False)
cfg = make("ours_full", base)
cfg = replace(cfg, augment=False)

tr, va, _ = build_loaders(cfg)

model = CoopDetector(cfg).cuda()
ckpt = torch.load(CKPT, map_location="cuda")
state = None
for k in ("model", "model_state", "model_state_dict", "state_dict"):
    if isinstance(ckpt, dict) and k in ckpt:
        state = ckpt[k]
        break
if state is None:
    state = ckpt
# map checkpoint keys onto the model's own key names, ignoring any torch.compile
# "_orig_mod." prefix on either side
_norm = lambda s: s.replace("_orig_mod.", "")
_model_keys = {_norm(k): k for k in model.state_dict()}
_dropped = [k for k in state if _norm(k) not in _model_keys]
state = {_model_keys[_norm(k)]: v for k, v in state.items() if _norm(k) in _model_keys}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"checkpoint keys not in model: {len(_dropped)}")
assert not missing, f"{len(missing)} model keys NOT loaded, e.g. {missing[:3]}"
model.eval()
print(f"loaded {CKPT}  epoch={ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'}")
print(f"missing keys={len(missing)}  unexpected keys={len(unexpected)}")
if missing:
    print("  first missing:", missing[:5])

keys = ["val_loss_det", "ap30", "ap50", "ap70", "recall50", "precision50", "n_gt",
        "iou_mean", "centre_rmse", "w_rmse", "l_rmse", "yaw_rmse",
        "heat_pos_mean", "heat_neg_mean", "iou_hist"]

results = {}
for name, loader in (("TRAIN", tr), ("VAL", va)):
    r = evaluate(model, itertools.islice(loader, N), cfg, hard=True, tau=cfg.tau, budget_frac=cfg.budget_frac)
    results[name] = r

print("\n%-14s %14s %14s" % ("metric", "TRAIN", "VAL"))
for k in keys:
    a, b = results["TRAIN"].get(k), results["VAL"].get(k)
    if isinstance(a, float):
        print("%-14s %14.4f %14.4f" % (k, a, b))
    else:
        print("%-14s %14s %14s" % (k, a, b))
