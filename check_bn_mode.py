import torch
import torch.nn as nn
from src.config import Config
from src.train import _pick_device
from src.models.coop_detector import CoopDetector
from src.metrics import evaluate, _to_device
from src.data.opv2v import OPV2VDataset, opv2v_collate
from torch.utils.data import Subset, DataLoader

cfg = Config(exp_name="ours_full_40m")
device = _pick_device(cfg)

model = CoopDetector(cfg).to(device)
ck = torch.load("results/checkpoints/ours_full_40m/best.pth", map_location=device)
state = ck.get("model", ck)
model.load_state_dict(state)

full_val = OPV2VDataset(cfg.val_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
va = DataLoader(Subset(full_val, list(range(64))), batch_size=cfg.batch_size,
                num_workers=0, collate_fn=opv2v_collate)

print("=== [1] eval() normal (uses BN running stats) ===")
r1 = evaluate(model, va, cfg, hard=True, tau=cfg.tau, budget_frac=cfg.budget_frac)
print(f"AP50={r1['ap50']:.4f}  heat_max={r1['heat_max']:.4f}")

print("=== [2] batch-stats mode AP (live stats, momentum=0) ===")
bn_mods = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
saved = [(m.training, m.momentum) for m in bn_mods]
for m in bn_mods:
    m.train(); m.momentum = 0.0
r2 = evaluate(model, va, cfg, hard=True, tau=cfg.tau, budget_frac=cfg.budget_frac)
print(f"AP50={r2['ap50']:.4f}  heat_max={r2['heat_max']:.4f}")
for m, (tr, mom) in zip(bn_mods, saved):
    m.train(tr); m.momentum = mom

print("=== [3] precise-BN mode (running stats recalibrated over 200 train frames, no aug) ===")
full_train = OPV2VDataset(cfg.train_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
tr_loader = DataLoader(Subset(full_train, list(range(200))), batch_size=cfg.batch_size,
                       num_workers=0, collate_fn=opv2v_collate)
for m in bn_mods:
    m.reset_running_stats(); m.momentum = None; m.train()
with torch.no_grad():
    for scene in tr_loader:
        scene = _to_device(scene, device)
        model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)
model.eval()
r3 = evaluate(model, va, cfg, hard=True, tau=cfg.tau, budget_frac=cfg.budget_frac)
print(f"AP50={r3['ap50']:.4f}  heat_max={r3['heat_max']:.4f}")

print()
print("SUMMARY")
print(f"  [1] running-stats eval AP50 = {r1['ap50']:.4f}")
print(f"  [2] live batch-stats   AP50 = {r2['ap50']:.4f}")
print(f"  [3] precise-BN recal   AP50 = {r3['ap50']:.4f}")
