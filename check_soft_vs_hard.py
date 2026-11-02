import torch
from src.config import Config
from src.train import _pick_device
from src.models.coop_detector import CoopDetector
from src.metrics import evaluate
from src.data.opv2v import OPV2VDataset, opv2v_collate
from torch.utils.data import Subset, DataLoader

cfg = Config(exp_name="overfit_test_warpfix")
device = _pick_device(cfg)

model = CoopDetector(cfg).to(device)
ck = torch.load("results/checkpoints/overfit_test_warpfix/best.pth", map_location=device)
state = ck.get("model", ck)
model.load_state_dict(state)

full_val = OPV2VDataset(cfg.val_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
va = DataLoader(Subset(full_val, list(range(8))), batch_size=cfg.batch_size,
                num_workers=0, collate_fn=opv2v_collate)

print("=== hard=True (current eval path) ===")
r_hard = evaluate(model, va, cfg, hard=True, tau=cfg.tau, budget_frac=cfg.budget_frac)
print(f"AP50={r_hard['ap50']:.4f}  heat_max={r_hard['heat_max']:.4f}  score_max={r_hard['score_max']:.4f}")

print("=== hard=False (matches training-time selection) ===")
r_soft = evaluate(model, va, cfg, hard=False, tau=cfg.tau, budget_frac=cfg.budget_frac)
print(f"AP50={r_soft['ap50']:.4f}  heat_max={r_soft['heat_max']:.4f}  score_max={r_soft['score_max']:.4f}")
