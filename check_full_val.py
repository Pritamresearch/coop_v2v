import torch
from src.config import Config
from src.train import _pick_device
from src.models.coop_detector import CoopDetector
from src.metrics import evaluate
from src.data.opv2v import OPV2VDataset, opv2v_collate
from torch.utils.data import DataLoader

cfg = Config(exp_name="ours_full_40m")
device = _pick_device(cfg)

model = CoopDetector(cfg).to(device)
ck = torch.load("results/checkpoints/ours_full_40m/best.pth", map_location=device)
state = ck.get("model", ck)
model.load_state_dict(state)
print("checkpoint best_ap50 (recorded at save time):", ck.get("best_ap50"))

full_val = OPV2VDataset(cfg.val_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
va = DataLoader(full_val, batch_size=cfg.batch_size, num_workers=0, collate_fn=opv2v_collate)

r = evaluate(model, va, cfg, hard=True, tau=cfg.tau, budget_frac=cfg.budget_frac)
print(f"FULL val set: ap30={r['ap30']:.4f}  ap50={r['ap50']:.4f}  ap70={r['ap70']:.4f}")
print(f"n_gt={r.get('n_gt')}  n_pred={r.get('n_pred')}  precision50={r.get('precision50')}  recall50={r.get('recall50')}")
