import torch
from src.config import Config
from src.train import _pick_device
from src.models.coop_detector import CoopDetector
from src.metrics import _to_device
from src.data.opv2v import OPV2VDataset, opv2v_collate
from torch.utils.data import Subset, DataLoader

cfg = Config(exp_name="overfit_test_warpfix")
device = _pick_device(cfg)

model = CoopDetector(cfg).to(device)
ck = torch.load("results/checkpoints/overfit_test_warpfix/best.pth", map_location=device)
state = ck.get("model", ck) if isinstance(ck, dict) else ck
model.load_state_dict(state)
model.eval()

full_val = OPV2VDataset(cfg.val_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
va = DataLoader(Subset(full_val, list(range(8))), batch_size=cfg.batch_size,
                num_workers=0, collate_fn=opv2v_collate)

with torch.no_grad():
    for i, scene in enumerate(va):
        scene = _to_device(scene, device)
        out = model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)
        gt_mask = scene["gt"]["mask"].bool()
        heat = out["heat"]
        n_pos = int(gt_mask.sum().item())
        pos_vals = heat[gt_mask] if n_pos > 0 else torch.tensor([])
        print(f"frame {i}: heat_max={heat.max().item():.4f}  n_pos_cells={n_pos}  "
              f"pos_vals_at_gt={pos_vals.cpu().numpy().round(4).tolist()[:10]}")
