"""
Decisive diagnostic: can the model overfit 8 real scenes to near-perfect detection?

If YES (TP approaches n_gt, loss keeps dropping to near 0): the pipeline is
correct, and the full-dataset issue is about scale/imbalance/schedule, not a bug.

If NO (TP stays low despite loss dropping, on just 8 memorizable scenes):
there is a genuine bug in loss/data/eval that no amount of full-dataset
training time will fix.

Run: python overfit_test.py
Takes ~15-20 minutes on GPU for 100 epochs over 8 scenes.
"""
import torch
from torch.utils.data import DataLoader, Subset
from src.config import Config
from src.baselines import make
from src.models.coop_detector import CoopDetector
from src.loss.coop_loss import coop_loss
from src.metrics import evaluate, _to_device
from src.data.opv2v import OPV2VDataset, opv2v_collate

cfg = make("ours_full", Config())
cfg.epochs = 100
cfg.warmup_epochs = 100   # stay in warmup the whole time -- pure detection loss only
cfg.lr = 2e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading 8 fixed scenes from train set...")
full_ds = OPV2VDataset(cfg.train_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
small_ds = Subset(full_ds, list(range(8)))
loader = DataLoader(small_ds, batch_size=1, collate_fn=opv2v_collate)

model = CoopDetector(cfg).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

print(f"Overfitting {len(small_ds)} scenes for {cfg.epochs} epochs...")
for epoch in range(cfg.epochs):
    model.train()
    total_loss = 0.0
    for scene in loader:
        scene = _to_device(scene, device)
        opt.zero_grad(set_to_none=True)
        out = model(scene, hard=False, gumbel_T=1.0)
        loss, parts = coop_loss(out, scene, cfg, lam=0.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += parts["total"]

    if epoch % 10 == 0 or epoch == cfg.epochs - 1:
        val = evaluate(model, loader, cfg, hard=True, tau=cfg.tau, budget_frac=cfg.budget_frac)
        print(f"epoch {epoch:3d}  loss={total_loss/len(loader):.4f}  "
              f"AP70={val['ap70']:.3f}  TP={val['tp50']}  GT={val['n_gt']}  "
              f"Recall={val['recall50']:.3f}  heat_pos={val['heat_pos_mean']:.4f}  "
              f"heat_neg={val['heat_neg_mean']:.4f}")

print("\nDone. If TP/n_gt is not approaching 1.0 by epoch 90-100, there is a real bug.")
