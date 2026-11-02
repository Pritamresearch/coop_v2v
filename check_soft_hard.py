import sys; sys.path.insert(0,'.')
import torch
from torch.utils.data import DataLoader, Subset
from src.config import Config
from src.train import build_loaders
from src.models.coop_detector import CoopDetector
from src.metrics import evaluate

cfg = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device, flush=True)

tr, va, te = build_loaders(cfg)
va_small = DataLoader(Subset(va.dataset, list(range(100))),
                      batch_size=va.batch_size, shuffle=False,
                      collate_fn=va.collate_fn, num_workers=0)

model = CoopDetector(cfg).to(device)
ck = torch.load('results/checkpoints/ours_full/best.pth', map_location=device)
m, u = model.load_state_dict(ck['model'], strict=False)
real_missing = [k for k in m if 'num_batches_tracked' not in k]
print('load clean:', len(real_missing) == 0 and len(u) == 0, flush=True)
print('checkpoint epoch:', ck.get('epoch'), 'best_ap50:', ck.get('best_ap50'), flush=True)

# T used during training at that epoch, for reference
from src.train import gumbel_T
print('gumbel_T at ckpt epoch:', gumbel_T(ck.get('epoch', 0), cfg), flush=True)

for hard in [True, False]:
    r = evaluate(model, va_small, cfg, hard=hard)
    print(f'hard={hard}  ap50={r["ap50"]:.4f}  ap70={r["ap70"]:.4f}  '
          f'recall50={r["recall50"]:.4f}  precision50={r["precision50"]:.4f}  '
          f'val_loss={r["val_loss_det"]:.2f}  '
          f'pos={r["heat_pos_mean"]:.4f} neg={r["heat_neg_mean"]:.4f}', flush=True)
