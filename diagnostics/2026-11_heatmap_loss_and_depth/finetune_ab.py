"""A/B fine-tune from no_fusion_camfix epoch_40.pth: identical data order / schedule, only the
heatmap loss + training Gaussian differ.  Usage: finetune_ab.py <control|patched> <out_dir> <n_frames> <n_saves>
Nothing in src/ is modified; the 'patched' arm monkeypatches at runtime."""
import sys, os, time, math, random
sys.path.insert(0, "/mnt/scratch2/users/2641309/projects/coop_v2v")
import numpy as np, torch
from torch.utils.data import DataLoader

arm, out_dir, n_frames, n_saves = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
assert arm in ("control", "patched")
os.makedirs(out_dir, exist_ok=True)

if arm == "patched":
    import src.loss.detection as det
    import src.data.bev_augment as ba

    def gfl_centernet(pred, gt, alpha=2.0, beta=4.0, eps=1e-4, pos_weight=1.0):
        # standard CenterNet: both terms / n_pos; pos_weight ignored
        pred = pred.float().clamp(eps, 1.0 - eps); gt = gt.float()
        pos = gt.eq(1.0).float(); neg = 1.0 - pos
        pos_loss = -((1.0 - pred) ** alpha) * torch.log(pred) * pos
        neg_loss = -((1.0 - gt) ** beta) * (pred ** alpha) * torch.log(1.0 - pred) * neg
        return (pos_loss.sum() + neg_loss.sum()) / pos.sum().clamp(min=1.0)

    _orig_radius = ba.gaussian_radius_cells

    def draw_gaussian_centerpoint(heat, row, col, w_m, l_m, cell_m):
        # CenterPoint target: min_overlap=0.1, min_radius=2, sigma=diameter/6
        H, W = heat.shape
        radius = int(max(2.0, _orig_radius(w_m, l_m, cell_m, min_overlap=0.1)))
        d = 2 * radius + 1; sigma = d / 6.0
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        g = np.exp(-(x * x + y * y) / (2 * sigma * sigma)).astype(np.float32)
        ri, ci = int(round(row)), int(round(col))
        top, bot = max(0, ri - radius), min(H, ri + radius + 1)
        left, right = max(0, ci - radius), min(W, ci + radius + 1)
        if bot > top and right > left:
            gp = g[radius - (ri - top):radius + (bot - ri), radius - (ci - left):radius + (right - ci)]
            np.maximum(heat[top:bot, left:right], gp, out=heat[top:bot, left:right])
        if 0 <= ri < H and 0 <= ci < W:
            heat[ri, ci] = 1.0
        return heat

    det.gaussian_focal_loss = gfl_centernet           # detection_loss looks this up at call time
    ba.draw_gaussian_sized = draw_gaussian_centerpoint  # _build_bev_gt_from_boxes imports it at call time

from src.config import Config
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.loss.coop_loss import coop_loss
from src.metrics import _to_device
from src.train import lr_at_epoch, lambda_schedule, gumbel_T, _set_seed

ROOT = "/mnt/scratch2/users/2641309/projects/coop_v2v"
cfg = Config.load(f"{ROOT}/results/no_fusion_camfix/config.json")
_set_seed(cfg.seed)
dev = torch.device("cuda")

ck = torch.load("/mnt/scratch2/users/2641309/tmp_claude_diag/ab/nf_epoch_40.pth", map_location=dev)
model = CoopDetector(cfg).to(dev); model.load_state_dict(ck["model"])
epoch = ck["epoch"] + 1                         # continue as epoch 41, exactly like a resume
backbone_ids = set(id(p) for p in model.backbone.parameters())
params = [{"params": [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids], "lr": cfg.lr},
          {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": cfg.lr * 0.1}]
lr_mults = [g["lr"] / cfg.lr for g in params]
opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
opt.load_state_dict(ck["optimizer"])
cur_lr = lr_at_epoch(epoch, cfg)
for g, m in zip(opt.param_groups, lr_mults): g["lr"] = cur_lr * m
lam, T = lambda_schedule(epoch, cfg), gumbel_T(epoch, cfg)
print(f"[{arm}] start from epoch {ck['epoch']}, lr={cur_lr:.3e} lam={lam} T={T:.3f}", flush=True)

ds = OPV2VDataset(cfg.train_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
gen = torch.Generator(); gen.manual_seed(1234)  # same frame order in both arms
dl = DataLoader(ds, batch_size=1, shuffle=True, generator=gen, num_workers=cfg.num_workers,
                collate_fn=opv2v_collate, pin_memory=True,
                worker_init_fn=lambda w: (np.random.seed(1234 + w), random.seed(1234 + w)))
save_at = set(int(round(n_frames * (i + 1) / n_saves)) for i in range(n_saves))
accum = cfg.grad_accum_steps
model.train(); opt.zero_grad(set_to_none=True)
acc, t0 = {}, time.time()
for step_i, scene in enumerate(dl):
    if step_i >= n_frames: break
    scene = _to_device(scene, dev)
    out = model(scene, hard=False, gumbel_T=T)
    loss, parts = coop_loss(out, scene, cfg, lam=lam)
    if torch.isfinite(loss):
        (loss / accum).backward()
        for k in ("det_hm", "depth", "total"): acc.setdefault(k, []).append(parts.get(k, float("nan")))
        acc.setdefault("heat_mean", []).append(float(out["heat"].detach().mean()))
    if (step_i + 1) % accum == 0:
        torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], cfg.grad_clip)
        opt.step(); opt.zero_grad(set_to_none=True)
    if (step_i + 1) % 200 == 0:
        print(f"[{arm}] {step_i+1}/{n_frames}  " + "  ".join(f"{k}={np.mean(v[-200:]):.4f}" for k, v in acc.items())
              + f"  {(time.time()-t0)/(step_i+1):.2f}s/frame", flush=True)
    if (step_i + 1) in save_at:
        p = f"{out_dir}/{arm}_{step_i+1}.pth"
        torch.save({"model": model.state_dict(), "epoch": epoch, "frames_seen": step_i + 1, "arm": arm}, p)
        print(f"[{arm}] saved {p}", flush=True)
print(f"[{arm}] done", flush=True)
