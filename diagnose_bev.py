import argparse, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from src.config import Config
from src.models.coop_detector import CoopDetector
from src.metrics import decode, _to_device
from src.data.opv2v import OPV2VDataset, opv2v_collate

def draw_box(ax, cx, cy, w, l, yaw, color, alpha=0.8):
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    corners_local = np.array([[ l/2, w/2],[ l/2,-w/2],[-l/2,-w/2],[-l/2, w/2]])
    rot = np.array([[cos_y,-sin_y],[sin_y,cos_y]])
    corners = corners_local @ rot.T + np.array([cx, cy])
    ax.add_patch(plt.Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=1.5, alpha=alpha))
    ax.plot(cx, cy, '+', color=color, markersize=6)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="results/checkpoints/ours_full/best.pth")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--out", default="bev_diagnostic.png")
args = ap.parse_args()

cfg = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CoopDetector(cfg).to(device)
ck = torch.load(args.ckpt, map_location=device)
state = ck.get("model", ck) if isinstance(ck, dict) else ck
model.load_state_dict(state)
model.eval()
print(f"Loaded checkpoint epoch={ck.get('epoch','?')}: {args.ckpt}")

ds = OPV2VDataset(cfg.val_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
loader = DataLoader(Subset(ds, list(range(min(args.n, len(ds))))), batch_size=1, shuffle=False, collate_fn=opv2v_collate)

fig, axes = plt.subplots(1, args.n, figsize=(5*args.n, 5))
if args.n == 1: axes = [axes]
all_gt_cx, all_pred_cx, all_gt_cy, all_pred_cy = [], [], [], []

with torch.no_grad():
    for fi, scene in enumerate(loader):
        scene = _to_device(scene, device)
        out = model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)
        pred_boxes, scores = decode(out, cfg)
        gt_boxes = scene["gt"]["boxes"].cpu().numpy()  # shape (M,7), no batch dim
        ax = axes[fi]
        ax.set_xlim(-50,50); ax.set_ylim(-50,50); ax.set_aspect('equal')
        ax.set_facecolor('#1a1a2e')
        ax.set_title(f"Frame {fi} | GT={len(gt_boxes)} Pred={len(pred_boxes)}", fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        for g in gt_boxes:
            draw_box(ax, g[0], g[1], g[3], g[4], g[6], 'lime')
            all_gt_cx.append(g[0]); all_gt_cy.append(g[1])
        top_k = min(10, len(pred_boxes))
        if top_k > 0:
            for k in np.argsort(-scores)[:top_k]:
                p = pred_boxes[k]
                draw_box(ax, p[0], p[1], p[3], p[4], p[6], 'red', 0.6)
                all_pred_cx.append(p[0]); all_pred_cy.append(p[1])
        ax.plot(0, 0, 'w*', markersize=12)
        print(f"\n=== Frame {fi} ===")
        print("GT boxes (cx, cy, w, l, yaw_deg):")
        for g in gt_boxes[:5]:
            print(f"  cx={g[0]:+.1f}  cy={g[1]:+.1f}  w={g[3]:.1f}  l={g[4]:.1f}  yaw={math.degrees(g[6]):.1f}deg")
        print("Top predictions (cx, cy, w, l, yaw_deg, score):")
        if top_k > 0:
            for k in np.argsort(-scores)[:5]:
                p = pred_boxes[k]
                print(f"  cx={p[0]:+.1f}  cy={p[1]:+.1f}  w={p[3]:.1f}  l={p[4]:.1f}  yaw={math.degrees(p[6]):.1f}deg  score={scores[k]:.3f}")

if all_gt_cx:
    print(f"\n=== Systematic offset check ===")
    print(f"GT   cx mean={np.mean(all_gt_cx):+.1f}  cy mean={np.mean(all_gt_cy):+.1f}")
    print(f"Pred cx mean={np.mean(all_pred_cx):+.1f}  cy mean={np.mean(all_pred_cy):+.1f}")
    print(f"Offset dx={np.mean(all_pred_cx)-np.mean(all_gt_cx):+.1f}m  dy={np.mean(all_pred_cy)-np.mean(all_gt_cy):+.1f}m")

plt.tight_layout()
plt.savefig(args.out, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
print(f"\nSaved: {args.out}")
