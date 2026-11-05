"""Measure how often each depth bin occurs in the LiDAR depth labels of the training split.

    python scripts/depth_bin_frequency.py [n_frames] [linear|log]

Writes src/loss/depth_bin_freq_<spacing><K>.npy (fraction of labelled cells per bin), used by
depth_loss_weight="inv_sqrt_freq". Labels come from the real training loader (all agents, as in training).
"""
import sys
sys.path.insert(0, ".")
from dataclasses import replace
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from src.config import Config
from src.baselines import make
from src.data.opv2v import OPV2VDataset, opv2v_collate

n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
spacing = sys.argv[2] if len(sys.argv) > 2 else "linear"
cfg = replace(make("no_fusion", Config()), depth_source="lidar", depth_bins=spacing)
ds = OPV2VDataset(cfg.train_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
sub = Subset(ds, np.linspace(0, len(ds) - 1, n).astype(int).tolist())
counts = torch.zeros(cfg.depth_K, dtype=torch.float64)
for scene in DataLoader(sub, batch_size=1, num_workers=cfg.num_workers, collate_fn=opv2v_collate):
    lab = scene["depth_gt_img"]
    counts += torch.bincount(lab[lab >= 0].flatten(), minlength=cfg.depth_K).double()
freq = (counts / counts.sum()).numpy()
out = f"src/loss/depth_bin_freq_{spacing}{cfg.depth_K}.npy"
np.save(out, freq)
edges = np.linspace(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K + 1)
for lo, hi in ((4, 20), (20, 35), (35, 51.2), (51.2, 90)):
    sel = (edges[:-1] >= lo) & (edges[:-1] < hi)
    print(f"  {lo:>4}-{hi:<4} m: {freq[sel].sum():.3f} of labelled cells")
print(f"{int(counts.sum())} labelled cells from {n} frames -> {out}")
