"""
Pre-flight check for the 4:3 input change (704x256 -> 512x384).
CPU only, run on the login node. Loads 3 validation frames and verifies:
  1. images come out at (lss_img_H, lss_img_W) = (384, 512)
  2. dataset IMG_H/IMG_W match config lss_img_H/lss_img_W
  3. intrinsics are scaled UNIFORMLY (fx/fy ratio unchanged from native)
  4. depth supervision tensor has the new feature-map size (48 x 64)
     and still contains valid depth bins (not all -1)
Usage:  python check_input_res.py
"""
import numpy as np
import torch

from src.config import Config
from src.data.opv2v import OPV2VDataset

cfg = Config()
H, W, ds = cfg.lss_img_H, cfg.lss_img_W, cfg.lss_downsample
print(f"config: lss_img = {H}x{W}  downsample = {ds}  -> feature map {H // ds}x{W // ds}")

ok = True
def check(cond, msg):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    ok &= bool(cond)

check(abs(W / H - 800 / 600) < 1e-6, f"aspect ratio {W}/{H} = {W/H:.4f} equals native 4:3")
check(H % 32 == 0 and W % 32 == 0, "H and W divisible by 32 (EfficientNet strides)")

ds_val = OPV2VDataset(cfg.val_dir, cfg, max_agents=cfg.max_agents)
check(ds_val.IMG_H == H and ds_val.IMG_W == W,
      f"dataset IMG_H/IMG_W = {ds_val.IMG_H}x{ds_val.IMG_W} matches config")

for i in range(3):
    s = ds_val[i]
    print(f"\nframe {i}: keys = {sorted(s.keys())}")
    for k, v in s.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:16s} {tuple(v.shape)}")

    img = s["images"]
    check(tuple(img.shape[-2:]) == (H, W), f"images spatial size {tuple(img.shape[-2:])}")

    K = s["intrins"][0, 0].numpy()          # ego, camera 0
    fx, fy = K[0, 0], K[1, 1]
    # native OPV2V cameras have fx == fy; uniform scaling must preserve that
    check(abs(fx / fy - 1.0) < 1e-3, f"fx/fy = {fx/fy:.4f} (uniform scaling -> 1.0)")
    check(0 < K[0, 2] < W and 0 < K[1, 2] < H,
          f"principal point ({K[0,2]:.1f}, {K[1,2]:.1f}) inside {W}x{H}")

    dkey = next((k for k in ("depth_gt_img", "depth_gt", "depth") if k in s), None)
    if dkey is None:
        check(False, "no depth supervision key found -- tell Claude the key name")
    else:
        d = s[dkey]
        check(tuple(d.shape[-2:]) == (H // ds, W // ds),
              f"{dkey} spatial size {tuple(d.shape[-2:])} == feature map {(H // ds, W // ds)}")
        valid = (d >= 0).float().mean().item()
        check(valid > 0.05, f"{dkey} valid fraction {valid:.3f} (>0.05 expected)")
        print(f"    {dkey} range {d.min().item()} .. {d.max().item()}")

print("\nALL CHECKS PASSED -- safe to submit" if ok else
      "\nSOME CHECKS FAILED -- do not submit; paste this output")