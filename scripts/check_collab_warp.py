"""Pre-training check: collaborator -> ego feature placement on real data (no model weights).

    python scripts/check_collab_warp.py [--frames 8] [--split val]

For each GT vehicle visible in a collaborator camera: take its pixel (u, v) and forward depth, lift that point with
the collaborator's own extrinsics (lss_backbone.frustum_to_ego), place it in the collaborator's BEV grid
(LSSBackbone._bev_coords), warp it to the ego frame (topk_scheduler.warp_to_ego with the loader's rel_pose), and
compare with the vehicle's true ego-frame cell, computed from world poses in the agents' yaml files (independent of
rel_pose). Repeated with the training BEV rotation augmentation (data.bev_augment.bev_rotate_scene) at fixed angles,
plus an ego-camera control. Exits 1 if any case puts fewer than 99% of pairs within one cell.

Found the double-rotation bug of 2026-11-05: as coded then, augmented collaborators landed 19-58 m off.
"""
from __future__ import annotations
import argparse, os, sys, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace
import numpy as np, torch, yaml
from src.config import Config
from src.data.opv2v import OPV2VDataset, _lidar_pose_to_T
from src.data import bev_augment
from src.models import lss_backbone as LB
from src.models.topk_scheduler import warp_to_ego

ANGLES = (30.0, 90.0, 180.0, -120.0)


class _FixedAngle:
    """Deterministic stand-in for a Generator: fixes the angle, makes both flip coins come up
    heads (uniform() < 0.5), and returns the low end of any other range (e.g. the scale)."""
    def __init__(self, deg): self.deg = deg
    def uniform(self, lo=None, hi=None):
        if lo is None: return 0.0                  # flip coin -> always flip
        if (lo, hi) == (-self.deg, self.deg) or hi == self.deg: return self.deg
        return lo


def run_check(cfg, n_frames=8, split="val", rotate_fn=None, verbose=True) -> bool:
    rotate_fn = rotate_fn or bev_augment.bev_rotate_scene
    cfg = replace(cfg, augment=False, depth_source="none")
    ds = OPV2VDataset(cfg.val_dir if split == "val" else cfg.train_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
    bb = types.SimpleNamespace(cfg=cfg)                         # _bev_coords only needs .cfg
    x0, y0, cell = cfg.bev_x_min, cfg.bev_y_min, cfg.cell_m

    def cell_of(xy): return int((xy[1] - y0) / cell), int((xy[0] - x0) / cell)

    def visible(p, rots, trans, intr):
        for c in range(4):
            q = (p - trans[c]) @ rots[c]; d = q[0]
            if cfg.lss_d_min <= d <= cfg.lss_d_max:
                u = intr[c][0, 0] * q[1] / d + intr[c][0, 2]; v = intr[c][1, 1] * (-q[2]) / d + intr[c][1, 2]
                if 0 <= u < cfg.lss_img_W and 0 <= v < cfg.lss_img_H: return c, float(u), float(v), float(d)
        return None

    def place(j, c, u, v, d, rots, trans, intr, rel, dscale=1.0):
        fr = torch.tensor([[[[u, v, d]]]], dtype=torch.float32)
        p = LB.frustum_to_ego(fr, torch.tensor(rots[j]), torch.tensor(trans[j]), torch.tensor(intr[j]),
                              None, None, depth_scale=dscale)[c, 0, 0, 0]
        r, cc, ok = LB.LSSBackbone._bev_coords(bb, p.view(1, 3))
        if not bool(ok[0]): return None
        oh = torch.zeros(cfg.grid_H, cfg.grid_W); oh[int(r[0]), int(cc[0])] = 1.0
        w = warp_to_ego(oh.unsqueeze(0), torch.tensor(rel[j]), cfg)[0]
        return None if float(w.max()) <= 0 else divmod(int(w.argmax()), cfg.grid_W)

    errs = {}
    for i in np.linspace(0, len(ds) - 1, n_frames).astype(int):
        s = ds[int(i)]; N = int(s["n_agents"]); _, ts, cav_dirs = ds.frames[int(i)]
        poses = [_lidar_pose_to_T((y := yaml.load(open(os.path.join(cd, f"{ts}.yaml")), Loader=yaml.Loader))
                                  .get("lidar_pose", y["true_ego_pos"])).astype(np.float64) for cd in cav_dirs[:N]]
        g = s["gt"]["boxes"].numpy(); rots, trans, intr, rel = (s[k].numpy() for k in ("rots", "trans", "intrins", "rel_pose"))
        pairs = []
        for k in range(len(g)):
            pw = poses[0] @ np.r_[g[k, :3], 1.0]
            for j in range(1, N):
                vis = visible((np.linalg.inv(poses[j]) @ pw)[:3].astype(np.float32), rots[j], trans[j], intr[j])
                if vis: pairs.append((k, j, vis))
            vis0 = visible(g[k, :3].astype(np.float32), rots[0], trans[0], intr[0])
            if vis0: pairs.append((k, 0, vis0))
        cases = {"no augmentation": (rots, trans, rel, g, 1.0)}
        for a in ANGLES:
            cases[f"rot {a:+.0f}"] = rotate_fn(rots.copy(), trans.copy(), rel.copy(), g.copy(), rng=_FixedAngle(a))
        # flip and scale are the same bug class as the rotation warp bug: a reflection or a rescale
        # applied to the extrinsics but not to rel_pose/boxes silently misplaces collaborators.
        for a in (0.0, 90.0):
            cases[f"rot {a:+.0f} +flip"] = rotate_fn(rots.copy(), trans.copy(), rel.copy(), g.copy(),
                                                     rng=_FixedAngle(a), flip=True)
            for sc in (0.95, 1.05):
                cases[f"rot {a:+.0f} +scale {sc}"] = rotate_fn(rots.copy(), trans.copy(), rel.copy(), g.copy(),
                                                               rng=_FixedAngle(a), scale_range=(sc, sc))
            cases[f"rot {a:+.0f} +flip+scale"] = rotate_fn(rots.copy(), trans.copy(), rel.copy(), g.copy(),
                                                           rng=_FixedAngle(a), flip=True, scale_range=(0.95, 1.05))
        for name, case in cases.items():
            r2, t2, rel2, b2 = case[:4]
            dscale = case[4] if len(case) > 4 else 1.0
            for k, j, (c, u, v, d) in pairs:
                key = (name, "ego camera (control)" if j == 0 else "collaborators")
                rel_use = rel2.copy(); rel_use[0] = 0.0             # ego row is never warped in the model
                cw = place(j, c, u, v, d, r2, t2, intr, rel_use, dscale)
                if cw is None: continue                              # outside the agent's own BEV grid
                t = cell_of(b2[k, :2])
                errs.setdefault(key, []).append(np.hypot(cw[0] - t[0], cw[1] - t[1]) * cell)
    ok = True
    for (name, who), e in errs.items():
        e = np.array(e); frac = float(np.mean(e <= 1.5 * cell)); passed = frac >= 0.99
        ok &= passed
        if verbose:
            print(f"  {'PASS' if passed else 'FAIL'}  {name:20s} {who:22s} pairs={len(e):5d}  median error {np.median(e):6.2f} m  within 1 cell {frac:.3f}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    a = ap.parse_args()
    good = run_check(Config(), a.frames, a.split)
    print("collaborator warp check:", "PASSED" if good else "FAILED")
    sys.exit(0 if good else 1)
