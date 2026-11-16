"""Cooperation checks on real val data, one shard of frames per call (CPU).

  fusion_checks.py <shard> <n_shards> <n_frames_total> <full_ckpt> <ego_ckpt> <out_pkl> [budget_min]

Per frame (val, unaugmented, all agents up to cfg.max_agents):
 (a) For each GT vehicle (ego frame): position in every collaborator's LiDAR frame via WORLD poses from the
     agents' yaml files (independent of rel_pose), distance to that collaborator, and whether it projects
     into one of the collaborator's 4 cameras (in image, forward depth within [d_min, d_max]; occlusion ignored).
 (b) Collaborator -> ego warp test, weight-independent: for each (vehicle, collaborator, camera) where the vehicle is
     visible, take its pixel (u, v) and forward depth d in that camera, lift that single frustum point with the
     collaborator's own extrinsics through the real lss_backbone.frustum_to_ego, put a one-hot in the collaborator's
     BEV grid (real _bev_coords), warp it with the real topk_scheduler.warp_to_ego and the loader's rel_pose, and
     compare the argmax cell with the vehicle's true ego-frame cell (from world poses).
     Repeated with the training BEV rotation augmentation applied at fixed angles: as the code does it
     (rel yaw + angle) and with only the translation rotated (rel yaw unchanged); ego camera as a control.
 (c) Detections from: full model as trained (budget 0.4, gate on), full model with budget_frac=1.0 and the
     utility gate off, ego-only model (on the ego agent only). Raw boxes/scores are saved; metrics in the merge step.
"""
import sys, os, time, math, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
import numpy as np, torch, yaml
from src.config import Config, RESULTS_ROOT
from src.data.opv2v import OPV2VDataset, opv2v_collate, _lidar_pose_to_T
from src.data.bev_augment import bev_rotate_scene
from src.models.coop_detector import CoopDetector
from src.models import lss_backbone as LB
from src.models.topk_scheduler import warp_to_ego
from src.metrics import decode, _to_device

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
shard, n_shards, n_total = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
full_ck, ego_ck, out_pkl = sys.argv[4], sys.argv[5], sys.argv[6]
budget = float(sys.argv[7]) * 60 if len(sys.argv) > 7 else 150 * 60
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
t0 = time.time()


def load(run, ck):
    cfg = replace(Config.load(f"{RESULTS_ROOT}/{run}/config.json"), device="cpu", augment=False, depth_source="none")
    c = torch.load(ck, map_location="cpu", weights_only=False)
    m = CoopDetector(cfg); m.load_state_dict(c["model"]); m.eval()
    return cfg, m, c["epoch"]


fcfg, fmodel, fep = load("ours_full_lidardepth", full_ck)
ecfg, emodel, eep = load("no_fusion_lidardepth", ego_ck)
assert fmodel.fusion.use_utility_gate, "expected the trained full model to use the utility gate"
ds = OPV2VDataset(fcfg.val_dir, fcfg, max_agents=fcfg.max_agents, use_gt_pose=fcfg.use_gt_pose)
idx = np.linspace(0, len(ds) - 1, n_total).astype(int).tolist()[shard::n_shards]
H, W, cell, x0, y0 = fcfg.grid_H, fcfg.grid_W, fcfg.cell_m, fcfg.bev_x_min, fcfg.bev_y_min
ANGLES = (30.0, 90.0, 180.0)
print(f"shard {shard}/{n_shards}: {len(idx)} frames | full epoch {fep} | ego epoch {eep}", flush=True)


class FixedAngle:                       # stand-in rng for bev_rotate_scene
    def __init__(self, deg): self.deg = deg
    def uniform(self, lo, hi): return self.deg


def cell_of(xy):                        # same truncation as LSSBackbone._bev_coords
    return int((xy[1] - y0) / cell), int((xy[0] - x0) / cell)


def project(p, rots, trans, intr):
    """p: (3,) point in the agent's LiDAR frame -> list of (cam, u, v, depth) where visible."""
    vis = []
    for c in range(4):
        q = (p - trans[c]) @ rots[c]                       # R^T (p - t); CARLA camera: x fwd, y right, z up
        d = q[0]
        if d < fcfg.lss_d_min or d > fcfg.lss_d_max: continue
        u = intr[c][0, 0] * q[1] / d + intr[c][0, 2]; v = intr[c][1, 1] * (-q[2]) / d + intr[c][1, 2]
        if 0 <= u < 512 and 0 <= v < 384: vis.append((c, float(u), float(v), float(d)))
    return vis


def lift_warp(pairs, rots, trans, intr, rel):
    """pairs: list of (agent j, cam c, u, v, d). Returns (lifted point in agent frame, warped ego cell or None)."""
    lifted, onehots = [], []
    for j, c, u, v, d in pairs:
        fr = torch.tensor([[[[u, v, d]]]], dtype=torch.float32)                  # (K=1, fH=1, fW=1, 3)
        p = LB.frustum_to_ego(fr, torch.tensor(rots[j]), torch.tensor(trans[j]), torch.tensor(intr[j]), None, None)[c, 0, 0, 0]
        r, cc, ok = fmodel.backbone._bev_coords(p.view(1, 3))
        oh = torch.zeros(H, W)
        if bool(ok[0]): oh[int(r[0]), int(cc[0])] = 1.0
        lifted.append(p.numpy()); onehots.append((oh, j))
    out = []
    for (oh, j), p in zip(onehots, lifted):
        w = warp_to_ego(oh.unsqueeze(0), torch.tensor(rel[j]), fcfg)[0]
        out.append((p, None if float(w.max()) <= 0 else divmod(int(w.argmax()), W)))
    return out


recs = []
with torch.no_grad():
    for n, i in enumerate(idx):
        if time.time() - t0 > budget: print("time budget reached", flush=True); break
        scen, ts, cav_dirs = ds.frames[i]; cav_dirs = cav_dirs[: fcfg.max_agents]
        s = ds[i]; N = int(s["n_agents"])
        poses = []
        for cd in cav_dirs[:N]:
            y = yaml.load(open(os.path.join(cd, f"{ts}.yaml")), Loader=yaml.Loader)
            poses.append(_lidar_pose_to_T(y.get("lidar_pose", y["true_ego_pos"])).astype(np.float64))
        g = s["gt"]["boxes"].numpy(); rots, trans, intr = s["rots"].numpy(), s["trans"].numpy(), s["intrins"].numpy()
        rel = s["rel_pose"].numpy()
        # ---- (a) geometry of every GT vehicle w.r.t. every collaborator (world poses) ----
        world = (poses[0] @ np.c_[g[:, :3], np.ones(len(g))].T).T[:, :3] if len(g) else np.zeros((0, 3))
        geo = []                                            # per GT: list of (j, dist, visible cams)
        for k in range(len(g)):
            row = []
            for j in range(1, N):
                pc = (np.linalg.inv(poses[j]) @ np.r_[world[k], 1.0])[:3]
                row.append((j, float(np.hypot(pc[0], pc[1])), project(pc.astype(np.float32), rots[j], trans[j], intr[j]), pc))
            geo.append(row)
        # ---- (b) warp test: one (vehicle, collaborator) pair per visible camera hit ----
        pairs, truth = [], []
        for k, row in enumerate(geo):
            for j, dist, vis, pc in row:
                if vis:
                    c, u, v, d = vis[0]; pairs.append((j, c, u, v, d)); truth.append((k, j, pc, dist))
        warp = {"none": []}
        if pairs:
            for (p, cellw), (k, j, pc, dist) in zip(lift_warp(pairs, rots, trans, intr, rel), truth):
                warp["none"].append({"j": j, "dist": dist, "lift_err": float(np.linalg.norm(p[:2] - pc[:2])),
                                     "true": cell_of(g[k, :2]), "warped": cellw})
            for ang in ANGLES:
                for variant in ("code", "fixed_yaw"):
                    r2, t2, rel2, b2 = bev_rotate_scene(rots.copy(), trans.copy(), rel.copy(), g.copy(), rng=FixedAngle(ang))
                    if variant == "fixed_yaw": rel2[:, 2] = rel[:, 2]           # rotate translation only
                    res = []
                    for (p, cellw), (k, j, pc, dist) in zip(lift_warp(pairs, r2, t2, intr, rel2), truth):
                        res.append({"j": j, "dist": dist, "true": cell_of(b2[k, :2]), "warped": cellw})
                    warp[f"{variant}_{int(ang)}"] = res
            # ego control under augmentation: ego-visible vehicles, ego camera, no warp
            r2, t2, rel2, b2 = bev_rotate_scene(rots.copy(), trans.copy(), rel.copy(), g.copy(), rng=FixedAngle(90.0))
            rel2[0] = 0.0                       # ego row is masked out of fusion in the model; control = lift only
            ctl = []
            for k in range(len(g)):
                vis = project(g[k, :3].astype(np.float32), rots[0], trans[0], intr[0])
                if vis:
                    c, u, v, d = vis[0]
                    (p, cellw), = lift_warp([(0, c, u, v, d)], r2, t2, intr, rel2)
                    ctl.append({"true": cell_of(b2[k, :2]), "warped": cellw})
            warp["ego_control_90"] = ctl
        # ---- (c) detections ----
        sc = _to_device(opv2v_collate([s]), torch.device("cpu"))
        dets = {}
        o = fmodel(sc, hard=True, gumbel_T=fcfg.gumbel_T1, tau=fcfg.tau, budget_frac=fcfg.budget_frac)
        dets["full"] = decode(o, fcfg, score_thresh=fcfg.eval_score_thresh, topk=200) + (int(o["n_cells"]),)
        fmodel.fusion.use_utility_gate = False
        o = fmodel(sc, hard=True, gumbel_T=fcfg.gumbel_T1, tau=fcfg.tau, budget_frac=1.0)
        fmodel.fusion.use_utility_gate = True
        dets["full_b1_nogate"] = decode(o, fcfg, score_thresh=fcfg.eval_score_thresh, topk=200) + (int(o["n_cells"]),)
        ego_sc = {k: (v[:1] if k in ("images", "intrins", "rots", "trans", "rel_pose", "depth_gt_img") else v) for k, v in sc.items()}
        ego_sc["n_agents"] = 1
        o = emodel(ego_sc, hard=True, gumbel_T=ecfg.gumbel_T1, tau=ecfg.tau, budget_frac=ecfg.budget_frac)
        dets["ego"] = decode(o, ecfg, score_thresh=ecfg.eval_score_thresh, topk=200) + (0,)
        recs.append({"frame": i, "n_agents": N, "gt": g,
                     "geo": [[(j, d, len(v) > 0) for j, d, v, _ in row] for row in geo], "warp": warp, "dets": dets})
        if (n + 1) % 5 == 0:
            print(f"  {n+1}/{len(idx)} frames  {(time.time()-t0)/(n+1):.0f}s/frame", flush=True)
pickle.dump({"full_epoch": fep, "ego_epoch": eep, "frames": recs}, open(out_pkl, "wb"))
print(f"saved {out_pkl}: {len(recs)} frames in {(time.time()-t0)/60:.1f} min")
