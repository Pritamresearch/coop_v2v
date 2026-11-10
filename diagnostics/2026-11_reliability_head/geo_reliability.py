"""Cross-agent test of purely geometric reliability (G1, G2). Pre-registered: commit d276afd.

For BEV cells observed by >= 2 agents, each via a LiDAR-labelled pixel whose TRUE 3D position falls
in that cell, ask whether the higher-scoring agent is the one with the smaller depth error.

Cell identity and per-agent error both come from LiDAR, so nothing depends on the model's own
(biased) predicted position.

    geo_reliability.py <ckpt> <shard> <n_shards> <n_frames> <out.npz>
"""
import sys, os, time
sys.path.insert(0, "/mnt/scratch2/users/2641309/projects/coop_v2v")
from dataclasses import replace
import numpy as np, torch, torch.nn.functional as F
from src.config import Config
from src.data.opv2v import OPV2VDataset, opv2v_collate, _load_pcd
from src.models.lss_backbone import LSSBackbone
from src.metrics import _to_device
from src.depth_bins import depth_bin_edges

ckpt, shard, n_shards, n_frames, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))

cfg = replace(Config(), device="cpu", augment=False, depth_source="lidar")
backbone = LSSBackbone(cfg)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
backbone.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("backbone.")})
backbone.eval()
edges = depth_bin_edges(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K, getattr(cfg, "depth_bins", "linear"))
ctr = torch.tensor((edges[:-1] + edges[1:]) / 2, dtype=torch.float32)
pts_d = torch.linspace(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K)

ds = OPV2VDataset(cfg.val_dir, cfg, max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
# Frames DISJOINT from the 200 used by the learned-head test (pre-registered).
used = set(np.linspace(0, len(ds) - 1, 200).astype(int).tolist())
avail = [i for i in range(len(ds)) if i not in used]
idxs = np.array(avail)[np.linspace(0, len(avail) - 1, n_frames).astype(int)][shard::n_shards]
print(f"shard {shard}/{n_shards}: {len(idxs)} frames, disjoint from the {len(used)} already used", flush=True)

rows = []
t0 = time.time()
with torch.no_grad():
    for n, i in enumerate(idxs):
        item = ds[int(i)]
        N = int(item["n_agents"])
        if N < 2:
            continue
        s = _to_device(opv2v_collate([item]), torch.device("cpu"))
        _, ts, cav_dirs = ds.frames[int(i)]
        per_agent = []
        for ai in range(N):
            lab = s["depth_gt_img"][ai]
            m = lab >= 0
            if not m.any():
                continue
            dl, _ = backbone.enc(s["images"][ai])
            p = F.softmax(dl.float(), dim=1)
            ent = (-(p * (p + 1e-9).log()).sum(1) / np.log(cfg.depth_K))
            pred = pts_d[p.argmax(1)]
            true = ctr[lab.clamp(min=0)]
            rots, trans, intr = s["rots"][ai], s["trans"][ai], s["intrins"][ai]
            nC, fH, fW = lab.shape
            gv, gu = torch.meshgrid(torch.arange(fH), torch.arange(fW), indexing="ij")
            # pixel centre in FULL-resolution image coords (the lift's frustum convention)
            uu = (gu.float() + 0.5) * cfg.lss_downsample
            vv = (gv.float() + 0.5) * cfg.lss_downsample

            # ---- ground plane z, per agent per frame, from that agent's OWN LiDAR ----
            pcd = os.path.join(cav_dirs[ai], f"{ts}.pcd")
            if not os.path.exists(pcd):
                continue
            pz = _load_pcd(pcd)[:, 2]
            z_g = float(np.percentile(pz, 2.0))        # LiDAR-derived (NOT available camera-only)
            # pose-only alternative: world ground is flat at z=0 in these CARLA towns, so in the
            # agent's own frame the ground sits at minus its sensor height.
            import yaml as _yaml
            _y = _yaml.load(open(os.path.join(cav_dirs[ai], f"{ts}.yaml")), Loader=_yaml.Loader)
            z_g_pose = -float(_y.get("lidar_pose", _y["true_ego_pos"])[2])

            cam_i, u_i, v_i, pr_i, tr_i, en_i, G1_i, G2_i, G2s_i, X_i, Y_i, tg_i, tgp_i, G2p_i = ([] for _ in range(14))
            for c in range(nC):
                mc = m[c]
                if not mc.any():
                    continue
                K = intr[c]
                # ray direction in this dataset's camera convention (x=depth, y=right, z=up)
                dx = torch.ones_like(uu)
                dy = (uu - K[0, 2]) / K[0, 0]
                dz = -(vv - K[1, 2]) / K[1, 1]
                dir_cam = torch.stack([dx, dy, dz], -1)                      # (fH,fW,3)
                dir_ego = dir_cam @ rots[c].T                                # (fH,fW,3)
                t_ego = trans[c]
                # flat-ground intersection: trans_z + t*dir_ego_z = z_g, depth along cam x = t
                denom = dir_ego[..., 2]
                t_ground_pose = (z_g_pose - t_ego[2]) / torch.where(denom.abs() < 1e-6, torch.full_like(denom, -1e-6), denom)
                t_ground = (z_g - t_ego[2]) / torch.where(denom.abs() < 1e-6,
                                                          torch.full_like(denom, -1e-6), denom)
                valid_g = (denom < -1e-6) & (t_ground > 0)                   # ray must point down
                t_ground = torch.where(valid_g, t_ground, torch.full_like(t_ground, float("nan")))

                pd, td = pred[c][mc], true[c][mc]
                tg = t_ground[mc]
                # TRUE 3D position of the labelled pixel, from LiDAR depth -> defines the cell
                dirn = dir_ego[mc]
                P = dirn * td.unsqueeze(-1) + t_ego.view(1, 3)
                cam_i.append(np.full(int(mc.sum()), c, np.int8))
                pr_i.append(pd.numpy()); tr_i.append(td.numpy()); en_i.append(ent[c][mc].numpy())
                X_i.append(P[:, 0].numpy()); Y_i.append(P[:, 1].numpy())
                G1_i.append(-(P[:, :2] - t_ego[:2].view(1, 2)).norm(dim=1).numpy())   # -(sender->cell range)
                tgp = torch.where(valid_g, t_ground_pose, torch.full_like(t_ground_pose, float('nan')))[mc]
                tgp_i.append(tgp.numpy())
                G2p_i.append(-(pd - tgp).abs().numpy())
                tg_i.append(tg.numpy())
                G2_i.append(-(pd - tg).abs().numpy())
                G2s_i.append(-torch.clamp(pd - tg, min=0).numpy())
            if not pr_i:
                continue
            per_agent.append(dict(agent=ai,
                                  x=np.concatenate(X_i), y=np.concatenate(Y_i),
                                  pred=np.concatenate(pr_i), true=np.concatenate(tr_i),
                                  ent=np.concatenate(en_i), G1=np.concatenate(G1_i),
                                  G2=np.concatenate(G2_i), G2s=np.concatenate(G2s_i),
                                  tg=np.concatenate(tg_i), tgp=np.concatenate(tgp_i),
                                  G2p=np.concatenate(G2p_i)))
        if len(per_agent) < 2:
            continue
        # ---- bin each agent's labelled pixels to BEV cells by TRUE position ----
        cellmaps = []
        for a in per_agent:
            col = ((a["x"] - cfg.bev_x_min) / cfg.cell_m).astype(int)
            row = ((a["y"] - cfg.bev_y_min) / cfg.cell_m).astype(int)
            ok = (col >= 0) & (col < cfg.grid_W) & (row >= 0) & (row < cfg.grid_H)
            key = row[ok].astype(np.int64) * cfg.grid_W + col[ok]
            err = np.abs(a["pred"][ok] - a["true"][ok])
            d = {}
            for j, k in enumerate(key):          # keep the smallest-error pixel per cell per agent
                if k not in d or err[j] < d[k][0]:
                    d[k] = (err[j], a["G1"][ok][j], a["G2"][ok][j], a["G2s"][ok][j], a["ent"][ok][j],
                            a["true"][ok][j], a["tg"][ok][j], a["G2p"][ok][j])
            cellmaps.append(d)
        shared = set(cellmaps[0])
        for d in cellmaps[1:]:
            shared |= set(d)
        for k in shared:
            present = [(ci, d[k]) for ci, d in enumerate(cellmaps) if k in d]
            if len(present) < 2:
                continue
            for p in range(len(present)):
                for q in range(p + 1, len(present)):
                    (_, A), (_, B) = present[p], present[q]
                    rows.append((int(i), A[0], B[0], A[1], B[1], A[2], B[2], A[3], B[3], A[4], B[4],
                                 A[5], B[5], A[6], B[6], A[7], B[7]))
        if (n + 1) % 5 == 0:
            print(f"  {n+1}/{len(idxs)} frames, {len(rows)} pairs, {(time.time()-t0)/(n+1):.1f}s/frame", flush=True)

r = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 17))
np.savez_compressed(out, frame=r[:, 0], errA=r[:, 1], errB=r[:, 2], G1A=r[:, 3], G1B=r[:, 4],
                    G2A=r[:, 5], G2B=r[:, 6], G2sA=r[:, 7], G2sB=r[:, 8], entA=r[:, 9], entB=r[:, 10],
                    trueA=r[:, 11], trueB=r[:, 12], tgA=r[:, 13], tgB=r[:, 14], G2pA=r[:, 15], G2pB=r[:, 16])
print(f"saved {out}: {len(r)} agent pairs from {len(idxs)} frames, {(time.time()-t0)/60:.1f} min")
