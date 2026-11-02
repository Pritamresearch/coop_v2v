"""Depth-oracle diagnostics on LiDAR-equipped OPV2V train frames (val has no .pcd).
  eval     <ckpt> <tag>                : depth accuracy of predicted depth vs LiDAR, and detection ranking with
                                         predicted depth vs LiDAR depth substituted into the lift
  finetune <lidar|zoe> <n_frames> <out>: fine-tune epoch_40 with LiDAR or ZoeDepth depth supervision
  evalpred <tag> <ckpt> [<ckpt> ...]   : ranking metrics with each checkpoint's own predicted depth
Frames are split by scenario: every 5th scenario is held out for evaluation."""
import sys, os, time, random
sys.path.insert(0, "/mnt/scratch2/users/2641309/projects/coop_v2v")
sys.path.insert(0, "/mnt/scratch2/users/2641309/tmp_claude_diag/ab")
from dataclasses import replace
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.config import Config
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.models import lss_backbone as LB
from src.loss.coop_loss import coop_loss
from src.metrics import _to_device
from src.train import lr_at_epoch, lambda_schedule, gumbel_T, _set_seed
import abm

ROOT = "/mnt/scratch2/users/2641309/projects/coop_v2v"
A = "/mnt/scratch2/users/2641309/tmp_claude_diag/ab"
cfg = Config.load(f"{ROOT}/results/no_fusion_camfix/config.json"); abm.cfg = cfg
dev = torch.device("cuda")
FRUSTUM_D = torch.linspace(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K)
EDGES = np.linspace(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K + 1)

# ---- oracle injection: LSSBackbone._forward_agent with depth_prob optionally replaced ----
_ORACLE = {"mode": None, "dgt": None, "k": 0}
_ORIG_FORWARD_AGENT = LB.LSSBackbone._forward_agent   # mode: None | "hybrid" (LiDAR where labelled, pred elsewhere) | "lidar_only"


def _forward_agent_oracle(self, images, intrins, rots, trans, post_rots, post_trans):
    if _ORACLE["mode"] is None:
        return _ORIG_FORWARD_AGENT(self, images, intrins, rots, trans, post_rots, post_trans)
    cfg_ = self.cfg
    depth_logits_img, context = self.enc(images)
    depth_prob = F.softmax(depth_logits_img, dim=1)
    if _ORACLE["mode"] is not None:
        g = _ORACLE["dgt"][_ORACLE["k"]]; _ORACLE["k"] += 1      # this agent's (n_cams, fH, fW), -1 = unlabelled
        onehot = F.one_hot(g.clamp(min=0), depth_prob.shape[1]).permute(0, 3, 1, 2).to(depth_prob.dtype)
        m = (g >= 0).unsqueeze(1)
        depth_prob = torch.where(m, onehot, depth_prob if _ORACLE["mode"] == "hybrid" else torch.zeros_like(depth_prob))
    feats_5d = context.unsqueeze(2) * depth_prob.unsqueeze(1)
    n_cams_t, C_t, K_t, fH_t, fW_t = feats_5d.shape
    M = n_cams_t * K_t * fH_t * fW_t
    pts_ego = LB.frustum_to_ego(self.frustum, rots, trans, intrins, post_rots, post_trans)
    row, col, valid = self._bev_coords(pts_ego)
    feats_flat = feats_5d.permute(0, 2, 3, 4, 1).reshape(M, C_t)
    rc_flat = torch.stack([row.reshape(M), col.reshape(M)], dim=-1)
    valid_flat = valid.reshape(M)
    F_agent = LB.bev_pool(feats_flat[valid_flat], rc_flat[valid_flat], cfg_.grid_H, cfg_.grid_W)
    F_agent = self.bev_refine(F_agent.unsqueeze(0)).squeeze(0)
    k_idx = torch.arange(K_t, device=images.device).view(1, K_t, 1, 1).expand(n_cams_t, K_t, fH_t, fW_t).reshape(M)
    depth_feats = torch.eye(K_t, device=images.device, dtype=images.dtype)[k_idx] * depth_prob.reshape(M).unsqueeze(-1)
    depth_bev = LB.bev_pool(depth_feats[valid_flat], rc_flat[valid_flat], cfg_.grid_H, cfg_.grid_W)
    depth_bev = depth_bev / (depth_bev.sum(0, keepdim=True) + 1e-6)
    return F_agent, depth_bev, depth_logits_img


def run_model(model, scene, mode):
    _ORACLE["mode"] = mode
    _ORACLE["dgt"] = scene["depth_gt_img"] if mode else None        # (N_agents, n_cams, fH, fW)
    _ORACLE["k"] = 0
    return model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)


LB.LSSBackbone._forward_agent = _forward_agent_oracle


def datasets():
    dl = OPV2VDataset(cfg.train_dir, replace(cfg, augment=False, depth_source="lidar"), max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
    scen = sorted(set(f[0] for f in dl.frames))
    held = set(scen[::5])
    ev = [i for i, f in enumerate(dl.frames) if f[0] in held]
    tr = [i for i, f in enumerate(dl.frames) if f[0] not in held]
    print(f"LiDAR frames={len(dl)} scenarios={len(scen)} held-out scenarios={len(held)} eval frames={len(ev)} train frames={len(tr)}", flush=True)
    return dl, ev, tr


def load(cp):
    ck = torch.load(cp, map_location=dev)
    m = CoopDetector(cfg).to(dev); m.load_state_dict(ck["model"]); m.eval()
    return m


def pack(out, scene):
    gt = scene["gt"]
    return {"out": {k: out[k].float().cpu() for k in ("heat", "offset", "size", "yaw", "zh") if k in out},
            "boxes": gt["boxes"].cpu().numpy(), "oof": gt["oof"].cpu().numpy(), "mask": gt["mask"].cpu().bool()}


def fmt(r):
    keys = ["ap30", "ap50", "recall50", "cdAP@1m", "cdAP@2m", "AP50_oracle_ctr", "gt_ctr_pct_median", "gt_ctr_auc",
            "gt_ctr_is_localmax", "heat_pos_mean", "heat_neg_mean", "n_gt"]
    return "  ".join(f"{k}={r[k]:.4f}" for k in keys)


def cmd_eval(cp, tag):
    dl, ev, _ = datasets()
    dz = OPV2VDataset(cfg.train_dir, replace(cfg, augment=False, depth_source="zoedepth"), max_agents=1, use_gt_pose=cfg.use_gt_pose)
    zkey = {(f[0], f[1]): j for j, f in enumerate(dz.frames)}
    model = load(cp)
    res = {m: [] for m in ("pred", "hybrid", "lidar_only")}
    dstat = {"pred_exp": [], "pred_argmax": [], "lidar": [], "prior_k64": []}
    old_ctr = 4.0 + (50.0 - 4.0) / 64 * (np.arange(64) + 0.5)
    with torch.no_grad():   # self-check: oracle code path with zero labels must equal the original forward
        scene = _to_device(opv2v_collate([dl[ev[0]]]), dev)
        h0 = run_model(model, scene, None)["heat"]
        empty = dict(scene); empty["depth_gt_img"] = torch.full_like(scene["depth_gt_img"], -1)
        h1 = run_model(model, empty, "hybrid")["heat"]
        print(f"self-check oracle-path vs original (no labels): max|dheat|={float((h0-h1).abs().max()):.2e}", flush=True)
        assert torch.allclose(h0, h1, atol=1e-5), "oracle lift path diverges from original"
    t0 = time.time()
    with torch.no_grad():
        for n, i in enumerate(ev):
            scene = _to_device(opv2v_collate([dl[i]]), dev)
            for mode in res:
                out = run_model(model, scene, None if mode == "pred" else mode)
                res[mode].append(pack(out, scene))
                if mode == "pred":
                    lg = scene["depth_gt_img"][0].cpu(); msk = lg >= 0
                    prob = F.softmax(out["depth_img"][0].float(), dim=1).cpu()          # (n_cams, K, fH, fW)
                    exp_d = (prob * FRUSTUM_D.view(1, -1, 1, 1)).sum(1)
                    am_d = FRUSTUM_D[prob.argmax(1)]
                    l = lg[msk].numpy(); ld = (EDGES[l] + EDGES[l + 1]) / 2
                    z = dz[zkey[(dl.frames[i][0], dl.frames[i][1])]]["depth_gt_img"][0][msk].numpy()
                    dstat["pred_exp"].append(exp_d[msk].numpy()); dstat["pred_argmax"].append(am_d[msk].numpy())
                    dstat["lidar"].append(ld); dstat["prior_k64"].append(np.where(z >= 0, old_ctr[np.clip(z, 0, 63)], np.nan))
            if (n + 1) % 50 == 0: print(f"  {n+1}/{len(ev)} {(time.time()-t0)/(n+1):.2f}s/frame", flush=True)
    ld = np.concatenate(dstat["lidar"])
    print(f"\n[{tag}] depth accuracy at LiDAR-labelled pixels (ego cameras, n={len(ld)}):")
    from scipy.stats import spearmanr
    for k in ("pred_exp", "pred_argmax", "prior_k64"):
        d = np.concatenate(dstat[k]); ok = ~np.isnan(d); e = d[ok] - ld[ok]
        print(f"  {k:12s}: median ratio={np.median(d[ok]/ld[ok]):.3f}  median err={np.median(e):+.2f} m  MAE={np.abs(e).mean():.2f} m  "
              f"|err|<1m={(np.abs(e)<1).mean():.3f}  |err|<3m={(np.abs(e)<3).mean():.3f}  spearman={spearmanr(d[ok], ld[ok])[0]:.3f}")
    print(f"\n[{tag}] detection ranking on {len(ev)} held-out LiDAR frames:")
    for mode, fr in res.items():
        print(f"  lift={mode:10s} " + fmt(abm.metrics(fr)), flush=True)


def cmd_finetune(arm, n_frames, out_path):
    _set_seed(cfg.seed)
    _, _, tr = datasets()
    src = "lidar" if arm == "lidar" else "zoedepth"
    ds = OPV2VDataset(cfg.train_dir, replace(cfg, depth_source="lidar"), max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
    if src == "zoedepth":   # same frames, ZoeDepth labels: build the zoedepth dataset and map the same (scenario, ts)
        dz = OPV2VDataset(cfg.train_dir, replace(cfg, depth_source="zoedepth"), max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
        zkey = {(f[0], f[1]): j for j, f in enumerate(dz.frames)}
        idx = [zkey[(ds.frames[i][0], ds.frames[i][1])] for i in tr]; ds = dz
    else:
        idx = tr
    ck = torch.load(f"{A}/nf_epoch_40.pth", map_location=dev)
    model = CoopDetector(cfg).to(dev); model.load_state_dict(ck["model"])
    epoch = ck["epoch"] + 1
    bid = set(id(p) for p in model.backbone.parameters())
    params = [{"params": [p for p in model.parameters() if p.requires_grad and id(p) not in bid], "lr": cfg.lr},
              {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": cfg.lr * 0.1}]
    mults = [g["lr"] / cfg.lr for g in params]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay); opt.load_state_dict(ck["optimizer"])
    for g, m in zip(opt.param_groups, mults): g["lr"] = lr_at_epoch(epoch, cfg) * m
    lam, T = lambda_schedule(epoch, cfg), gumbel_T(epoch, cfg)
    gen = torch.Generator(); gen.manual_seed(1234)
    passes = -(-n_frames // len(idx))
    dlr = DataLoader(Subset(ds, idx * passes), batch_size=1, shuffle=True, generator=gen, num_workers=cfg.num_workers,
                     collate_fn=opv2v_collate, pin_memory=True,
                     worker_init_fn=lambda w: (np.random.seed(1234 + w), random.seed(1234 + w)))
    print(f"[{arm}] depth_source={src} frames/pass={len(idx)} total={n_frames}", flush=True)
    model.train(); opt.zero_grad(set_to_none=True); acc = {}; t0 = time.time()
    for step_i, scene in enumerate(dlr):
        if step_i >= n_frames: break
        scene = _to_device(scene, dev)
        out = model(scene, hard=False, gumbel_T=T)
        loss, parts = coop_loss(out, scene, cfg, lam=lam)
        if torch.isfinite(loss):
            (loss / cfg.grad_accum_steps).backward()
            for k in ("det_hm", "depth", "total"): acc.setdefault(k, []).append(parts.get(k, float("nan")))
        if (step_i + 1) % cfg.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], cfg.grad_clip)
            opt.step(); opt.zero_grad(set_to_none=True)
        if (step_i + 1) % 200 == 0:
            print(f"[{arm}] {step_i+1}/{n_frames} " + " ".join(f"{k}={np.mean(v[-200:]):.4f}" for k, v in acc.items())
                  + f" {(time.time()-t0)/(step_i+1):.2f}s/frame", flush=True)
    torch.save({"model": model.state_dict(), "epoch": epoch, "arm": arm}, out_path)
    print(f"[{arm}] saved {out_path}", flush=True)


def cmd_evalpred(tag, cps):
    dl, ev, _ = datasets()
    for cp in cps:
        model = load(cp); fr = []
        with torch.no_grad():
            for i in ev:
                scene = _to_device(opv2v_collate([dl[i]]), dev)
                fr.append(pack(run_model(model, scene, None), scene))
        print(f"[{tag}] {os.path.basename(cp):22s} own predicted depth: " + fmt(abm.metrics(fr)), flush=True)
        del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "eval": cmd_eval(sys.argv[2], sys.argv[3])
    elif c == "finetune": cmd_finetune(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif c == "evalpred": cmd_evalpred(sys.argv[2], sys.argv[3:])
