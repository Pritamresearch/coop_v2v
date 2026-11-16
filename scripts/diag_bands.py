"""Per-distance-band diagnosis (CPU).

  diag_bands.py ego     <run_name> <ckpt> <out_json> [N] [budget_min]
      train vs val per band on the same number of frames (alternating, unaugmented) + val depth per band
  diag_bands.py compare <full_run> <full_ckpt> <ego_run> <ego_ckpt> <out_json> [N] [budget_min]
      full model vs ego-only model on the SAME val frames: per-band AP + depth per band (ego cameras)

GT bands (range from ego): 0-20, 20-35, 35-51.2, >51.2 m (grid corners). Detections are binned by their own
range. Depth bands use the LiDAR depth of each labelled image cell: 4-20, 20-35, 35-51.2, 51.2-90 m.
Depth metrics: expected-depth ratio, argmax-depth ratio (median pred/LiDAR), share of predicted probability
mass within +-2 m of the LiDAR depth, share of cells whose argmax is within +-2 m.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
import numpy as np, torch, torch.nn.functional as F
from src.config import Config, RESULTS_ROOT
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.metrics import decode, Evaluator, _ap_from_matches, _to_device
from src.depth_bins import depth_bin_edges

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
GT_BANDS = [(0, 20), (20, 35), (35, 51.2), (51.2, 1e9)]
D_BANDS = [(4, 20), (20, 35), (35, 51.2), (51.2, 90)]
blab = lambda lo, hi: f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"


def load_model(run, ckpt):
    cfg = replace(Config.load(f"{RESULTS_ROOT}/{run}/config.json"), device="cpu", augment=False)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = CoopDetector(cfg); m.load_state_dict(ck["model"]); m.eval()
    return cfg, m, ck["epoch"]


def forward(cfg, model, ds, i):
    scene = _to_device(opv2v_collate([ds[i]]), torch.device("cpu"))
    out = model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)
    b, s = decode(out, cfg, score_thresh=cfg.eval_score_thresh, topk=200)
    g = scene["gt"]
    return {"b": b, "s": s, "g": g["boxes"].numpy(), "oof": g["oof"].numpy()}, scene, out


class DepthAcc:
    """Collect per-cell depth statistics for the ego cameras (labels at LiDAR cells)."""
    def __init__(self, cfg, model):
        self.pts = model.backbone.frustum[:, 0, 0, 2].clone()            # (K,) metres, this model's bins
        e = depth_bin_edges(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K, getattr(cfg, "depth_bins", "linear"))
        self.ctr = (e[:-1] + e[1:]) / 2 if getattr(cfg, "depth_bins", "linear") == "linear" else np.sqrt(e[:-1] * e[1:])
        self.rows = {k: [] for k in ("lidar", "exp", "argmax", "mass2")}

    def add(self, out, lab):
        m = lab >= 0
        if not m.any(): return
        prob = F.softmax(out["depth_img"][0].float(), dim=1).permute(0, 2, 3, 1)[m]     # (n, K)
        d = torch.tensor(self.ctr[lab[m].numpy()], dtype=torch.float32)               # LiDAR depth per cell
        near = (self.pts.view(1, -1) - d.view(-1, 1)).abs() <= 2.0
        self.rows["lidar"].append(d.numpy()); self.rows["exp"].append((prob * self.pts).sum(1).numpy())
        self.rows["argmax"].append(self.pts[prob.argmax(1)].numpy()); self.rows["mass2"].append((prob * near).sum(1).numpy())

    def report(self):
        r = {k: np.concatenate(v) for k, v in self.rows.items()}; L = r["lidar"]; res = {}
        for lo, hi in D_BANDS + [(4, 90)]:
            s = (L >= lo) & (L < hi)
            if not s.any(): continue
            res[blab(lo, hi)] = {"n_cells": int(s.sum()), "exp_ratio": float(np.median(r["exp"][s] / L[s])),
                                 "argmax_ratio": float(np.median(r["argmax"][s] / L[s])), "mass_within_2m": float(r["mass2"][s].mean()),
                                 "argmax_within_2m": float((np.abs(r["argmax"][s] - L[s]) <= 2).mean())}
        return res


def band_metrics(cfg, fr):
    """AP50/AP30/recall50 (IoU) and centre-distance AP at 1/2/4 m, overall and per GT band."""
    rng = lambda x: np.hypot(x[:, 0], x[:, 1])
    res = {}
    for lo, hi in [(0, 1e9)] + GT_BANDS:
        gf = lambda x: (rng(x) >= lo) & (rng(x) < hi)
        ev = Evaluator(cfg); cd = {r: ([], []) for r in (1.0, 2.0, 4.0)}; n_gt = 0
        for f in fr:
            g, oof, b, s = f["g"], f["oof"], f["b"], f["s"]
            kg = gf(g); g, oof = g[kg], oof[kg]
            if len(b): kd = gf(b); b, s = b[kd], s[kd]
            ev.update(b, s, g, oof, kb=0.0); n_gt += len(g)
            for rad, (sc, tp) in cd.items():
                used = np.zeros(len(g), bool)
                for k in np.argsort(-s):
                    j = -1
                    if len(g):
                        dd = np.where(used, np.inf, np.linalg.norm(g[:, :2] - b[k, :2], axis=1)); j = int(dd.argmin())
                        if dd[j] > rad: j = -1
                    if j >= 0: used[j] = True
                    sc.append(s[k]); tp.append(int(j >= 0))
        e = ev.compute()
        res["all" if hi > 1e8 and lo == 0 else blab(lo, hi)] = {"n_gt": n_gt, "ap50": e["ap50"], "ap30": e["ap30"], "recall50": e["recall50"],
                                                              **{f"cdAP@{r:g}m": _ap_from_matches(sc, tp, n_gt) for r, (sc, tp) in cd.items()}}
    return res


def show(title, res):
    print(title)
    for k, v in res.items():
        print(f"   {k:>8} m  GT={v['n_gt']:5d}  AP50={v['ap50']:.4f} AP30={v['ap30']:.4f} recall50={v['recall50']:.3f}  "
              f"centre AP@1/2/4m={v['cdAP@1m']:.4f}/{v['cdAP@2m']:.4f}/{v['cdAP@4m']:.4f}")


def show_depth(title, res):
    print(title)
    for k, v in res.items():
        print(f"   LiDAR {k:>9} m (n={v['n_cells']:7d}): exp ratio={v['exp_ratio']:.3f}  argmax ratio={v['argmax_ratio']:.3f}  "
              f"mass within 2 m={v['mass_within_2m']:.3f}  argmax within 2 m={v['argmax_within_2m']:.3f}")


mode = sys.argv[1]; t0 = time.time(); R = {}
if mode == "ego":
    run, ckpt, out_json = sys.argv[2:5]
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 300; budget = float(sys.argv[6]) * 60 if len(sys.argv) > 6 else 120 * 60
    cfg, model, ep = load_model(run, ckpt)
    ds = {"train": OPV2VDataset(cfg.train_dir, replace(cfg, depth_source="none"), max_agents=1),
          "val": OPV2VDataset(cfg.val_dir, replace(cfg, depth_source="lidar"), max_agents=1)}
    idx = {s: np.linspace(0, len(d) - 1, N).astype(int).tolist() for s, d in ds.items()}
    fr = {"train": [], "val": []}; dep = DepthAcc(cfg, model)
    print(f"{run} epoch {ep}: depth_bins={getattr(cfg,'depth_bins','linear')} depth_w={getattr(cfg,'depth_loss_weight','none')}", flush=True)
    with torch.no_grad():
        for k in range(N):
            if time.time() - t0 > budget: break
            for s in ("train", "val"):
                f, scene, out = forward(cfg, model, ds[s], idx[s][k]); fr[s].append(f)
                if s == "val": dep.add(out, scene["depth_gt_img"][0])
            if (k + 1) % 25 == 0: print(f"  {k+1} frames/split, {(time.time()-t0)/(k+1):.1f}s per pair", flush=True)
    n = len(fr["val"]); R = {"run": run, "epoch": ep, "frames_per_split": n}
    print(f"\n{run} epoch {ep}: {n} train + {n} val frames (unaugmented, evenly spread)\n")
    for s in ("train", "val"):
        R[s] = band_metrics(cfg, fr[s]); show(f"[{s}] per distance band", R[s])
    print("\n[train vs val] AP50 ratio per band (train/val):  " + "  ".join(
        (f"{k}: {R['train'][k]['ap50']/R['val'][k]['ap50']:.1f}x" if R['val'][k]['ap50'] > 0 else f"{k}: n/a (val 0)") for k in R["val"]))
    print("[train vs val] GT share per band:  " + "  ".join(
        f"{k}: train {R['train'][k]['n_gt']/R['train']['all']['n_gt']:.2f} / val {R['val'][k]['n_gt']/R['val']['all']['n_gt']:.2f}" for k in R["val"] if k != "all"))
    R["depth_val"] = dep.report(); show_depth("\n[val depth vs LiDAR, ego cameras]", R["depth_val"])
elif mode == "compare":
    frun, fck, erun, eck, out_json = sys.argv[2:7]
    N = int(sys.argv[7]) if len(sys.argv) > 7 else 200; budget = float(sys.argv[8]) * 60 if len(sys.argv) > 8 else 130 * 60
    fcfg, fmodel, fep = load_model(frun, fck); ecfg, emodel, eep = load_model(erun, eck)
    fds = OPV2VDataset(fcfg.val_dir, replace(fcfg, depth_source="lidar"), max_agents=fcfg.max_agents)
    eds = OPV2VDataset(ecfg.val_dir, replace(ecfg, depth_source="lidar"), max_agents=1)
    assert [f[:2] for f in fds.frames] == [f[:2] for f in eds.frames], "frame lists differ"
    idx = np.linspace(0, len(fds) - 1, N).astype(int).tolist()
    ffr, efr = [], []; fdep, edep = DepthAcc(fcfg, fmodel), DepthAcc(ecfg, emodel)
    print(f"full: {frun} epoch {fep} | ego-only: {erun} epoch {eep}", flush=True)
    with torch.no_grad():
        for k, i in enumerate(idx):
            if time.time() - t0 > budget: break
            f, scene, out = forward(fcfg, fmodel, fds, i); ffr.append(f); fdep.add(out, scene["depth_gt_img"][0])
            f, scene, out = forward(ecfg, emodel, eds, i); efr.append(f); edep.add(out, scene["depth_gt_img"][0])
            if (k + 1) % 10 == 0: print(f"  {k+1} val frames, {(time.time()-t0)/(k+1):.1f}s per frame (both models)", flush=True)
    n = len(ffr); R = {"full": {"run": frun, "epoch": fep}, "ego": {"run": erun, "epoch": eep}, "val_frames": n}
    print(f"\nsame {n} val frames for both models\n")
    R["full"]["bands"] = band_metrics(fcfg, ffr); show(f"[full model {frun} epoch {fep}] per distance band", R["full"]["bands"])
    R["ego"]["bands"] = band_metrics(ecfg, efr); show(f"[ego-only {erun} epoch {eep}] per distance band", R["ego"]["bands"])
    print("\n[full / ego] per band:  " + "  ".join(
        f"{k}: AP50 {R['full']['bands'][k]['ap50']:.4f} vs {R['ego']['bands'][k]['ap50']:.4f}, centre AP@2m "
        f"{R['full']['bands'][k]['cdAP@2m']:.4f} vs {R['ego']['bands'][k]['cdAP@2m']:.4f}" for k in R["ego"]["bands"]))
    R["full"]["depth"] = fdep.report(); show_depth("\n[full model depth vs LiDAR, ego cameras]", R["full"]["depth"])
    R["ego"]["depth"] = edep.report(); show_depth("[ego-only depth vs LiDAR, ego cameras]", R["ego"]["depth"])
json.dump(R, open(out_json, "w"), indent=2, default=float)
print(f"\nsaved {out_json}  total {(time.time()-t0)/60:.1f} min")
