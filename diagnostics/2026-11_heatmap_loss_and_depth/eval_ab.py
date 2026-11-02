"""Evaluate checkpoints on a fixed, evenly-spread val subset with ranking-based metrics.
Usage: eval_ab.py <n_val_frames> <out_tsv> <ckpt> [<ckpt> ...]"""
import sys, os, time
sys.path.insert(0, "/mnt/scratch2/users/2641309/projects/coop_v2v")
from dataclasses import replace
import numpy as np, torch
from src.config import Config
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.metrics import decode, Evaluator, _ap_from_matches, _peaks, _to_device, rotated_iou

ROOT = "/mnt/scratch2/users/2641309/projects/coop_v2v"
n_val, out_tsv, ckpts = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
cfg = Config.load(f"{ROOT}/results/no_fusion_camfix/config.json")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = OPV2VDataset(cfg.val_dir, replace(cfg, augment=False), max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
ds.augment = False
idxs = torch.linspace(0, len(ds) - 1, n_val).long().tolist()


def iou5(d, g):
    return rotated_iou(np.array([d[0], d[1], d[3], d[4], d[6]]), np.array([g[0], g[1], g[3], g[4], g[6]]))


def metrics(frames):
    r = {}
    for th in (0.01, 0.20):
        ev = Evaluator(cfg)
        for f in frames:
            b, s = decode(f["out"], cfg, score_thresh=th, topk=200)
            ev.update(b, s, f["boxes"], f["oof"], kb=0.0)
        m = ev.compute()
        sfx = "" if th == 0.01 else "@th0.20"
        for k in ("ap30", "ap50", "ap70", "recall50"):
            r[k + sfx] = m[k]
    # centre-distance AP + oracle-centre AP50 (threshold 0.01)
    cd = {1.0: ([], []), 2.0: ([], [])}; oc = ([], []); n_gt = 0
    pct, auc, ismax, hpos, hneg, rawpk = [], [], [], [], [], []
    for f in frames:
        b, s = decode(f["out"], cfg, score_thresh=0.01, topk=200)
        g = f["boxes"]; n_gt += len(g); order = np.argsort(-s)
        for rad in cd:
            used = np.zeros(len(g), bool)
            for k in order:
                j = -1
                if len(g):
                    dd = np.where(used, 1e9, np.linalg.norm(g[:, :2] - b[k, :2], axis=1)); j = int(dd.argmin())
                    if dd[j] >= rad: j = -1
                if j >= 0: used[j] = True
                cd[rad][0].append(s[k]); cd[rad][1].append(int(j >= 0))
        used = np.zeros(len(g), bool)
        for k in order:
            best, bj = 0.0, -1
            for j in range(len(g)):
                if used[j] or np.linalg.norm(g[j, :2] - b[k, :2]) > 4.0: continue
                dm = b[k].copy(); dm[0], dm[1] = g[j, 0], g[j, 1]
                v = iou5(dm, g[j])
                if v > best: best, bj = v, j
            ok = bj >= 0 and best >= 0.5
            if ok: used[bj] = True
            oc[0].append(s[k]); oc[1].append(int(ok))
        h = f["out"]["heat"]; m = f["mask"]
        rawpk.append(int((_peaks(h) > 0.2).sum()))
        if m.any():
            hv, nv = h[m], h[~m]
            srt = torch.sort(nv).values
            pct.append((torch.searchsorted(srt, hv).float() / len(nv)).numpy())
            ismax.append((_peaks(h)[m] > 0).numpy())
            hpos.append(float(hv.mean())); hneg.append(float(nv.mean()))
    r["cdAP@1m"] = _ap_from_matches(*cd[1.0], n_gt); r["cdAP@2m"] = _ap_from_matches(*cd[2.0], n_gt)
    r["AP50_oracle_ctr"] = _ap_from_matches(*oc, n_gt)
    p = np.concatenate(pct)
    r["gt_ctr_pct_median"] = float(np.median(p)); r["gt_ctr_auc"] = float(p.mean())
    r["neg_cells_above_typical_gt"] = float((1 - np.median(p)) * 65536)
    r["gt_ctr_is_localmax"] = float(np.concatenate(ismax).mean())
    r["heat_pos_mean"] = float(np.mean(hpos)); r["heat_neg_mean"] = float(np.mean(hneg))
    r["raw_peaks>0.2"] = float(np.mean(rawpk)); r["n_gt"] = n_gt
    return r


rows = []
for cp in ckpts:
    t0 = time.time()
    ck = torch.load(cp, map_location=dev)
    model = CoopDetector(cfg).to(dev); model.load_state_dict(ck["model"]); model.eval()
    frames = []
    with torch.no_grad():
        for i in idxs:
            scene = _to_device(opv2v_collate([ds[i]]), dev)
            out = model(scene, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)
            gt = scene["gt"]
            frames.append({"out": {k: out[k].float().cpu() for k in ("heat", "offset", "size", "yaw", "zh") if k in out},
                           "boxes": gt["boxes"].cpu().numpy(), "oof": gt["oof"].cpu().numpy(),
                           "mask": gt["mask"].cpu().bool()})
    r = metrics(frames); r["ckpt"] = os.path.basename(cp)
    rows.append(r)
    print(f"{r['ckpt']}  ({time.time()-t0:.0f}s)  " + "  ".join(f"{k}={v:.4f}" for k, v in r.items() if k != "ckpt"), flush=True)
    del model, frames; torch.cuda.empty_cache()

keys = [k for k in rows[0] if k != "ckpt"]
with open(out_tsv, "w") as fh:
    fh.write("ckpt\t" + "\t".join(keys) + "\n")
    for r in rows: fh.write(r["ckpt"] + "\t" + "\t".join(f"{r[k]:.5f}" for k in keys) + "\n")
print("wrote", out_tsv)
