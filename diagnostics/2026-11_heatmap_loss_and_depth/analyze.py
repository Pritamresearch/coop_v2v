"""Analyse cached real forward passes: threshold sweep, heatmap-vs-regression decomposition,
heat ranking at GT centres, and loss terms under current vs CenterNet normalisation."""
import sys
sys.path.insert(0, "/mnt/scratch2/users/2641309/projects/coop_v2v")
import numpy as np, torch
from src.config import Config
from src.baselines import make
from src.metrics import decode, Evaluator, _ap_from_matches, _peaks
from src.loss.detection import gaussian_focal_loss

cache, preset = sys.argv[1], sys.argv[2]
cfg = make(preset, Config())
C = torch.load(cache, map_location="cpu", weights_only=False)
print(f"== {cache}  epoch={C['epoch']}  frames={len(C['frames'])}")


def ap_sweep(frames, label):
    print(f"\n[{label}] real decode() threshold sweep (AP is global, greedy-by-score matching)")
    for th in [0.05, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        ev = Evaluator(cfg); npk = 0
        for f in frames:
            b, s = decode(f["out"], cfg, score_thresh=th, topk=200); npk += len(s)
            ev.update(b, s, f["gt"]["boxes"].numpy(), f["gt"]["oof"].numpy(), kb=0.0)
        r = ev.compute()
        print(f"  th={th:.2f} peaks/fr={npk/len(frames):6.1f} AP30={r['ap30']:.4f} AP50={r['ap50']:.4f} "
              f"AP70={r['ap70']:.4f} rec50={r['recall50']:.3f} prec50={r['precision50']:.3f}")


def iou5(d, g):
    from src.metrics import rotated_iou
    return rotated_iou(np.array([d[0], d[1], d[3], d[4], d[6]]), np.array([g[0], g[1], g[3], g[4], g[6]]))


def decomposition(frames, label, th=0.20):
    """Match decoded peaks to GT by centre distance, then swap in GT attributes to see which
    component is limiting IoU-AP."""
    modes = ["pred", "gt_yaw", "gt_size", "gt_size+yaw", "gt_centre"]
    sc = {m: [] for m in modes}; tp = {m: [] for m in modes}
    cd = {r: ([], []) for r in (1.0, 2.0)}
    n_gt = 0; near = {0.8: 0, 2.0: 0}
    for f in frames:
        b, s = decode(f["out"], cfg, score_thresh=th, topk=200)
        g = f["gt"]["boxes"].numpy(); n_gt += len(g)
        if len(g) and len(b):
            D = np.linalg.norm(b[:, None, :2] - g[None, :, :2], axis=-1)
            for r in near: near[r] += int((D.min(0) < r).sum())
        order = np.argsort(-s)
        for r in cd:  # centre-distance AP
            used = np.zeros(len(g), bool)
            for k in order:
                j = -1
                if len(g):
                    dd = np.where(used, 1e9, np.linalg.norm(g[:, :2] - b[k, :2], axis=1)); j = int(dd.argmin())
                    if dd[j] >= r: j = -1
                if j >= 0: used[j] = True
                cd[r][0].append(s[k]); cd[r][1].append(int(j >= 0))
        for m in modes:
            used = np.zeros(len(g), bool)
            for k in order:
                d = b[k].copy(); best, bj = 0.0, -1
                for j in range(len(g)):
                    if used[j] or np.linalg.norm(g[j, :2] - b[k, :2]) > 4.0: continue
                    dm = d.copy()
                    if "yaw" in m: dm[6] = g[j, 6]
                    if "size" in m: dm[3], dm[4] = g[j, 3], g[j, 4]
                    if m == "gt_centre": dm[0], dm[1] = g[j, 0], g[j, 1]
                    v = iou5(dm, g[j])
                    if v > best: best, bj = v, j
                ok = bj >= 0 and best >= 0.5
                if ok: used[bj] = True
                sc[m].append(s[k]); tp[m].append(int(ok))
    print(f"\n[{label}] decomposition at score_thresh={th}  (n_gt={n_gt})")
    for r in near: print(f"  GT with a decoded peak within {r}m: {near[r]/max(n_gt,1):.3f}")
    for r in cd:  print(f"  centre-distance AP@{r}m (heatmap-only quality): {_ap_from_matches(cd[r][0], cd[r][1], n_gt):.4f}")
    for m in modes: print(f"  AP50 with {m:12s}: {_ap_from_matches(sc[m], tp[m], n_gt):.4f}")


def heat_ranking(frames, label):
    pos_v, neg_q, rank_pct, ismax, auc = [], [], [], [], []
    stats = []
    for f in frames:
        h = f["out"]["heat"]; m = f["gt"]["mask"].bool(); gh = f["gt"]["heat"]
        stats.append((int(m.sum()), int(gh.eq(1.0).sum()), int((gh > 0.01).sum()), int((gh > 0.1).sum()),
                      int((gh > 0.5).sum()), h.numel()))
        if not m.any(): continue
        pk = _peaks(h)
        hv = h[m]; nv = h[~m]
        pos_v.append(hv.numpy()); neg_q.append(np.quantile(nv.numpy(), [0.5, 0.9, 0.99]))
        srt = torch.sort(nv).values
        rank_pct.append((torch.searchsorted(srt, hv).float() / len(nv)).numpy())
        ismax.append((pk[m] > 0).numpy())
        # count local-max peaks scoring above each GT centre's own heat (competitors)
        auc.append(np.mean([(nv < x).float().mean().item() for x in hv]))
    st = np.array(stats)
    print(f"\n[{label}] real GT stats per frame (mean): mask cells={st[:,0].mean():.1f}  gt==1 cells={st[:,1].mean():.1f}  "
          f"gt>0.01={st[:,2].mean():.0f}  gt>0.1={st[:,3].mean():.0f}  gt>0.5={st[:,4].mean():.0f}  total={st[0,5]}")
    print(f"  neg:pos ratio  vs gt==1 cells: {(st[:,5]-st[:,1]).sum()/max(st[:,1].sum(),1):.0f}:1   "
          f"vs gt>0.01 cells: {(st[:,5]-st[:,2]).sum()/max(st[:,2].sum(),1):.0f}:1")
    pv = np.concatenate(pos_v); rp = np.concatenate(rank_pct); im = np.concatenate(ismax); nq = np.array(neg_q).mean(0)
    print(f"  heat at GT centres: mean={pv.mean():.3f} median={np.median(pv):.3f}   neg cells q50/q90/q99={nq.round(3)}")
    print(f"  GT-centre percentile among neg cells: median={np.median(rp):.4f}; frac above neg-q99.9: {(rp>0.999).mean():.3f}")
    print(f"  => expected #neg cells outscoring a typical GT centre: {(1-np.median(rp))*65536:.0f} per frame")
    print(f"  GT centre is itself a 3x3 local max: {im.mean():.3f}")
    print(f"  pos-vs-neg cell AUC: {np.mean(auc):.4f}")


def loss_terms(frames, label):
    cur, cn, gp, gn, gpc, gnc = [], [], [], [], [], []
    for f in frames:
        h = f["out"]["heat"].clamp(1e-4, 1 - 1e-4); gh = f["gt"]["heat"]
        z = torch.logit(h).requires_grad_(True); p = torch.sigmoid(z)
        pos = gh.eq(1.0).float(); neg = 1 - pos
        lp = -((1 - p) ** 2) * torch.log(p) * pos; ln = -((1 - gh) ** 4) * p ** 2 * torch.log(1 - p) * neg
        npos = pos.sum().clamp(min=1); nneg = neg.sum()
        for (wp, wn, Lacc, Gp, Gn) in [(4 / npos, 1 / nneg, cur, gp, gn), (1 / npos, 1 / npos, cn, gpc, gnc)]:
            Lp = (lp.sum() * wp); Ln = (ln.sum() * wn)
            gP = torch.autograd.grad(Lp, z, retain_graph=True)[0].abs().sum().item()
            gN = torch.autograd.grad(Ln, z, retain_graph=True)[0].abs().sum().item()
            Lacc.append((Lp.item(), Ln.item())); Gp.append(gP); Gn.append(gN)
        assert abs(gaussian_focal_loss(f["out"]["heat"], gh, pos_weight=4.0).item() - sum(cur[-1])) < 1e-3
    cur, cn = np.array(cur).mean(0), np.array(cn).mean(0)
    print(f"\n[{label}] heatmap loss on the real outputs (pos term, neg term) and |dL/dlogit| summed:")
    print(f"  current  : L_pos={cur[0]:.3f} L_neg={cur[1]:.4f}  grad_pos={np.mean(gp):.3f} grad_neg={np.mean(gn):.4f}  "
          f"neg share={np.mean(gn)/(np.mean(gp)+np.mean(gn)):.3f}")
    print(f"  CenterNet: L_pos={cn[0]:.3f} L_neg={cn[1]:.2f}   grad_pos={np.mean(gpc):.3f} grad_neg={np.mean(gnc):.2f}  "
          f"neg share={np.mean(gnc)/(np.mean(gpc)+np.mean(gnc)):.3f}")


for split in ["val", "train"]:
    fr = [f for f in C["frames"] if f["split"] == split]
    if not fr: continue
    heat_ranking(fr, split)
    loss_terms(fr, split)
    ap_sweep(fr, split)
    decomposition(fr, split)
