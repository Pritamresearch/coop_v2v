"""Merge fusion_checks.py shards and report (a) recoverable fraction, (b) warp test, (c) subset AP.
Usage: fusion_checks_merge.py <out_json> <shard.pkl> [<shard.pkl> ...]

Subset AP (IoU 0.5 / 0.3 and centre distance <= 2 m): detections are restricted to the same range region as the
GT pool (far = >20 m from ego, near = <=20 m). Matching is greedy by score against every pool GT; a detection
matched to a pool GT outside the evaluated subset is ignored (neither TP nor FP); unmatched detections are FP.
Recoverable = >20 m from ego AND <20 m from >=1 collaborator AND inside that collaborator's camera FoV.
"""
import sys, os, json, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.metrics import _ap_from_matches, rotated_iou

out_json, files = sys.argv[1], sys.argv[2:]
F = []; meta = None
for f in files:
    d = pickle.load(open(f, "rb")); F += d["frames"]; meta = (d["full_epoch"], d["ego_epoch"])
print(f"{len(F)} val frames (full epoch {meta[0]}, ego epoch {meta[1]})\n")
R = {"frames": len(F), "full_epoch": meta[0], "ego_epoch": meta[1]}
rng = lambda x: np.hypot(x[:, 0], x[:, 1]) if len(x) else np.zeros(0)


def flags(fr):
    g = fr["gt"]; r = rng(g)
    rec = np.array([any(d < 20 and v for _, d, v in row) for row in fr["geo"]], bool)
    vis_any = np.array([any(v for _, _, v in row) for row in fr["geo"]], bool)
    return r, rec, vis_any


# ---------------- (a) ----------------
tot = {"far": 0, "rec": 0, "vis_any": 0}; band = {}
for fr in F:
    r, rec, vis = flags(fr); far = r > 20
    tot["far"] += far.sum(); tot["rec"] += (far & rec).sum(); tot["vis_any"] += (far & vis).sum()
    for lo, hi in ((20, 35), (35, 51.2), (51.2, 1e9)):
        s = (r >= lo) & (r < hi); b = band.setdefault(f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}", [0, 0]); b[0] += s.sum(); b[1] += (s & rec).sum()
n_collab = np.mean([fr["n_agents"] - 1 for fr in F])
print(f"(a) GT vehicles >20 m from ego: {tot['far']}  (mean collaborators per frame {n_collab:.2f})")
print(f"    within 20 m of a collaborator AND in its camera FoV ('recoverable'): {tot['rec']/tot['far']:.3f}")
print(f"    in some collaborator's camera FoV at any distance: {tot['vis_any']/tot['far']:.3f}")
for k, (n, m) in band.items(): print(f"    ego-range {k:>8} m: recoverable {m/max(n,1):.3f}  (n={n})")
R["a"] = {"far": int(tot["far"]), "recoverable_frac": tot["rec"] / tot["far"], "visible_any_frac": tot["vis_any"] / tot["far"],
          "by_band": {k: {"n": int(n), "recoverable_frac": m / max(n, 1)} for k, (n, m) in band.items()}}

# ---------------- (b) ----------------
print("\n(b) collaborator -> ego warp test (weight-independent, real data); error = |warped cell - true ego cell| x 0.4 m")
R["b"] = {}
keys = sorted({k for fr in F for k in fr["warp"]}, key=lambda k: (k != "none", k))
for key in keys:
    rows = [w for fr in F for w in fr["warp"].get(key, [])]
    if not rows: continue
    inside = [w for w in rows if w["warped"] is not None]
    err = np.array([np.hypot(w["warped"][0] - w["true"][0], w["warped"][1] - w["true"][1]) * 0.4 for w in inside])
    near = np.array([w.get("dist", 0) < 20 for w in inside]) if inside else np.zeros(0, bool)
    res = {"pairs": len(rows), "landed_in_grid": len(inside) / len(rows), "median_err_m": float(np.median(err)) if len(err) else None,
           "within_1cell": float(np.mean(err <= 0.4 * 1.5)) if len(err) else None, "within_2m": float(np.mean(err <= 2)) if len(err) else None}
    if key == "none":
        le = np.array([w["lift_err"] for w in rows]); res["lift_err_median_m"] = float(np.median(le)); res["lift_err_p95_m"] = float(np.percentile(le, 95))
        res["within_2m_collab_near20"] = float(np.mean(err[near] <= 2)) if near.any() else None
    R["b"][key] = res
    extra = f"  lift error median {res['lift_err_median_m']:.3f} m (p95 {res['lift_err_p95_m']:.3f})" if key == "none" else ""
    print(f"    {key:16s}: pairs={res['pairs']:5d}  in grid {res['landed_in_grid']:.3f}  median err {res['median_err_m']:.2f} m  "
          f"within 1 cell {res['within_1cell']:.3f}  within 2 m {res['within_2m']:.3f}{extra}")


# ---------------- subset AP ----------------
def subset_ap(variant, region, subset):
    res = {}
    for crit in ("iou50", "iou30", "c2m"):
        sc, tp, n_eval = [], [], 0
        for fr in F:
            g = fr["gt"]; r, rec, _ = flags(fr)
            pool = (r > 20) if region == "far" else (r <= 20)
            ev = pool & {"rec": rec, "nonrec": ~rec, "all": np.ones_like(rec)}[subset]
            gi = np.where(pool)[0]; n_eval += int(ev.sum())
            b, s, _ = fr["dets"][variant]
            if len(b):
                keep = (rng(b) > 20) if region == "far" else (rng(b) <= 20); b, s = b[keep], s[keep]
            used = np.zeros(len(g), bool)
            for k in np.argsort(-s):
                best, bj = -np.inf, -1
                for j in gi:
                    if used[j]: continue
                    if crit == "c2m":
                        v = -np.hypot(*(g[j, :2] - b[k, :2]))
                        if v >= -2.0 and v > best: best, bj = v, j
                    else:
                        v = rotated_iou(np.array([b[k, 0], b[k, 1], b[k, 3], b[k, 4], b[k, 6]]), np.array([g[j, 0], g[j, 1], g[j, 3], g[j, 4], g[j, 6]]))
                        if v >= (0.5 if crit == "iou50" else 0.3) and v > best: best, bj = v, j
                if bj >= 0:
                    used[bj] = True
                    if ev[bj]: sc.append(s[k]); tp.append(1)
                else:
                    sc.append(s[k]); tp.append(0)
        res[crit] = _ap_from_matches(sc, tp, n_eval); res["n_gt"] = n_eval
    return res


print("\n(c) AP on subsets (same frames): full = as trained (budget 0.4, gate on); full_b1_nogate = budget 1.0, gate off; ego = ego-only")
R["ap"] = {}
for region, subset, label in (("far", "rec", "far & recoverable"), ("far", "nonrec", "far & not recoverable"), ("near", "all", "near (<=20 m)")):
    for v in ("full", "full_b1_nogate", "ego"):
        a = subset_ap(v, region, subset); R["ap"][f"{label}|{v}"] = a
        print(f"    {label:22s} {v:15s}: AP50={a['iou50']:.4f}  AP30={a['iou30']:.4f}  centre AP@2m={a['c2m']:.4f}  (GT={a['n_gt']})")

# ---------------- bootstrap over frames for centre AP@2m (criterion fixed before the fine-tune results) ----------------
def per_frame_c2m(variant, region, subset):
    out = []
    for fr in F:
        g = fr["gt"]; r, rec, _ = flags(fr)
        pool = (r > 20) if region == "far" else (r <= 20)
        ev = pool & {"rec": rec, "nonrec": ~rec, "all": np.ones_like(rec)}[subset]
        gi = np.where(pool)[0]; b, s, _ = fr["dets"][variant]
        if len(b):
            keep = (rng(b) > 20) if region == "far" else (rng(b) <= 20); b, s = b[keep], s[keep]
        used = np.zeros(len(g), bool); sc, tp = [], []
        for k in np.argsort(-s):
            best, bj = -np.inf, -1
            for j in gi:
                if used[j]: continue
                v = -np.hypot(*(g[j, :2] - b[k, :2]))
                if v >= -2.0 and v > best: best, bj = v, j
            if bj >= 0:
                used[bj] = True
                if ev[bj]: sc.append(s[k]); tp.append(1)
            else:
                sc.append(s[k]); tp.append(0)
        out.append((sc, tp, int(ev.sum())))
    return out


def ap_of(frames_lists, sel):
    sc = [x for i in sel for x in frames_lists[i][0]]; tp = [x for i in sel for x in frames_lists[i][1]]
    return _ap_from_matches(sc, tp, sum(frames_lists[i][2] for i in sel))


rs = np.random.default_rng(0); B = 1000; nF = len(F)
L = {k: per_frame_c2m(*k) for k in [("full", "far", "rec"), ("full", "far", "nonrec"),
                                    ("ego", "far", "rec"), ("ego", "far", "nonrec"),
                                    ("full_b1_nogate", "far", "rec")]}
boot = {k: [] for k in L}; d_rec_nonrec, d_rec_ego, d_nonrec_ego, d_did = [], [], [], []
for _ in range(B):
    sel = rs.integers(0, nF, nF)
    a = {k: ap_of(v, sel) for k, v in L.items()}
    for k in L: boot[k].append(a[k])
    d_rec_nonrec.append(a[("full", "far", "rec")] - a[("full", "far", "nonrec")])
    g_rec    = a[("full", "far", "rec")]    - a[("ego", "far", "rec")]
    g_nonrec = a[("full", "far", "nonrec")] - a[("ego", "far", "nonrec")]
    d_rec_ego.append(g_rec); d_nonrec_ego.append(g_nonrec)
    # SPECIFICITY: the cooperative gain must be LARGER where collaborators can actually
    # see the vehicle. full_rec - full_nonrec does not test this -- recoverable vehicles
    # may simply be easier for both models -- so difference the two gains.
    d_did.append(g_rec - g_nonrec)
ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
print("\n    bootstrap 95% CI over frames (centre AP@2m):")
for k in L: print(f"      {k[0]:15s} far&{k[2]:6s}: {ap_of(L[k], range(nF)):.4f}  CI [{ci(boot[k])[0]:.4f}, {ci(boot[k])[1]:.4f}]")
print(f"      full: recoverable - not recoverable : CI [{ci(d_rec_nonrec)[0]:+.4f}, {ci(d_rec_nonrec)[1]:+.4f}]")
print(f"      recoverable: full - ego (same frames): CI [{ci(d_rec_ego)[0]:+.4f}, {ci(d_rec_ego)[1]:+.4f}]")
print(f"      NOT recoverable: full - ego          : CI [{ci(d_nonrec_ego)[0]:+.4f}, {ci(d_nonrec_ego)[1]:+.4f}]")
print(f"      SPECIFICITY (gain_rec - gain_nonrec) : CI [{ci(d_did)[0]:+.4f}, {ci(d_did)[1]:+.4f}]")

# ---- PRE-REGISTERED 2026-11-16 (diagnostics/2026-11_cooperation/PREREG_COOPERATION.md) ----
P_primary = ci(d_rec_ego)[0] > 0                 # ours > ego on far-recoverable, CI excludes 0
P_specific = ci(d_did)[0] > 0                    # that gain exceeds the far-not-recoverable gain
print(f"\n    PRE-REGISTERED 2026-11-16  primary (far-recoverable, ours>ego): {'MET' if P_primary else 'NOT MET'}"
      f";  specificity (gain_rec > gain_nonrec): {'MET' if P_specific else 'NOT MET'}"
      f"  ->  {'PASS' if P_primary and P_specific else 'FAIL'}")
print("    (the 2026-11-05 criterion below is SUPERSEDED: full_rec - full_nonrec is not a\n     cooperation test, and its 0.012 threshold came from an epoch-20 ego-only run)")
crit1 = ci(d_rec_nonrec)[0] > 0; crit2 = ci(boot[("full", "far", "rec")])[0] > 0.012
print(f"    SUCCESS CRITERION (fixed 2026-11-05): recoverable > not-recoverable: {'MET' if crit1 else 'NOT MET'};  "
      f"recoverable > 0.012 (ego-only, epoch 20): {'MET' if crit2 else 'NOT MET'}  ->  {'PASS' if crit1 and crit2 else 'FAIL'}")
R["bootstrap"] = {"|".join(k): ci(v) for k, v in boot.items()} | {"diff_rec_minus_nonrec": ci(d_rec_nonrec), "diff_rec_full_minus_ego": ci(d_rec_ego),
                                                                  "diff_nonrec_full_minus_ego": ci(d_nonrec_ego),
                                                                  "specificity_gain_rec_minus_gain_nonrec": ci(d_did),
                                                                  "prereg_2026_11_16_primary": bool(ci(d_rec_ego)[0] > 0),
                                                                  "prereg_2026_11_16_specificity": bool(ci(d_did)[0] > 0),
                                                                  "criterion_pass": bool(crit1 and crit2)}

cells = {v: np.mean([fr["dets"][v][2] for fr in F]) for v in ("full", "full_b1_nogate")}
print(f"\n    mean cells transmitted per frame: full {cells['full']:.0f}, full_b1_nogate {cells['full_b1_nogate']:.0f} (grid has 65536)")
R["cells"] = cells
json.dump(R, open(out_json, "w"), indent=2, default=float)
print(f"\nsaved {out_json}")
