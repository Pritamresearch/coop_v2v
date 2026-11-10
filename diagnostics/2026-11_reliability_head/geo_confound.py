"""Is G2's win rate real, or an artefact of most labelled pixels being road surface?

If a pixel lies on the ground, flat-ground depth ~ true depth, so G2 = -|pred - ground| is nearly
-|pred - true| = minus the error itself. That would make G2 a readout of the error rather than a
prediction of it -- and useless for VEHICLES, which are exactly what we detect and which sit above
the ground plane. Splits the same pairs by how far each pixel is from the ground plane.
"""
import glob, json, sys
import numpy as np
D = "/mnt/scratch2/users/2641309/tmp_claude_diag"
d = [np.load(f) for f in sorted(glob.glob(f"{D}/geo_[0-9].npz"))]
X = {k: np.concatenate([x[k] for x in d]) for k in d[0]}
assert "trueA" in X, "re-extraction has not landed yet"
tie = X["errA"] == X["errB"]
X = {k: v[~tie] for k, v in X.items()}
rng = np.random.default_rng(0)

# how far is each pixel above the ground plane, in metres of depth along the ray?
elevA, elevB = np.abs(X["trueA"] - X["tgA"]), np.abs(X["trueB"] - X["tgB"])
finite = np.isfinite(elevA) & np.isfinite(elevB)
print(f"{len(X['errA'])} pairs; both ground-depths finite for {finite.mean():.3f}")
print(f"|true - ground| quantiles (m): " + "  ".join(
    f"p{q}={np.nanpercentile(np.concatenate([elevA, elevB]), q):.2f}" for q in (10, 25, 50, 75, 90)))

def wins(sel, m):
    return np.mean((sel[m] & (X["errA"][m] < X["errB"][m])) | (~sel[m] & (X["errB"][m] < X["errA"][m])))

def boot(sel, m):
    fr = X["frame"][m]; uf = np.unique(fr)
    idx = {f: np.where(fr == f)[0] for f in uf}
    sa, ea, eb = sel[m], X["errA"][m], X["errB"][m]
    out = []
    for _ in range(1000):
        pick = rng.choice(uf, len(uf), replace=True)
        s = np.concatenate([idx[f] for f in pick])
        out.append(np.mean((sa[s] & (ea[s] < eb[s])) | (~sa[s] & (eb[s] < ea[s]))))
    return np.percentile(out, [2.5, 97.5])

SC = {"G1 (-sender range)": X["G1A"] > X["G1B"],
      "G2 (|pred-ground|)": X["G2A"] > X["G2B"],
      "G2signed":           X["G2sA"] > X["G2sB"],
      "entropy":            X["entA"] < X["entB"]}
# "ground-like" = both pixels within 1 m of the ground plane; "elevated" = both above 2 m
GROUND = finite & (elevA < 1.0) & (elevB < 1.0)
ELEV   = finite & (elevA > 2.0) & (elevB > 2.0)
print(f"\nground-like pairs: {GROUND.sum()} ({GROUND.mean():.3f})   elevated pairs: {ELEV.sum()} ({ELEV.mean():.3f})")
R = {}
print(f"\n{'score':>22} {'ALL':>22} {'GROUND-LIKE':>22} {'ELEVATED (vehicles)':>24}")
for name, sel in SC.items():
    cells = []
    for m in (finite, GROUND, ELEV):
        if m.sum() < 200:
            cells.append("n/a".rjust(22)); continue
        w = wins(sel, m); lo, hi = boot(sel, m)
        R.setdefault(name, {})[int(m.sum())] = [float(w), float(lo), float(hi)]
        cells.append(f"{w:.3f} [{lo:.3f},{hi:.3f}]".rjust(22))
    print(f"{name:>22} " + " ".join(cells))
print("\nIf G2 collapses toward 0.5 on ELEVATED pairs, its headline number is a ground-surface")
print("artefact and it is not usable for vehicle detection.")
json.dump(R, open(sys.argv[1], "w"), indent=2, default=float)
