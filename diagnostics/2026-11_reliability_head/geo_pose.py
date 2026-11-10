"""Does G2 survive a POSE-ONLY ground plane (no LiDAR), which is what a camera-only system has?"""
import glob, json, sys
import numpy as np
D = "/mnt/scratch2/users/2641309/tmp_claude_diag"
d = [np.load(f) for f in sorted(glob.glob(f"{D}/geo_[0-9].npz"))]
X = {k: np.concatenate([x[k] for x in d]) for k in d[0]}
assert "G2pA" in X, "pose-ground re-extraction has not landed"
tie = X["errA"] == X["errB"]
X = {k: v[~tie] for k, v in X.items()}
rng = np.random.default_rng(0)
elevA, elevB = np.abs(X["trueA"] - X["tgA"]), np.abs(X["trueB"] - X["tgB"])
fin = np.isfinite(elevA) & np.isfinite(elevB) & np.isfinite(X["G2pA"]) & np.isfinite(X["G2pB"])
GROUND = fin & (elevA < 1.0) & (elevB < 1.0)
ELEV = fin & (elevA > 2.0) & (elevB > 2.0)

def ev(sel, m):
    w = np.mean((sel[m] & (X["errA"][m] < X["errB"][m])) | (~sel[m] & (X["errB"][m] < X["errA"][m])))
    fr = X["frame"][m]; uf = np.unique(fr); idx = {f: np.where(fr == f)[0] for f in uf}
    sa, ea, eb = sel[m], X["errA"][m], X["errB"][m]
    b = [np.mean((sa[s] & (ea[s] < eb[s])) | (~sa[s] & (eb[s] < ea[s])))
         for s in (np.concatenate([idx[f] for f in rng.choice(uf, len(uf), replace=True)]) for _ in range(1000))]
    lo, hi = np.percentile(b, [2.5, 97.5])
    return w, lo, hi

SC = {"G2  (LiDAR ground)": X["G2A"] > X["G2B"],
      "G2p (POSE ground)":  X["G2pA"] > X["G2pB"],
      "entropy":            X["entA"] < X["entB"]}
print(f"{'score':>20} {'ALL':>22} {'GROUND-LIKE':>22} {'ELEVATED (vehicles)':>22}")
R = {}
for n, sel in SC.items():
    cells = []
    for tag, m in (("all", fin), ("ground", GROUND), ("elev", ELEV)):
        w, lo, hi = ev(sel, m); R.setdefault(n, {})[tag] = [float(w), float(lo), float(hi)]
        cells.append(f"{w:.3f} [{lo:.3f},{hi:.3f}]".rjust(22))
    print(f"{n:>20} " + " ".join(cells))
gp = R["G2p (POSE ground)"]
print(f"\nPose-only ground plane, pre-registered rule (CI lo > 0.5 and > entropy {R['entropy']['all'][0]:.3f}):")
for tag in ("all", "elev"):
    lo = gp[tag][1]
    print(f"  {tag:>7}: win {gp[tag][0]:.3f} CI lo {lo:.3f} -> "
          f"{'PASS' if lo > 0.5 and lo > R['entropy'][tag][0] else 'FAIL'}")
json.dump(R, open(sys.argv[1], "w"), indent=2)
