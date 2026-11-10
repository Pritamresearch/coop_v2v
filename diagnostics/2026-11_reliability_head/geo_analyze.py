"""Analyse the geometric cross-agent test. Pre-registered rule: commit d276afd."""
import glob, json, sys
import numpy as np
D = "/mnt/scratch2/users/2641309/tmp_claude_diag"
fs = sorted(glob.glob(f"{D}/geo_[0-9].npz"))
d = [np.load(f) for f in fs]
X = {k: np.concatenate([x[k] for x in d]) for k in d[0]}
tie = X["errA"] == X["errB"]
X = {k: v[~tie] for k, v in X.items()}
print(f"{len(fs)} shards, {len(X['errA'])} agent pairs ({tie.sum()} ties dropped), "
      f"{len(np.unique(X['frame']))} frames")
rng = np.random.default_rng(0)

def winrate(name, sel):
    """sel: boolean 'score picks agent A'. Win if the picked agent has the smaller error."""
    return np.mean((sel & (X["errA"] < X["errB"])) | (~sel & (X["errB"] < X["errA"])))

SCORES = {                      # higher = predicted more reliable (entropy inverted)
    "G1 (-sender range)":  X["G1A"]  > X["G1B"],
    "G2 (|pred-ground|)":  X["G2A"]  > X["G2B"],
    "G2signed (secondary)":X["G2sA"] > X["G2sB"],
    "entropy (baseline)":  X["entA"] < X["entB"],
}
frames = X["frame"]; uf = np.unique(frames)
idx_by_f = {f: np.where(frames == f)[0] for f in uf}
R = {"n_pairs": int(len(X["errA"])), "n_frames": int(len(uf))}
print(f"\n{'score':>24} {'win rate':>10} {'95% CI (frame bootstrap)':>28}")
pt = {}
for name, sel in SCORES.items():
    w = winrate(name, sel); pt[name] = w
    boots = []
    for _ in range(2000):
        pick = rng.choice(uf, len(uf), replace=True)
        s = np.concatenate([idx_by_f[f] for f in pick])
        boots.append(np.mean((sel[s] & (X["errA"][s] < X["errB"][s])) |
                             (~sel[s] & (X["errB"][s] < X["errA"][s]))))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    R[name] = {"win": float(w), "ci": [float(lo), float(hi)]}
    print(f"{name:>24} {w:>10.4f} {f'[{lo:.4f}, {hi:.4f}]':>28}")

ent = pt["entropy (baseline)"]
print(f"\nDECISION RULE (pre-registered d276afd): CI lower bound > 0.5 AND above entropy ({ent:.4f})")
ok = {}
for name in ("G1 (-sender range)", "G2 (|pred-ground|)"):
    lo = R[name]["ci"][0]
    c1, c2 = lo > 0.5, lo > ent
    ok[name] = c1 and c2
    print(f"  {name:>24}: CI lo {lo:.4f}  >0.5 {'MET' if c1 else 'NOT MET'}  "
          f"| >entropy {'MET' if c2 else 'NOT MET'}  -> {'PASS' if ok[name] else 'FAIL'}")
lo_s = R["G2signed (secondary)"]["ci"][0]
print(f"  {'G2signed (secondary)':>24}: CI lo {lo_s:.4f}  (reported, not deciding)")
R["verdict"] = {k: bool(v) for k, v in ok.items()}
R["pass_any"] = bool(any(ok.values()))
print(f"\nVERDICT: {'PASS for ' + ', '.join(k for k, v in ok.items() if v) if R['pass_any'] else 'FAIL -- option 3 stands'}")
json.dump(R, open(sys.argv[1], "w"), indent=2)
