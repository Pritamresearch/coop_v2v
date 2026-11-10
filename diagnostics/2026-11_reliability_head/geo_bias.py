"""Is G2p just the inverted range prior +z in disguise? Pre-registered: commit f7fb034."""
import glob, json, sys
import numpy as np
D = "/mnt/scratch2/users/2641309/tmp_claude_diag"
rng = np.random.default_rng(0)

def load(pat):
    d = [np.load(f) for f in sorted(glob.glob(f"{D}/{pat}"))]
    return {k: np.concatenate([x[k] for x in d]) for k in d[0]}

def pred(X, s):
    return np.where(X[f"G2s{s}"] == 0.0, X[f"tg{s}"] + X[f"G2{s}"], X[f"tg{s}"] - X[f"G2{s}"])

def wr(sel, tie, eA, eB, fr, n_boot=2000):
    """win rate with ties = 0.5, plus a frame bootstrap CI."""
    right = ((sel & (eA < eB)) | (~sel & (eB < eA))).astype(float)
    v = np.where(tie, 0.5, right)
    uf = np.unique(fr); idx = {f: np.where(fr == f)[0] for f in uf}
    b = [v[np.concatenate([idx[f] for f in rng.choice(uf, len(uf), replace=True)])].mean()
         for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

R, verdicts = {}, {}
for pat, lab in (("geo_[0-9].npz", "no_fusion ep100"), ("geoB_[0-9].npz", "arm-B ep20")):
    X = load(pat)
    tie = X["errA"] == X["errB"]
    pA, pB = pred(X, "A"), pred(X, "B")
    g2 = X["G2pA"] > X["G2pB"]          # G2p picks A
    pz = pA > pB                        # +z picks A (larger predicted depth)
    dis = g2 != pz
    print(f"\n########## {lab}")
    print(f"  pairs {len(tie)}   G2p vs +z disagreement rate: {dis.mean():.4f} ({dis.sum()} pairs)")
    print(f"  of disagreement pairs, ties: {tie[dis].mean():.4f}")
    # overall, for context
    for name, sel in (("G2p", g2), ("+z (inverted prior)", pz)):
        w, lo, hi = wr(sel, tie, X["errA"], X["errB"], X["frame"])
        R.setdefault(lab, {})[f"{name} (all pairs)"] = [w, lo, hi]
        print(f"    all pairs   {name:>20}: {w:.4f} [{lo:.4f}, {hi:.4f}]")
    # the decisive comparison
    m = dis
    for name, sel in (("G2p", g2), ("+z (inverted prior)", pz)):
        w, lo, hi = wr(sel[m], tie[m], X["errA"][m], X["errB"][m], X["frame"][m])
        R[lab][f"{name} (disagree only)"] = [w, lo, hi]
        print(f"    DISAGREE    {name:>20}: {w:.4f} [{lo:.4f}, {hi:.4f}]")
    lo_g2 = R[lab]["G2p (disagree only)"][1]
    verdicts[lab] = lo_g2 > 0.5
    print(f"    -> G2p CI lower bound {lo_g2:.4f} > 0.5 ? {'MET' if verdicts[lab] else 'NOT MET'}")

ok = all(verdicts.values())
print(f"\nPRE-REGISTERED RULE (f7fb034): G2p CI lower bound > 0.5 on BOTH checkpoints")
for k, v in verdicts.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")
print(f"\nVERDICT: {'PASS -- G2 is not merely the inverted range prior; proceed to step 2' if ok else 'FAIL -- STOP: the honest framing is bias exploitation, not geometric reliability'}")
R["verdict"] = {k: bool(v) for k, v in verdicts.items()}
R["pass"] = bool(ok)
json.dump(R, open(sys.argv[1], "w"), indent=2)
