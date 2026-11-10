"""Step 1: replicate G2 on arm-B, handle ties and NaN explicitly, and add the RANGE-PRIOR baseline.

The range prior is the key new baseline: CoCam4D (arXiv 2610.11577) weights observations by an
analytic sigma_z^2 ~ z^4/f^2, i.e. reliability is a monotone function of predicted depth alone.
A positional prior already beat our learned head, so G2 must be shown to beat a pure range prior.

Predicted depth is recovered exactly from the saved fields:
  G2 = -|pred - tg|, G2signed = -max(0, pred - tg)
  => pred = tg + G2 if G2signed == 0 (pred <= tg), else pred = tg - G2
"""
import glob, json, sys
import numpy as np

D = "/mnt/scratch2/users/2641309/tmp_claude_diag"
rng = np.random.default_rng(0)


def load(pat):
    d = [np.load(f) for f in sorted(glob.glob(f"{D}/{pat}"))]
    return {k: np.concatenate([x[k] for x in d]) for k in d[0]}


def recover_pred(X, s):
    tg, g2, g2s = X[f"tg{s}"], X[f"G2{s}"], X[f"G2s{s}"]
    return np.where(g2s == 0.0, tg + g2, tg - g2)


def evaluate(X, label):
    predA, predB = recover_pred(X, "A"), recover_pred(X, "B")
    nanA = ~np.isfinite(X["tgA"]) | ~np.isfinite(X["G2A"])
    nanB = ~np.isfinite(X["tgB"]) | ~np.isfinite(X["G2B"])
    anynan = nanA | nanB
    tie = X["errA"] == X["errB"]
    n = len(tie)
    print(f"\n########## {label}: {n} pairs, {len(np.unique(X['frame']))} frames")
    print(f"  ties (errA == errB): {tie.sum()} ({tie.mean():.3f})  -> decision coverage {1-tie.mean():.3f}")
    print(f"  pairs with an undefined ground depth (ray above horizon): {anynan.sum()} ({anynan.mean():.3f})")
    if tie.sum():
        print(f"    of TIED pairs, undefined-ground fraction: {anynan[tie].mean():.3f}  "
              f"(non-tied: {anynan[~tie].mean():.3f})")
        both0 = tie & (X["errA"] == 0)
        print(f"    tied pairs where BOTH errors are exactly 0: {both0.sum()} ({both0.mean():.3f} of all pairs)")

    scores = {
        "G2p (pose ground)":   (X["G2pA"], X["G2pB"], 1),
        "G2 (LiDAR ground)":   (X["G2A"], X["G2B"], 1),
        "range prior -z":      (-predA, -predB, 1),
        "range prior -z^2":    (-predA ** 2, -predB ** 2, 1),
        "G1 (-sender range)":  (X["G1A"], X["G1B"], 1),
        "entropy":             (X["entA"], X["entB"], -1),
    }
    out = {}
    print(f"\n  {'score':>20} {'ties=0.5':>22} {'ties dropped':>22}")
    for name, (a, b, sgn) in scores.items():
        sel = (sgn * a) > (sgn * b)
        right = ((sel & (X["errA"] < X["errB"])) | (~sel & (X["errB"] < X["errA"]))).astype(float)
        corr = np.where(tie, 0.5, right)
        cells = []
        for tag, m in (("ties=0.5", np.ones(n, bool)), ("dropped", ~tie)):
            v = corr[m] if tag == "ties=0.5" else corr[m]
            fr = X["frame"][m]; uf = np.unique(fr); idx = {f: np.where(fr == f)[0] for f in uf}
            w = v.mean()
            boots = [v[np.concatenate([idx[f] for f in rng.choice(uf, len(uf), replace=True)])].mean()
                     for _ in range(800)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out.setdefault(name, {})[tag] = [float(w), float(lo), float(hi)]
            cells.append(f"{w:.3f} [{lo:.3f},{hi:.3f}]".rjust(22))
        print(f"  {name:>20} " + " ".join(cells))

    # head-to-head on the pairs where the two disagree about which agent to pick
    for other in ("range prior -z", "range prior -z^2"):
        sa = X["G2pA"] > X["G2pB"]
        ob, oa = scores[other][1], scores[other][0]
        so = oa > ob
        dis = sa != so
        if dis.sum() > 100:
            m = dis & ~tie
            wg = ((sa[m] & (X["errA"][m] < X["errB"][m])) | (~sa[m] & (X["errB"][m] < X["errA"][m]))).mean()
            print(f"  head-to-head vs {other}: they disagree on {dis.mean():.3f} of pairs; "
                  f"on those, G2p is right {wg:.3f} of the time")
            out[f"vs {other}"] = {"disagree_frac": float(dis.mean()), "g2p_win_when_disagree": float(wg)}
    return out


R = {}
for pat, lab in (("geo_[0-9].npz", "no_fusion ep100"), ("geoB_[0-9].npz", "arm-B ep20 (replication)")):
    try:
        R[lab] = evaluate(load(pat), lab)
    except Exception as e:
        print(f"{lab}: {e}")
json.dump(R, open(sys.argv[1], "w"), indent=2)
print(f"\nsaved {sys.argv[1]}")
