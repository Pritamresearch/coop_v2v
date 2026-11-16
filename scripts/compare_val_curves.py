"""Compare validation curves of two runs at matched epochs, from the logs only (no GPU).

  python scripts/compare_val_curves.py ego_armb60 no_fusion_lidardepth --at 10 20 30 40 50 60

Reads `[<run>] epoch <N> ...` headers and the `AP30/50/70=` line that follows, from
logs/<run>_<jobid>*.log, stripping carriage-return progress bars. Validation runs every
val_every epochs, so a value is carried for the epochs between validations; epochs with
no validation of their own are marked with '~' and the epoch actually measured is given.
"""
from __future__ import annotations
import argparse, glob, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EP = re.compile(r"^\[(?P<run>[^\]]+)\] epoch\s+(?P<ep>\d+)\s+lam=")
AP = re.compile(r"AP30/50/70=([0-9.]+)/([0-9.]+)/([0-9.]+)")


def curve(run: str) -> dict[int, tuple[float, float, float]]:
    pat = re.compile(rf"{re.escape(run)}_\d[A-Za-z0-9]*\.log$")
    out: dict[int, tuple[float, float, float]] = {}
    for path in sorted(p for p in glob.glob(f"logs/{run}_*.log") if pat.search(os.path.basename(p))):
        with open(path, "rb") as f:
            txt = f.read().decode("utf-8", "replace").replace("\r", "\n")
        ep = None
        for line in txt.split("\n"):
            if "it/s" in line:
                continue
            m = EP.match(line.strip())
            if m:
                ep = int(m.group("ep")); continue
            a = AP.search(line)
            if a and ep is not None:
                out[ep] = (float(a.group(1)), float(a.group(2)), float(a.group(3)))
                ep = None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs=2)
    ap.add_argument("--at", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60])
    a = ap.parse_args()

    A, B = (curve(r) for r in a.runs)
    for r, c in zip(a.runs, (A, B)):
        if not c:
            print(f"no validation points found for {r}"); return 1
        print(f"{r:24s} {len(c)} validation points, epochs {min(c)}-{max(c)}")

    def at(c, e):
        if e in c: return e, c[e]
        lower = [k for k in c if k <= e]
        if not lower: return None, None
        k = max(lower); return k, c[k]

    print(f"\n{'epoch':>6s} | {a.runs[0]:>26s} | {a.runs[1]:>26s} | {'AP50 diff':>10s}")
    print(f"{'':>6s} | {'AP30':>12s} {'AP50':>13s} | {'AP30':>12s} {'AP50':>13s} |")
    print("-" * 84)
    for e in a.at:
        ka, va = at(A, e); kb, vb = at(B, e)
        if va is None or vb is None:
            print(f"{e:>6d} | {'(no data)':>26s} | {'(no data)':>26s} |"); continue
        ma = " " if ka == e else "~"; mb = " " if kb == e else "~"
        d = va[1] - vb[1]
        print(f"{e:>6d} | {va[0]:>11.4f}{ma} {va[1]:>12.4f}{ma} | {vb[0]:>11.4f}{mb} {vb[1]:>12.4f}{mb} | "
              f"{d:>+10.4f}")
    print("-" * 84)
    print("~ = nearest earlier validation (val_every>1); the epoch shown is the requested one.")

    common = sorted(set(A) & set(B))
    if common:
        wins = sum(1 for e in common if A[e][1] > B[e][1])
        print(f"\nover {len(common)} shared epochs: {a.runs[0]} AP50 higher at {wins}, "
              f"{a.runs[1]} higher at {len(common) - wins - sum(1 for e in common if A[e][1] == B[e][1])}, "
              f"tied at {sum(1 for e in common if A[e][1] == B[e][1])}")
        early = [e for e in common if e <= 20]
        if early:
            da = sum(A[e][1] - B[e][1] for e in early) / len(early)
            print(f"mean AP50 difference over epochs <=20 (before either LR anneals much): {da:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
