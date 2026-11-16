"""Verify runs by their OUTPUTS, not by sacct COMPLETED (CPU, seconds).

Every job script until 2026-11-16 ended in `python ... | tee x.log` followed by a
trailing echo, with no `set -o pipefail`. tee's exit code won the pipeline, so a
python crash left the job reporting COMPLETED with exit 0 and printed "Done" as if
nothing was wrong. sacct therefore cannot be trusted for anything run before that
fix, and this script checks the artefacts instead:

  * last.pth exists and what epoch it holds
  * cfg.epochs / cfg.max_epoch -> what the run was supposed to reach
  * whether the checkpoint reached it (COMPLETE / SHORT / OVERRUN)
  * how many validation points the training log actually contains
  * the log's last non-progress line, and whether a traceback appears anywhere

  python scripts/verify_runs.py              # all runs with a config.json
  python scripts/verify_runs.py --run ego_armb60
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_ROOT

AP = re.compile(r"AP30/50/70=([0-9.]+)/([0-9.]+)/([0-9.]+)")


def log_facts(run: str) -> dict:
    """Read every log for this run, stripping carriage-return progress bars."""
    # Exact match only: logs/<run>_<jobid>.log. A plain prefix glob makes "no_fusion"
    # swallow no_fusion_lidardepth's logs and "ours_full" swallow ours_full_warpfix_ft's,
    # which silently attributes another run's validation history to this one.
    # The suffix must START with a digit (a job id, possibly with a hand-added tag such
    # as "10657729fix"). Requiring digits-only would drop that real 120-epoch log;
    # allowing anything lets "no_fusion" swallow no_fusion_lidardepth's history.
    pat = re.compile(rf"{re.escape(run)}_\d[A-Za-z0-9]*\.log$")
    logs = sorted(p for p in glob.glob(f"logs/{run}_*.log") if pat.search(os.path.basename(p)))
    if not logs:
        return {"logs": 0, "val_points": 0, "last_line": "(no log)", "traceback": False, "best_ap50": None}
    val, tb, last = 0, False, ""
    for path in logs:
        try:
            with open(path, "rb") as f:
                txt = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        for raw in txt.replace("\r", "\n").split("\n"):
            if "it/s" in raw or not raw.strip():
                continue
            if "Traceback (most recent call last)" in raw or "Error:" in raw:
                tb = True
            if AP.search(raw):
                val += 1
            last = raw.strip()
    aps = [float(m.group(2)) for p in logs
           for m in [AP.search(l) for l in open(p, "rb").read().decode("utf-8", "replace").replace("\r", "\n").split("\n")]
           if m]
    return {"logs": len(logs), "val_points": val, "last_line": last[:110],
            "traceback": tb, "best_ap50": max(aps) if aps else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    a = ap.parse_args()

    runs = [a.run] if a.run else sorted(
        os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{RESULTS_ROOT}/*/config.json"))

    print(f"{'run':28s} {'ckpt_ep':>7s} {'target':>6s} {'status':>9s} {'val_pts':>7s} {'best_AP50':>9s}  notes")
    print("-" * 118)
    bad = 0
    for run in runs:
        cfg_path = f"{RESULTS_ROOT}/{run}/config.json"
        if not os.path.exists(cfg_path):
            print(f"{run:28s} {'-':>7s} {'-':>6s} {'NO CONFIG':>9s}"); bad += 1; continue
        cfg = json.load(open(cfg_path))
        last = f"{RESULTS_ROOT}/checkpoints/{run}/last.pth"
        ep = "-"
        if os.path.exists(last):
            try:
                import torch
                ep = int(torch.load(last, map_location="cpu", weights_only=False)["epoch"])
            except Exception as e:                      # noqa: BLE001
                ep = f"ERR({type(e).__name__})"
        mx, tot = cfg.get("max_epoch", -1), cfg.get("epochs")
        target = mx if (isinstance(mx, int) and mx >= 0) else (tot - 1 if tot else None)
        if isinstance(ep, int) and target is not None:
            status = "COMPLETE" if ep >= target else "SHORT"
        elif not os.path.exists(last):
            status = "NO CKPT"
        else:
            status = "UNKNOWN"
        f = log_facts(run)
        notes = []
        if f["traceback"]:        notes.append("TRACEBACK in log")
        if f["logs"] == 0:        notes.append("no training log")
        if status == "SHORT":     notes.append(f"stopped {target - ep} epochs early")
        if status != "COMPLETE":  bad += 1
        ap50 = f"{f['best_ap50']:.4f}" if f["best_ap50"] is not None else "-"
        print(f"{run:28s} {str(ep):>7s} {str(target):>6s} {status:>9s} {f['val_points']:>7d} {ap50:>9s}  "
              f"{'; '.join(notes) if notes else ''}")
        if f["logs"]:
            print(f"{'':28s} last log line: {f['last_line']}")

    print("-" * 118)
    print(f"{len(runs)} runs checked; {bad} not verifiably COMPLETE.")
    print("'target' is max_epoch when set, else epochs-1 (the loop is range(start, epochs)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
