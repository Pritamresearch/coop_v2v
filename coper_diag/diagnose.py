"""Turn prediction dumps into a diagnosis report.

Usage
-----
python -m coper_diag.diagnose \
    --dump train_eval=dumps/train300_eval.pkl \
    --dump train_bnbatch=dumps/train300_bnbatch.pkl \
    --dump train_precisebn=dumps/train300_precisebn.pkl \
    --dump val_eval=dumps/val_eval.pkl \
    --dump test_eval=dumps/test_eval.pkl \
    --out report

Each dump is a pickle of a list of frame dicts (see metrics.py). Produce them
with `export_frames()` below from your existing evaluate() loop.

The interpretation thresholds in `interpret()` are heuristics I chose to make
the decision explicit; they are not from a paper. Read the raw numbers too.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from .metrics import full_report


def export_frames(model, loader, decode_fn, to_numpy_boxes, gt_fn, vis_fn=None, max_frames=None):
    """Generic exporter. Adapt the three callables to your code:
      decode_fn(model_out, batch) -> (boxes (N,7) tensor/array, scores (N,))
      gt_fn(batch) -> (M,7) array in the same frame/convention
      vis_fn(batch) -> (M,) bool or None
    Use decode with a LOW score threshold (e.g. 0.05) and generous top-k so AP
    sees the whole ranking."""
    import torch
    frames = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_frames and i >= max_frames:
                break
            out = model(batch)
            b, s = decode_fn(out, batch)
            frames.append({"frame_id": str(i),
                           "pred_boxes": to_numpy_boxes(b),
                           "pred_scores": np.asarray(s.detach().cpu() if hasattr(s, "detach") else s),
                           "gt_boxes": np.asarray(gt_fn(batch)),
                           "gt_visible": None if vis_fn is None else np.asarray(vis_fn(batch), bool)})
    return frames


def _g(rep, key, default=float("nan")):
    return rep.get(key, default) if rep else default


def interpret(reports: dict) -> list[str]:
    lines = []
    tr, trb, trp = reports.get("train_eval"), reports.get("train_bnbatch"), reports.get("train_precisebn")
    va, te = reports.get("val_eval"), reports.get("test_eval")

    if tr and va:
        a_tr, a_va = _g(tr, "AP50"), _g(va, "AP50")
        lines.append(f"[split] AP50 train-subset(eval mode) = {a_tr:.3f}, val = {a_va:.3f}.")
        if a_tr >= 0.6 and a_va < 0.35:
            lines.append("  -> Large generalisation gap with a healthy eval path: the model memorises "
                         "training scenes. Prioritise BEV-space augmentation (BDA) and GT visibility "
                         "filtering before architecture changes.")
        elif a_tr < 0.35:
            lines.append("  -> The model is poor on its OWN training frames in eval mode. This is not "
                         "classic overfitting; the eval path or the model is broken at scale "
                         "(check BN below, then depth). Do not add regularisation yet.")
    for name, alt in (("train_bnbatch", trb), ("train_precisebn", trp)):
        if tr and alt:
            d = _g(alt, "AP50") - _g(tr, "AP50")
            lines.append(f"[BN] {name} AP50 - eval-mode AP50 = {d:+.3f}.")
            if d > 0.05:
                lines.append("  -> BatchNorm running statistics are hurting eval. Switch BEV encoder+head "
                             "to GroupNorm and freeze backbone BN (one change, one run).")
    if va and te:
        d = _g(te, "AP50") - _g(va, "AP50")
        lines.append(f"[split] test AP50 - val AP50 = {d:+.3f}.")
        if abs(d) > 0.05:
            lines.append("  -> Val and test differ materially; report test (as published work does) and "
                         "keep val only for model selection.")
    for name, r in reports.items():
        if r and "AP50_visible_only" in r:
            a, v = _g(r, "AP50"), _g(r, "AP50_visible_only")
            lines.append(f"[visibility:{name}] AP50 all-GT = {a:.3f}, visible-only = {v:.3f}, "
                         f"GT not visible = {100*_g(r,'frac_gt_not_visible'):.1f}%.")
            if _g(r, "frac_gt_not_visible") > 0.15:
                lines.append("  -> A substantial share of training/eval GT is not visible to the ego "
                             "cameras. Train on visible GT only, and report AP under BOTH protocols, "
                             "stating which one the published baseline used.")
    for name, r in reports.items():
        y = r.get("yaw", {}) if r else {}
        if y.get("n", 0) > 20:
            lines.append(f"[yaw:{name}] raw RMSE {y['raw_rmse_deg']:.1f} deg, flip-aware RMSE "
                         f"{y['flipaware_rmse_deg']:.1f}, flip-aware median {y['flipaware_median_deg']:.1f}, "
                         f"flipped(>90) {100*y['frac_flipped_gt90']:.1f}%.")
            if y["frac_flipped_gt90"] > 0.1 and y["flipaware_median_deg"] < 15:
                lines.append("  -> Error is mostly 180 deg flips (harmless for IoU-based AP). Yaw is not "
                             "what limits AP50; look at localisation instead.")
            elif y["flipaware_median_deg"] > 30:
                lines.append("  -> Orientation is genuinely wrong, not just flipped. Consistent with the "
                             "head memorising road-aligned headings; BDA rotation directly targets this.")
        c = r.get("centre", {}) if r else {}
        if c.get("n", 0) > 20:
            rr, tt, bias = c["radial_rmse_m"], c["tangential_rmse_m"], c["radial_mean_bias_m(+ = too far)"]
            lines.append(f"[centre:{name}] radial RMSE {rr:.2f} m (bias {bias:+.2f}), tangential {tt:.2f} m.")
            if rr > 1.5 * tt:
                lines.append("  -> Error lies along camera rays: depth is the localisation bottleneck. "
                             "Run the depth-prior audit (ZoeDepth vs LiDAR) next.")
            if abs(bias) > 0.3:
                lines.append("  -> Systematic radial bias: depth scale error (check median_scale_gt_over_pred "
                             "of the ZoeDepth prior).")
    return lines


def to_markdown(reports, lines):
    keys = ["AP30", "AP50", "AP70", "AP50_visible_only", "frac_gt_not_visible", "n_gt"]
    out = ["# Detector diagnosis report", "", "| run | " + " | ".join(keys) + " |",
           "|---|" + "---|" * len(keys)]
    for name, r in reports.items():
        cells = []
        for k in keys:
            v = r.get(k, float("nan"))
            cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += ["", "## Interpretation (heuristic thresholds)", ""] + [f"- {l}" for l in lines]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="append", required=True, help="name=path.pkl")
    ap.add_argument("--out", default="report")
    ap.add_argument("--score_thr", type=float, default=0.2,
                    help="score threshold for yaw/centre statistics (not for AP)")
    a = ap.parse_args(argv)
    reports = {}
    for d in a.dump:
        name, path = d.split("=", 1)
        with open(path, "rb") as f:
            frames = pickle.load(f)
        reports[name] = full_report(frames, score_thr=a.score_thr)
    lines = interpret(reports)
    Path(a.out + ".json").write_text(json.dumps(reports, indent=2, default=float))
    Path(a.out + ".md").write_text(to_markdown(reports, lines))
    print(to_markdown(reports, lines))


if __name__ == "__main__":
    main()
