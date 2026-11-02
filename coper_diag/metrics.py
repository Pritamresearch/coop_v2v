"""Detection metrics with the breakdowns needed to localise a train/val gap.

Frame record format (one dict per frame), produced by your evaluate():
    {
      "frame_id":    str,
      "pred_boxes":  (N, 7) float  x, y, z, l, w, h, yaw  (ego frame)
      "pred_scores": (N,)   float
      "gt_boxes":    (M, 7) float
      "gt_visible":  (M,)   bool   optional; False = not visible to ego cameras
    }

AP follows the OpenCOOD convention as I understand it (verify against your
own eval before comparing numbers): rotated BEV polygon IoU, greedy matching
per frame in descending score order, all frames pooled, VOC all-point
interpolated AP. Here it is extended with *ignore* GT: a prediction whose best
match is an ignored GT is neither TP nor FP, and ignored GT do not count in
the recall denominator. That is how "visible-only" and per-range AP are made
fair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import bev_iou_matrix, wrap_half_pi, wrap_pi


# ----------------------------------------------------------------------------
# AP
# ----------------------------------------------------------------------------
def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """All-point interpolated AP (VOC2010+ / OpenCOOD style)."""
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _match_frame(pred, scores, gt, ignore, iou_thr, pred_valid):
    """Return per-pred labels: 1 TP, 0 FP, -1 dropped (matched ignore / invalid)."""
    order = np.argsort(-scores, kind="stable")
    labels = np.full(len(pred), -1, dtype=np.int8)
    if len(pred) == 0:
        return order, labels
    iou = bev_iou_matrix(pred, gt) if len(gt) else np.zeros((len(pred), 0))
    taken = np.zeros(len(gt), dtype=bool)
    for i in order:
        if len(gt):
            cand = np.where(taken, -1.0, iou[i])
            j = int(np.argmax(cand))
            if cand[j] >= iou_thr:
                taken[j] = True
                labels[i] = -1 if ignore[j] else 1
                continue
        labels[i] = 0 if pred_valid[i] else -1
    return order, labels


def average_precision(frames, iou_thr=0.5, gt_ignore_fn=None, pred_valid_fn=None):
    """Pooled AP over frames.

    gt_ignore_fn(frame) -> (M,) bool   which GT to ignore (default: none)
    pred_valid_fn(frame) -> (N,) bool  which unmatched preds count as FP (default: all)
    """
    all_scores, all_labels, n_gt = [], [], 0
    for fr in frames:
        pred = np.asarray(fr["pred_boxes"], float).reshape(-1, 7)
        sc = np.asarray(fr["pred_scores"], float).reshape(-1)
        gt = np.asarray(fr["gt_boxes"], float).reshape(-1, 7)
        ign = gt_ignore_fn(fr) if gt_ignore_fn else np.zeros(len(gt), bool)
        val = pred_valid_fn(fr) if pred_valid_fn else np.ones(len(pred), bool)
        _, lab = _match_frame(pred, sc, gt, ign, iou_thr, val)
        keep = lab >= 0
        all_scores.append(sc[keep])
        all_labels.append(lab[keep])
        n_gt += int((~ign).sum())
    if n_gt == 0:
        return float("nan"), 0
    s = np.concatenate(all_scores) if all_scores else np.zeros(0)
    l = np.concatenate(all_labels) if all_labels else np.zeros(0)
    o = np.argsort(-s, kind="stable")
    tp = np.cumsum(l[o] == 1)
    fp = np.cumsum(l[o] == 0)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    return voc_ap(rec, prec), n_gt


def ap_by_range(frames, iou_thr, bins):
    """AP per GT-distance bin [(lo, hi), ...]. Out-of-bin GT are ignored;
    unmatched preds count as FP only if their own centre is in the bin."""
    out = {}
    for lo, hi in bins:
        def ign(fr, lo=lo, hi=hi):
            g = np.asarray(fr["gt_boxes"], float).reshape(-1, 7)
            d = np.hypot(g[:, 0], g[:, 1])
            return ~((d >= lo) & (d < hi))

        def val(fr, lo=lo, hi=hi):
            p = np.asarray(fr["pred_boxes"], float).reshape(-1, 7)
            d = np.hypot(p[:, 0], p[:, 1])
            return (d >= lo) & (d < hi)

        ap, n = average_precision(frames, iou_thr, ign, val)
        out[f"{lo:g}-{hi:g}m"] = {"ap": ap, "n_gt": n}
    return out


def ap_visible_only(frames, iou_thr):
    """AP where GT not visible to the ego are ignored (needs 'gt_visible')."""
    def ign(fr):
        g = np.asarray(fr["gt_boxes"]).reshape(-1, 7)
        v = fr.get("gt_visible")
        return np.zeros(len(g), bool) if v is None else ~np.asarray(v, bool)
    return average_precision(frames, iou_thr, ign)


# ----------------------------------------------------------------------------
# Localisation and yaw diagnostics (centre-distance matching, IoU-independent)
# ----------------------------------------------------------------------------
@dataclass
class MatchedPairs:
    pred: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    gt: np.ndarray = field(default_factory=lambda: np.zeros((0, 7)))
    gt_visible: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))


def match_by_centre(frames, max_dist=2.0, score_thr=0.2):
    """Greedy matching on BEV centre distance (<= max_dist m) per frame, in
    descending score order (the same priority AP uses), so the yaw/centre
    statistics describe the detections that actually drive AP rather than
    low-score clutter.

    Yaw error is measured on these pairs, NOT on IoU-matched pairs: IoU is
    blind to a 180 deg flip and strongly penalises ~90 deg errors, so IoU
    matching hides exactly the failure you are trying to see.
    """
    P, G, V = [], [], []
    for fr in frames:
        p = np.asarray(fr["pred_boxes"], float).reshape(-1, 7)
        s = np.asarray(fr["pred_scores"], float).reshape(-1)
        g = np.asarray(fr["gt_boxes"], float).reshape(-1, 7)
        vis = fr.get("gt_visible")
        vis = np.ones(len(g), bool) if vis is None else np.asarray(vis, bool)
        keep = s >= score_thr
        p, s = p[keep], s[keep]
        if len(p) == 0 or len(g) == 0:
            continue
        d = np.hypot(p[:, None, 0] - g[None, :, 0], p[:, None, 1] - g[None, :, 1])
        taken = np.zeros(len(g), bool)
        for i in np.argsort(-s, kind="stable"):
            di = np.where(taken, np.inf, d[i])
            j = int(np.argmin(di))
            if di[j] <= max_dist:
                taken[j] = True
                P.append(p[i:i + 1]); G.append(g[j:j + 1]); V.append(vis[j:j + 1])
    if not P:
        return MatchedPairs()
    return MatchedPairs(np.concatenate(P), np.concatenate(G), np.concatenate(V))


def yaw_report(pairs: MatchedPairs) -> dict:
    """Raw vs flip-aware yaw error.

    Reference values for interpreting raw RMSE (errors in degrees):
      uniform random heading          -> RMSE = 180/sqrt(3) ~ 103.9
      a fraction p of exact 180 flips -> RMSE = 180*sqrt(p)  (p=0.2 -> 80.5)
    So a raw RMSE of ~80 deg is equally consistent with "~20 % flips, rest
    perfect" and with "mostly random"; only the flip-aware numbers decide.
    """
    if len(pairs.pred) == 0:
        return {"n": 0}
    raw = np.degrees(np.abs(wrap_pi(pairs.pred[:, 6] - pairs.gt[:, 6])))
    flip = np.degrees(np.abs(wrap_half_pi(pairs.pred[:, 6] - pairs.gt[:, 6])))
    hist, edges = np.histogram(raw, bins=[0, 10, 30, 60, 90, 120, 150, 170, 180.001])
    return {
        "n": int(len(raw)),
        "raw_rmse_deg": float(np.sqrt(np.mean(raw ** 2))),
        "raw_median_deg": float(np.median(raw)),
        "flipaware_rmse_deg": float(np.sqrt(np.mean(flip ** 2))),
        "flipaware_median_deg": float(np.median(flip)),
        "frac_flipped_gt90": float(np.mean(raw > 90)),
        "frac_within_10deg_flipaware": float(np.mean(flip < 10)),
        "frac_near_90deg_flipaware(60-90)": float(np.mean(flip > 60)),
        "raw_hist_deg": {f"{edges[i]:g}-{min(edges[i+1],180):g}": int(hist[i]) for i in range(len(hist))},
    }


def centre_error_report(pairs: MatchedPairs, cam_xy=None) -> dict:
    """Decompose centre error into radial (along the camera ray) and
    tangential components.

    LSS places features along each pixel ray, so depth error shows up as
    RADIAL error. A large signed radial bias (mean != 0) points to a depth
    scale problem (e.g. monocular depth priors that are systematically too
    near/far); radial >> tangential points to depth in general; tangential >=
    radial points elsewhere (calibration, extrinsics, BEV resolution).

    cam_xy: (C, 2) camera optical centres in ego frame; each GT uses the
    nearest one. Default: ego origin (cameras sit within ~1-2 m of it, so the
    ray direction error is small beyond ~5 m).
    """
    if len(pairs.pred) == 0:
        return {"n": 0}
    g = pairs.gt[:, :2]
    delta = pairs.pred[:, :2] - g
    cams = np.zeros((1, 2)) if cam_xy is None else np.asarray(cam_xy, float).reshape(-1, 2)
    nearest = cams[np.argmin(np.linalg.norm(g[:, None] - cams[None], axis=2), axis=1)]
    ray = g - nearest
    u = ray / np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-6)
    radial = np.sum(delta * u, axis=1)
    tang = delta[:, 0] * (-u[:, 1]) + delta[:, 1] * u[:, 0]
    dist = np.linalg.norm(ray, axis=1)
    out = {
        "n": int(len(radial)),
        "centre_rmse_m": float(np.sqrt(np.mean(np.sum(delta ** 2, 1)))),
        "radial_rmse_m": float(np.sqrt(np.mean(radial ** 2))),
        "radial_mean_bias_m(+ = too far)": float(np.mean(radial)),
        "tangential_rmse_m": float(np.sqrt(np.mean(tang ** 2))),
        "size_rel_err_l": float(np.mean(np.abs(pairs.pred[:, 3] / pairs.gt[:, 3] - 1))),
        "size_rel_err_w": float(np.mean(np.abs(pairs.pred[:, 4] / pairs.gt[:, 4] - 1))),
    }
    # bias vs distance: a depth-scale error grows linearly with distance
    by = {}
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 60)]:
        m = (dist >= lo) & (dist < hi)
        if m.sum() >= 5:
            by[f"{lo}-{hi}m"] = {"n": int(m.sum()),
                                 "radial_bias_m": float(radial[m].mean()),
                                 "radial_rmse_m": float(np.sqrt((radial[m] ** 2).mean())),
                                 "tang_rmse_m": float(np.sqrt((tang[m] ** 2).mean()))}
    out["by_distance"] = by
    return out


def full_report(frames, cam_xy=None, range_bins=((0, 10), (10, 20), (20, 30)), score_thr=0.2):
    rep = {"n_frames": len(frames)}
    for thr in (0.3, 0.5, 0.7):
        ap, n = average_precision(frames, thr)
        rep[f"AP{int(thr*100)}"] = ap
    rep["n_gt"] = n
    if any(fr.get("gt_visible") is not None for fr in frames):
        for thr in (0.3, 0.5, 0.7):
            rep[f"AP{int(thr*100)}_visible_only"] = ap_visible_only(frames, thr)[0]
        vis = np.concatenate([np.asarray(fr["gt_visible"], bool) for fr in frames
                              if fr.get("gt_visible") is not None])
        rep["frac_gt_not_visible"] = float(1 - vis.mean()) if len(vis) else float("nan")
    rep["AP50_by_range"] = ap_by_range(frames, 0.5, list(range_bins))
    pairs = match_by_centre(frames, max_dist=2.0, score_thr=score_thr)
    rep["yaw"] = yaw_report(pairs)
    rep["centre"] = centre_error_report(pairs, cam_xy)
    if len(pairs.gt_visible) and (~pairs.gt_visible).any():
        inv = MatchedPairs(pairs.pred[~pairs.gt_visible], pairs.gt[~pairs.gt_visible])
        rep["yaw_on_invisible_gt"] = yaw_report(inv)
    return rep
