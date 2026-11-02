"""Depth audits: (A) is the ZoeDepth prior metric-accurate on CARLA? (B) is the
LSS depth distribution's entropy a calibrated uncertainty?

(A) decides whether depth supervision is a cause of poor localisation.
(B) is a research question your paper DEPENDS on: the utility
U = ||F|| * exp(-alpha * D) is only meaningful if the splatted entropy D
predicts where camera features are wrong. To my knowledge no camera
cooperative-perception paper reports that check (I have not done an
exhaustive search; treat it as a likely gap, not a certainty). Measuring it is
cheap and turns an assumption into evidence.

Camera convention: OpenCV axes (x right, y down, z forward). OPV2V/CARLA
extrinsics are UE4 axes; convert before calling (your loader already does
this if projections are correct).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr


# ----------------------------------------------------------------------------
# LiDAR -> sparse depth
# ----------------------------------------------------------------------------
def project_lidar(points_xyz, lidar2cam, K, img_hw, d_min=0.5):
    """Return (uv (N,2) float pixel coords, depth (N,)) keeping the nearest
    point per integer pixel. lidar2cam: 4x4 into OpenCV camera axes."""
    H, W = img_hw
    ph = np.c_[np.asarray(points_xyz, float)[:, :3], np.ones(len(points_xyz))]
    pc = (ph @ lidar2cam.T)[:, :3]
    z = pc[:, 2]
    m = z > d_min
    pc, z = pc[m], z[m]
    uv = pc @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    uv, z, ui, vi = uv[ok], z[ok], ui[ok], vi[ok]
    key = vi * W + ui
    order = np.lexsort((z, key))
    first = np.ones(len(order), bool)
    first[1:] = key[order][1:] != key[order][:-1]
    sel = order[first]
    return uv[sel], z[sel]


# ----------------------------------------------------------------------------
# (A) prior vs LiDAR
# ----------------------------------------------------------------------------
def depth_metrics(pred, gt):
    pred = np.asarray(pred, float); gt = np.asarray(gt, float)
    m = (gt > 0) & (pred > 0) & np.isfinite(pred)
    p, g = pred[m], gt[m]
    if len(g) == 0:
        return {"n": 0}
    ratio = g / p
    r = np.maximum(p / g, g / p)
    return {
        "n": int(len(g)),
        "abs_rel": float(np.mean(np.abs(p - g) / g)),
        "rmse_m": float(np.sqrt(np.mean((p - g) ** 2))),
        "delta<1.25": float(np.mean(r < 1.25)),
        "delta<1.25^2": float(np.mean(r < 1.25 ** 2)),
        # >1 means the prior is too NEAR; a value far from 1 = global scale error
        "median_scale_gt_over_pred": float(np.median(ratio)),
        "abs_rel_after_median_scaling": float(np.mean(np.abs(p * np.median(ratio) - g) / g)),
    }


def depth_metrics_by_range(pred, gt, bins=((0, 10), (10, 20), (20, 30), (30, 50), (50, 1e9))):
    out = {}
    gt = np.asarray(gt, float); pred = np.asarray(pred, float)
    for lo, hi in bins:
        m = (gt >= lo) & (gt < hi)
        out[f"{lo:g}-{hi:g}m"] = depth_metrics(pred[m], gt[m])
    return out


def sample_dense_at(depth_map, uv):
    """Nearest-pixel lookup of a dense map at float pixel coords."""
    H, W = depth_map.shape
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    return depth_map[v, u]


# ----------------------------------------------------------------------------
# Depth bins
# ----------------------------------------------------------------------------
def depth_bin_edges(d_min, d_max, n_bins, mode="UD"):
    """UD: uniform. LID: linear-increasing (CaDDN), finer near the camera."""
    if mode == "UD":
        return np.linspace(d_min, d_max, n_bins + 1)
    if mode == "LID":
        i = np.arange(n_bins + 1)
        return d_min + (d_max - d_min) * i * (i + 1) / (n_bins * (n_bins + 1))
    raise ValueError(mode)


def depth_to_bin(d, edges):
    """-1 for out of range."""
    d = np.asarray(d, float)
    b = np.searchsorted(edges, d, side="right") - 1
    return np.where((d >= edges[0]) & (d < edges[-1]), b, -1)


def bin_budget_report(gt_depths, edges, bev_half_range=20.0, cam_offset=2.0):
    """How much of the depth range is useful for the BEV grid you evaluate.
    With +/-20 m, the farthest relevant ray is ~sqrt(2)*20 + offset ~ 30 m,
    so bins beyond that are spent on points that are splatted outside the grid."""
    gt = np.asarray(gt_depths, float)
    useful = np.sqrt(2) * bev_half_range + cam_offset
    return {
        "bins_total": int(len(edges) - 1),
        "bins_beyond_useful_range": int(np.sum(edges[:-1] >= useful)),
        "useful_range_m": float(useful),
        "frac_lidar_beyond_dmax": float(np.mean(gt >= edges[-1])) if len(gt) else float("nan"),
        "frac_lidar_below_dmin": float(np.mean(gt < edges[0])) if len(gt) else float("nan"),
    }


# ----------------------------------------------------------------------------
# (B) calibration of the predicted categorical depth
# ----------------------------------------------------------------------------
def normalized_entropy(probs, axis=0):
    p = np.clip(np.asarray(probs, float), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis) / np.log(p.shape[axis])


def auroc(score, positive):
    """P(score_pos > score_neg) via Mann-Whitney U."""
    score = np.asarray(score, float); positive = np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def depth_uncertainty_calibration(probs, gt_depth, edges, large_err_m=2.0, n_ece_bins=15):
    """probs: (K, h, w) softmax depth distribution at feature resolution.
    gt_depth: (h, w) sparse LiDAR depth at the same resolution (0 = none).

    Reports whether the entropy H (what your utility penalises) ranks pixels
    by actual depth error:
      spearman(H, |E[d]-d_gt|) > 0 and AUROC(H -> error > large_err_m) >> 0.5
    support the utility's premise. ECE measures over/under-confidence of the
    argmax bin. Pool many images before drawing conclusions.
    """
    K = probs.shape[0]
    centres = 0.5 * (edges[:-1] + edges[1:])
    m = (gt_depth > edges[0]) & (gt_depth < edges[-1])
    P = probs[:, m].T                   # (n, K)
    g = gt_depth[m]
    if len(g) == 0:
        return {"n": 0}
    H = normalized_entropy(P, axis=1)
    exp_d = P @ centres
    err = np.abs(exp_d - g)
    gt_bin = depth_to_bin(g, edges)
    conf = P.max(1)
    correct = P.argmax(1) == gt_bin
    ece, rel = 0.0, []
    bins = np.linspace(0, 1, n_ece_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            ece += s.mean() * abs(conf[s].mean() - correct[s].mean())
            rel.append((float(conf[s].mean()), float(correct[s].mean()), int(s.sum())))
    rho = spearmanr(H, err).correlation if len(g) > 2 else float("nan")
    return {
        "n": int(len(g)),
        "spearman_entropy_vs_abs_err": float(rho),
        f"auroc_entropy_detects_err>{large_err_m}m": auroc(H, err > large_err_m),
        "ece_argmax_bin": float(ece),
        "top1_bin_acc": float(correct.mean()),
        "within1_bin_acc": float(np.mean(np.abs(P.argmax(1) - gt_bin) <= 1)),
        "mean_abs_err_expected_depth_m": float(err.mean()),
        "reliability(conf,acc,n)": rel,
        "n_bins": int(K),
    }


def bev_uncertainty_vs_localisation(D_at_gt, centre_err_m, large_err_m=1.0):
    """Object-level version for the paper: D (splatted entropy) sampled at each
    matched GT centre vs that object's centre error. This is the direct test
    of 'exp(-alpha*D) down-weights unreliable cells'."""
    D = np.asarray(D_at_gt, float); e = np.asarray(centre_err_m, float)
    return {"n": int(len(D)),
            "spearman_D_vs_centre_err": float(spearmanr(D, e).correlation) if len(D) > 2 else float("nan"),
            f"auroc_D_detects_err>{large_err_m}m": auroc(D, e > large_err_m)}
