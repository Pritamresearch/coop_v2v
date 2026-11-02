"""Synthetic tests: every metric/transform is checked against a known answer."""
import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn

from coper_diag import augment as AG
from coper_diag import depth_audit as DA
from coper_diag import metrics as M
from coper_diag import norm_tools as NT
from coper_diag import visibility as VIS
from coper_diag.geometry import bev_corners, bev_iou_matrix, wrap_half_pi, wrap_pi

RNG = np.random.default_rng(0)


def rand_boxes(n, rng=RNG, rng_m=18.0):
    xy = rng.uniform(-rng_m, rng_m, (n, 2))
    return np.c_[xy, np.full(n, 0.8), rng.uniform(3.8, 4.8, n), rng.uniform(1.7, 2.1, n),
                 np.full(n, 1.5), rng.uniform(-np.pi, np.pi, n)]


# ---------------------------------------------------------------- geometry
def test_iou_identity_flip_and_shift():
    b = rand_boxes(1)
    assert bev_iou_matrix(b, b)[0, 0] == pytest.approx(1.0)
    f = b.copy(); f[0, 6] += np.pi
    assert bev_iou_matrix(f, b)[0, 0] == pytest.approx(1.0)        # IoU is flip-blind
    s = b.copy(); s[0, 6] = 0; s2 = s.copy(); s2[0, 0] += s[0, 3] / 2   # half-length shift
    l = s[0, 3]
    assert bev_iou_matrix(s2, s)[0, 0] == pytest.approx((l / 2) / (1.5 * l), rel=1e-6)


def test_wraps():
    assert wrap_pi(3 * np.pi / 2) == pytest.approx(-np.pi / 2)
    assert wrap_half_pi(np.pi) == pytest.approx(0.0, abs=1e-9)
    assert abs(wrap_half_pi(np.pi / 2)) == pytest.approx(np.pi / 2)


# ---------------------------------------------------------------- AP
def make_frames(nf=40, ng=8, keep=1.0, noise=0.0, rng=None):
    rng = rng or np.random.default_rng(1)
    frames = []
    for i in range(nf):
        g = rand_boxes(ng, rng)
        p = g.copy(); p[:, :2] += rng.normal(0, noise, (ng, 2))
        k = rng.random(ng) < keep
        frames.append({"frame_id": str(i), "pred_boxes": p[k], "pred_scores": rng.uniform(0.5, 1, k.sum()),
                       "gt_boxes": g, "gt_visible": np.ones(ng, bool)})
    return frames


def test_ap_perfect_and_half():
    assert M.average_precision(make_frames(), 0.7)[0] == pytest.approx(1.0)
    ap = M.average_precision(make_frames(nf=200, keep=0.5), 0.5)[0]
    assert 0.45 < ap < 0.55


def test_ap_ignore_semantics():
    fr = make_frames(nf=30)
    for f in fr:  # half of GT invisible, and model predicts all GT
        f["gt_visible"] = np.arange(len(f["gt_boxes"])) % 2 == 0
    # predicting the invisible ones is neither rewarded nor punished
    assert M.ap_visible_only(fr, 0.5)[0] == pytest.approx(1.0)
    for f in fr:  # now predict ONLY visible ones: all-GT AP halves, visible AP stays 1
        f["pred_boxes"] = f["pred_boxes"][f["gt_visible"]]
        f["pred_scores"] = f["pred_scores"][f["gt_visible"]]
    assert M.ap_visible_only(fr, 0.5)[0] == pytest.approx(1.0)
    assert M.average_precision(fr, 0.5)[0] == pytest.approx(0.5, abs=0.02)


def test_ap_by_range_runs_and_is_perfect():
    r = M.ap_by_range(make_frames(), 0.5, [(0, 10), (10, 30)])
    assert all(v["ap"] == pytest.approx(1.0) for v in r.values())


# ---------------------------------------------------------------- yaw
def test_yaw_flip_mixture_reference_values():
    rng = np.random.default_rng(2)
    g = rand_boxes(5000, rng)
    p = g.copy()
    flip = rng.random(5000) < 0.2
    p[flip, 6] += np.pi
    pairs = M.MatchedPairs(p, g, np.ones(5000, bool))
    y = M.yaw_report(pairs)
    assert y["raw_rmse_deg"] == pytest.approx(180 * np.sqrt(flip.mean()), rel=1e-3)   # ~80 deg
    assert y["flipaware_rmse_deg"] < 1e-6
    assert y["frac_flipped_gt90"] == pytest.approx(flip.mean())
    p2 = g.copy(); p2[:, 6] = rng.uniform(-np.pi, np.pi, 5000)
    y2 = M.yaw_report(M.MatchedPairs(p2, g))
    assert y2["raw_rmse_deg"] == pytest.approx(180 / np.sqrt(3), rel=0.03)            # ~103.9
    assert y2["flipaware_rmse_deg"] == pytest.approx(90 / np.sqrt(3), rel=0.03)       # ~52


def test_centre_decomposition_detects_depth_scale():
    g = rand_boxes(2000, np.random.default_rng(3))
    g = g[np.hypot(g[:, 0], g[:, 1]) > 3]
    p = g.copy(); p[:, :2] *= 1.1          # 10 % too far along rays from origin
    c = M.centre_error_report(M.MatchedPairs(p, g))
    assert c["radial_mean_bias_m(+ = too far)"] > 0.5
    assert c["tangential_rmse_m"] < 1e-6
    p2 = g.copy(); p2[:, 1] += 0.5         # lateral shift: mixes components
    c2 = M.centre_error_report(M.MatchedPairs(p2, g))
    assert c2["tangential_rmse_m"] > 0.2


def test_match_by_centre_prefers_high_score():
    g = rand_boxes(1)
    good, junk = g.copy(), g.copy()
    junk[0, 6] += 1.0; junk[0, 0] += 0.1
    fr = [{"pred_boxes": np.r_[junk, good], "pred_scores": np.array([0.3, 0.9]), "gt_boxes": g}]
    pairs = M.match_by_centre(fr, 2.0, 0.2)
    assert len(pairs.pred) == 1 and pairs.pred[0, 6] == pytest.approx(g[0, 6])


# ---------------------------------------------------------------- augmentation
def _cam(yaw_deg):
    """cam2ego for a camera at (1,0,1.6) looking along ego yaw_deg, OpenCV axes."""
    t = np.radians(yaw_deg)
    fwd = np.array([np.cos(t), np.sin(t), 0]); right = np.array([np.sin(t), -np.cos(t), 0]); down = np.array([0, 0, -1])
    T = np.eye(4); T[:3, :3] = np.stack([right, down, fwd], 1); T[:3, 3] = [1.0, 0, 1.6]
    return T


def _proj(K, ego2cam, X):
    pc = (np.c_[X, np.ones(len(X))] @ ego2cam.T)[:, :3]
    uv = pc @ K.T
    return uv[:, :2] / uv[:, 2:3], pc[:, 2]


@pytest.mark.parametrize("seed", range(6))
def test_ida_bda_projection_consistency(seed):
    rng = np.random.default_rng(seed)
    K = np.array([[400.0, 0, 400], [0, 400, 300], [0, 0, 1]])
    c2e = _cam(rng.uniform(-180, 180))
    X = (c2e @ np.c_[rng.uniform(-6, 6, (50, 2)), rng.uniform(5, 40, 50), np.ones(50)].T).T[:, :3]
    uv, z = _proj(K, np.linalg.inv(c2e), X)
    ip = AG.sample_ida(rng); Mx = AG.ida_matrix(ip); K2 = AG.ida_apply_intrinsics(Mx, K)
    A = AG.bda_matrix(AG.sample_bda(rng))
    X2 = AG.bda_apply_points(A, X)
    uv2, z2 = _proj(K2, np.linalg.inv(A @ c2e), X2)
    expect = (np.c_[uv, np.ones(len(uv))] @ Mx.T)[:, :2]
    np.testing.assert_allclose(uv2, expect, atol=1e-6)
    np.testing.assert_allclose(z2, z, rtol=1e-9)       # depth labels unchanged by BDA


def test_ida_image_warp_matches_matrix():
    rng = np.random.default_rng(7)
    img = np.zeros((600, 800), np.float32)
    u, v = 517, 233
    img[v - 3:v + 4, u - 3:u + 4] = 1.0          # 7x7 blob centred on (u, v)
    checked = 0
    for _ in range(20):
        ip = AG.sample_ida(rng)
        Mx = AG.ida_matrix(ip)
        w = AG.ida_apply_image(Mx, img, ip.out_hw)
        exp = Mx @ [u, v, 1]
        if not (8 <= exp[0] < 504 and 8 <= exp[1] < 376):
            continue
        vv, uu = np.nonzero(w > 0.5)
        assert abs(uu.mean() - exp[0]) <= 0.75 and abs(vv.mean() - exp[1]) <= 0.75
        checked += 1
    assert checked >= 5


def test_bda_boxes_match_corner_transform():
    b = rand_boxes(20)
    for p in [AG.BDAParams(17, True, False, 1.03), AG.BDAParams(-9, False, True, 0.97), AG.BDAParams(0, True, True, 1)]:
        A = AG.bda_matrix(p)
        b2 = AG.bda_apply_boxes(A, b)
        for i in range(len(b)):
            c = np.c_[bev_corners(b[i]), np.zeros(4)]
            c2 = AG.bda_apply_points(A, c)[:, :2]
            # same footprint (as a set of corners): IoU 1 between the boxes
            ref = np.r_[c2.mean(0), 0, b2[i, 3:6], b2[i, 6]]
            assert bev_iou_matrix(b2[i:i + 1], ref[None])[0, 0] == pytest.approx(1.0)
            assert set(np.round(c2, 6).ravel()) == set(np.round(bev_corners(b2[i]), 6).ravel())
    # explicit flip identities
    b1 = np.array([[5, 2, 0, 4, 2, 1.5, 0.3]])
    assert AG.bda_apply_boxes(AG.bda_matrix(AG.BDAParams(flip_y=True)), b1)[0, 6] == pytest.approx(-0.3)
    assert AG.bda_apply_boxes(AG.bda_matrix(AG.BDAParams(flip_x=True)), b1)[0, 6] == pytest.approx(np.pi - 0.3)


def test_downsample_depth_min():
    d = np.zeros((8, 8)); d[0, 0] = 10; d[1, 1] = 5; d[7, 7] = 3
    o = AG.downsample_depth_min(d, 4)
    assert o[0, 0] == 5 and o[1, 1] == 3 and o[0, 1] == 0


# ---------------------------------------------------------------- visibility
def test_points_in_boxes():
    b = np.array([[10, 0, 0.8, 4, 2, 1.6, np.pi / 4]])
    inside = np.array([[10, 0, 0.8], [10.5, 0.5, 1.0]])
    outside = np.array([[10, 3, 0.8], [20, 0, 0.8]])
    assert VIS.points_in_boxes_count(np.r_[inside, outside], b)[0] == 2


def test_vis_map_lookup():
    spec = VIS.VisMapSpec()
    vm = np.zeros((256, 256), np.uint8)
    vm[:128, :] = 255                       # everything in front of ego visible (+x is up)
    b = np.array([[10, 0, 0, 4, 2, 1.5, 0], [-10, 0, 0, 4, 2, 1.5, 0], [200, 0, 0, 4, 2, 1.5, 0]])
    np.testing.assert_array_equal(VIS.visible_from_map(b, vm, spec), [True, False, False])


def test_frustum():
    K = np.array([[400.0, 0, 400], [0, 400, 300], [0, 0, 1]])
    cams = [(K, np.linalg.inv(_cam(0)))]
    b = np.array([[15, 0, 0.8, 4, 2, 1.5, 0], [-15, 0, 0.8, 4, 2, 1.5, 0]])
    np.testing.assert_array_equal(VIS.in_any_camera_frustum(b, cams, (600, 800)), [True, False])


# ---------------------------------------------------------------- BN tools
class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, 8, 3, padding=1); self.bn = nn.BatchNorm2d(8)
        self.h = nn.Sequential(nn.Conv2d(8, 8, 1), nn.BatchNorm2d(8))

    def forward(self, x):
        return self.h(torch.relu(self.bn(self.c(x))))


def test_bn_batch_stats_context_does_not_touch_running_stats():
    m = Tiny().eval()
    before = m.bn.running_mean.clone()
    with NT.bn_use_batch_stats(m):
        m(torch.randn(2, 3, 8, 8) * 5 + 3)
    assert torch.equal(before, m.bn.running_mean) and not m.bn.training


def test_recalibrate_and_gn():
    torch.manual_seed(0)
    m = Tiny()
    data = [torch.randn(1, 3, 8, 8) * 2 + 1 for _ in range(50)]
    NT.recalibrate_bn(m, data, lambda mm, b: mm(b), n_batches=50)
    assert not m.training and m.bn.num_batches_tracked.item() == 50
    g = NT.bn_to_gn(Tiny(), 4)
    assert NT.count_bn(g) == 0 and isinstance(g.bn, nn.GroupNorm)
    f = NT.freeze_bn(Tiny()); f.train()
    assert f.training and not f.bn.training


# ---------------------------------------------------------------- depth
def test_project_lidar_keeps_nearest():
    K = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
    pts = np.array([[0, 0, 10.0], [0, 0, 5.0], [1, 1, 10.0]])
    uv, d = DA.project_lidar(pts, np.eye(4), K, (100, 100))
    assert len(d) == 2 and 5.0 in d and 10.0 in d


def test_depth_metrics_detect_scale():
    gt = np.random.default_rng(4).uniform(2, 50, 1000)
    r = DA.depth_metrics(gt / 1.25, gt)
    assert r["median_scale_gt_over_pred"] == pytest.approx(1.25)
    assert r["abs_rel_after_median_scaling"] < 1e-9


def test_bins():
    e = DA.depth_bin_edges(1, 50, 64, "LID")
    assert e[0] == 1 and e[-1] == pytest.approx(50) and np.all(np.diff(e) > 0)
    assert np.all(np.diff(np.diff(e)) > 0)
    assert DA.depth_to_bin([0.5, 1.0, 49.99, 50], DA.depth_bin_edges(1, 50, 64)).tolist() == [-1, 0, 63, -1]
    rep = DA.bin_budget_report(np.array([10, 60.0]), DA.depth_bin_edges(1, 50, 64))
    assert rep["bins_beyond_useful_range"] > 20 and rep["frac_lidar_beyond_dmax"] == 0.5


def _probs_with_noise(gt, edges, sig):
    c = 0.5 * (edges[:-1] + edges[1:])
    mean = gt + np.random.default_rng(5).normal(0, sig)
    logits = -((c[:, None] - mean[None]) ** 2) / (2 * sig[None] ** 2)
    p = np.exp(logits - logits.max(0)); return p / p.sum(0)


def test_calibration_informative_vs_not():
    edges = DA.depth_bin_edges(1, 50, 64)
    rng = np.random.default_rng(6)
    gt = rng.uniform(3, 45, 4000)
    sig = rng.uniform(0.3, 6, 4000)                      # wider dist <-> larger error
    good = DA.depth_uncertainty_calibration(_probs_with_noise(gt, edges, sig).reshape(64, 40, 100),
                                            gt.reshape(40, 100), edges)
    assert good["spearman_entropy_vs_abs_err"] > 0.3
    assert good["auroc_entropy_detects_err>2.0m"] > 0.7
    # entropy unrelated to error: shuffle sharpness against the noise used
    p = _probs_with_noise(gt, edges, sig)
    gt_bad = gt + rng.normal(0, rng.permutation(sig))
    bad = DA.depth_uncertainty_calibration(p.reshape(64, 40, 100), gt_bad.reshape(40, 100), edges)
    assert bad["spearman_entropy_vs_abs_err"] < good["spearman_entropy_vs_abs_err"]


# ---------------------------------------------------------------- end to end
def test_diagnose_cli(tmp_path):
    from coper_diag.diagnose import main
    rng = np.random.default_rng(8)
    good = make_frames(nf=60, rng=rng)
    bad = make_frames(nf=60, keep=0.3, noise=0.8, rng=rng)
    for f in bad:
        f["pred_boxes"][:, 6] += np.pi * (rng.random(len(f["pred_boxes"])) < 0.25)
        f["gt_visible"] = rng.random(len(f["gt_boxes"])) > 0.3
    paths = {}
    for n, fr in [("train_eval", good), ("val_eval", bad)]:
        p = tmp_path / f"{n}.pkl"; pickle.dump(fr, open(p, "wb")); paths[n] = p
    main(["--dump", f"train_eval={paths['train_eval']}", "--dump", f"val_eval={paths['val_eval']}",
          "--out", str(tmp_path / "rep")])
    md = (tmp_path / "rep.md").read_text()
    assert "generalisation gap" in md and "visible-only" in md


def test_torch_bda_geometry_matches_numpy():
    from coper_diag.lss_hooks import bda_geometry, ida_to_post
    rng = np.random.default_rng(9)
    A = AG.bda_matrix(AG.sample_bda(rng))
    g = rng.normal(size=(1, 2, 3, 4, 5, 3))
    out = bda_geometry(torch.tensor(g, dtype=torch.float64), torch.tensor(A[None]))
    np.testing.assert_allclose(out.numpy(), AG.bda_apply_points(A, g), atol=1e-10)
    # LSS undo-IDA convention: inv(post_rots) @ (uv' - post_trans) == uv
    Mx = AG.ida_matrix(AG.sample_ida(rng))
    pr, pt = ida_to_post(Mx)
    uv = np.array([123.0, 77.0, 1.0]); uvp = Mx @ uv
    back = torch.linalg.inv(pr.double()) @ (torch.tensor([uvp[0], uvp[1], 0.0]).double() - pt.double())
    np.testing.assert_allclose(back[:2].numpy(), uv[:2], atol=1e-4)
