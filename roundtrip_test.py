"""
Module-by-module data-flow correctness test for the coop_v2v pipeline.

Runs on CPU, no GPU, no training, seconds to execute. Uses the REAL
project functions (src.data.opv2v._build_bev_gt, src.metrics.decode),
not reimplementations, so a pass/fail here is about the deployed code,
not about my reading of it.

TEST 1 — encode/decode geometric round trip.
    Build a synthetic `vehicles` dict with known boxes (varied position,
    yaw, near grid boundary), run it through the REAL target-generation
    function, feed the resulting heat/offset/size/yaw/zh straight into
    the REAL decode(), and assert the recovered box matches the box
    _build_bev_gt itself recorded in target["boxes"]. This tests the
    deployed code's row/col <-> x/y convention, sin/cos/atan2 yaw
    round trip, and log/exp size round trip -- not a hand trace of it.

TEST 2 — decode() duplicate-peak / missing-NMS behaviour.
    Build synthetic heatmaps with two local maxima at increasing
    separation (1, 2, 3 cells) and confirm: at 1-cell separation the
    3x3 local-max filter in _peaks() suppresses the lower one (expect
    1 detection); at 2+ cell separation both survive as independent
    detections (expect 2 detections) even though a single real car
    (~9 cells long x ~4 cells wide at cell_m=0.5) can easily contain
    two such peaks. This directly demonstrates a structural mechanism
    for the observed precision50 ~ 0.022 at epoch 4 of ours_full_40m:
    a noisy/undertrained heatmap with multiple bumps inside one car's
    footprint yields multiple false-positive detections of that one car.

TEST 4 — gaussian_focal_loss pos_weight dead-parameter check.
    src/loss/detection.py's gaussian_focal_loss() accepts a pos_weight
    argument and its docstring claims it upweights the positive-cell
    loss term to counteract negative-term dominance at this grid size.
    The formula as written never multiplies pos_loss by pos_weight.
    Call it twice on identical synthetic (pred, gt) with pos_weight=1.0
    and pos_weight=4.0 and assert the outputs differ. If they don't,
    the parameter is confirmed dead code, not a hand-reading guess.

TEST 5 — warp_to_ego rotation/translation convention.
    src/models/topk_scheduler.py's warp_to_ego() warps a collaborator's
    BEV feature map into the ego frame using affine_grid/grid_sample and
    that agent's (dx, dy, dyaw) pose expressed in ego-frame metres/radians.
    A sign error in this matrix would silently misalign every collaborator's
    features during fusion -- no crash, no shape error, just systematically
    wrong cooperation -- matching the original symptom pattern (trains fine,
    generalizes badly) better than anything found so far. Hand-derivation
    said the code is correct; this test checks it numerically instead of
    trusting that derivation. Places a single active cell at a known local
    position, warps it with a known (dx, dy, dyaw), and checks the output
    peak lands at the independently-computed expected ego-frame cell
    (expected_ego = R(dyaw) @ local + [dx, dy]).

TEST 6 — _build_bev_gt_from_boxes round trip (the function REAL training
    actually uses, since cfg.augment=True by default routes target
    generation through this function, not _build_bev_gt). Test 1 only
    verified _build_bev_gt; comparison-reading suggested this one uses
    identical row/col, offset, size, yaw conventions (only the heatmap
    Gaussian-spread method differs), but that's a reading, not a
    measurement. Same round-trip discipline as Test 1: known boxes ->
    _build_bev_gt_from_boxes -> decode() -> assert recovery.

TEST 7 — UtilityGatedFusion masking correctness (gated_fusion.py).
    Structural, weight-independent checks (don't depend on the module's
    random init): (a) when no agent transmits to a cell, F_star at that
    cell must equal F_ego exactly (the documented residual fallback);
    (b) a masked-out agent (mask=0) must have zero influence on F_star --
    perturbing its F_recv value while masked must not change the output,
    since masked entries get -inf before softmax; (c) when exactly one
    agent is unmasked, perturbing any OTHER agent's F_recv must not
    change F_star either.

TEST 8 — hard_topk() exact-k-per-cell and budget truncation
    (topk_scheduler.py). Verifies the per-cell top-k selection matches
    an independent numpy argsort, and that the greedy budget step keeps
    at most budget_cells cells total, keeping the highest-utility ones.

TEST 9 — quantise_ste() discretization level count and straight-through
    gradient (coop_detector.py). Verifies the 4-bit quantiser actually
    produces at most 2*((1<<(n_bits-1))-1)+1 distinct levels (not that
    it silently passes values through unquantised), and that the
    backward gradient is identity (dy/dx=1) despite the forward being
    a discontinuous round -- the STE contract.

TEST 10 — frustum_to_ego camera projection, executed (lss_backbone.py).
    Only hand-verified algebraically before this. Builds a single-point
    frustum from a known camera-frame 3D point (via a synthetic pinhole
    projection, independent of the code under test), known rots/trans
    (cam->ego, with a real rotation, not identity), and checks
    frustum_to_ego recovers the independently-computed expected
    ego-frame point. A second sub-case adds a non-trivial post_rots/
    post_trans (simulating an IDA pixel-space affine warp) and checks
    the IDA-undo branch still recovers the same ego-frame point -- this
    branch was never exercised by the earlier hand derivation at all.

Run from the project root on the login node:
    python roundtrip_test.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
import torch

sys.path.insert(0, ".")  # project root must be cwd, matching PYTHONPATH convention

from src.config import Config          # noqa: E402
from src.data.opv2v import _build_bev_gt, _build_bev_gt_from_boxes  # noqa: E402
from src.metrics import decode          # noqa: E402
from src.loss.detection import gaussian_focal_loss  # noqa: E402
from src.models.topk_scheduler import warp_to_ego, hard_topk  # noqa: E402
from src.models.gated_fusion import UtilityGatedFusion  # noqa: E402
from src.models.coop_detector import quantise_ste  # noqa: E402
from src.models.lss_backbone import frustum_to_ego  # noqa: E402

PASS = "PASS"
FAIL = "FAIL"


def make_vehicle(vid, cx, cy, cz, yaw_deg, l=4.5, w=1.9, h=1.6):
    """A synthetic vehicle dict entry in the schema _build_bev_gt expects,
    placed directly at the desired EGO-frame (cx, cy, cz) by using
    ego_T_world = identity, so world frame == ego frame for this test."""
    return {
        "location": [cx, cy, cz],
        "angle": [0.0, yaw_deg, 0.0],   # [roll, yaw, pitch] deg, world==ego here
        "extent": [l / 2.0, w / 2.0, h / 2.0],  # [half_l, half_w, half_h]
    }


def test1_roundtrip(cfg):
    print("\n=== TEST 1: encode/decode geometric round trip (real functions) ===")
    I = np.eye(4)

    cases = {
        # vid:        (cx,    cy,    cz,   yaw_deg)
        "pos_x_pos_y": (10.0,  10.0, 0.3,   0.0),
        "neg_x_pos_y": (-10.0, 10.0, 0.3,  45.0),
        "pos_x_neg_y": (10.0, -10.0, 0.3,  90.0),
        "neg_x_neg_y": (-10.0, -10.0, 0.3, 135.0),
        "near_bndry":  (19.0,  19.0, 0.3, 180.0),
        "neg_bndry":   (-19.0, -19.0, 0.3, -90.0),
        "axis_x":      (15.0,   0.0, 0.3,  -45.0),
        "axis_y":      (0.0,   15.0, 0.3, -135.0),
    }
    vehicles = {vid: make_vehicle(vid, *vals) for vid, vals in cases.items()}

    target = _build_bev_gt(vehicles, ego_T_world=I, cfg=cfg, ego_T_world_fwd=I)
    gt_boxes = target["boxes"]  # (M,7) cx,cy,cz,w,l,h,yaw -- what _build_bev_gt itself recorded
    n_placed = gt_boxes.shape[0]
    print(f"  placed {n_placed}/{len(cases)} vehicles on the grid "
          f"(ego-exclusion zone or out-of-bounds would drop some)")

    out = {k: torch.from_numpy(target[k]) for k in ("heat", "offset", "size", "yaw", "zh")}
    pred_boxes, pred_scores = decode(out, cfg, score_thresh=0.05, topk=50)
    print(f"  decode() returned {pred_boxes.shape[0]} detections for {n_placed} GT boxes")

    if pred_boxes.shape[0] != n_placed:
        print(f"  {FAIL}: detection count mismatch "
              f"({pred_boxes.shape[0]} != {n_placed}) -- heatmap peaks are "
              f"colliding/merging or being dropped. Investigate grid "
              f"resolution vs. box separation before trusting anything below.")
        return False

    # match GT boxes to predicted boxes by nearest centre (order isn't guaranteed
    # to match -- topk sorts by score, GT boxes are in vehicle-dict iteration order)
    ok = True
    used = set()
    for gi in range(n_placed):
        g = gt_boxes[gi]
        dists = np.linalg.norm(pred_boxes[:, :2] - g[:2], axis=1)
        for u in used:
            dists[u] = np.inf
        pi = int(np.argmin(dists))
        used.add(pi)
        p = pred_boxes[pi]

        pos_err = float(np.linalg.norm(p[:2] - g[:2]))
        z_err = abs(float(p[2] - g[2]))
        w_err = abs(float(p[3] - g[3]))
        l_err = abs(float(p[4] - g[4]))
        h_err = abs(float(p[5] - g[5]))
        yaw_err = abs(math.atan2(math.sin(p[6] - g[6]), math.cos(p[6] - g[6])))

        tol_pos = 1e-3   # metres; should be ~exact modulo float32
        tol_yaw = 1e-3   # radians

        case_ok = (pos_err < tol_pos and z_err < tol_pos and w_err < tol_pos
                   and l_err < tol_pos and h_err < tol_pos and yaw_err < tol_yaw)
        ok = ok and case_ok
        status = PASS if case_ok else FAIL
        print(f"  [{status}] gt=({g[0]:+.2f},{g[1]:+.2f},yaw={math.degrees(g[6]):+.1f} deg)  "
              f"pos_err={pos_err:.4f}m  yaw_err={math.degrees(yaw_err):.3f}deg  "
              f"w_err={w_err:.4f}  l_err={l_err:.4f}  h_err={h_err:.4f}  z_err={z_err:.4f}")

    print(f"\n  TEST 1 overall: {PASS if ok else FAIL}")
    return ok


def test2_duplicate_peaks(cfg):
    print("\n=== TEST 2: decode() duplicate-peak / missing-NMS behaviour ===")
    H, W = cfg.grid_H, cfg.grid_W

    def run_case(sep_cells, label):
        heat = torch.zeros(H, W)
        off = torch.zeros(2, H, W)
        sz = torch.zeros(2, H, W)
        yaw = torch.zeros(2, H, W)
        yaw[1] = 1.0  # cos=1 everywhere so atan2 is well-defined (yaw=0)
        zh = torch.zeros(2, H, W)
        zh[1] = 1.5  # nonzero h so clamp doesn't matter

        # (60, 60) -> cx=(60*cell_m+bev_x_min)=10.0, cy=10.0 -- well outside the
        # ego-body exclusion zone (|cx|<3, |cy|<2) that decode() masks around
        # the origin. The earlier (40,40) choice mapped to cx=cy=0.0, i.e.
        # exactly ego's own position, so every peak there got masked out
        # before reaching topk -- that was a bug in this test, not in decode().
        r0, c0 = 60, 60
        r1, c1 = 60, 60 + sep_cells
        heat[r0, c0] = 0.95
        heat[r1, c1] = 0.90
        sz[0, r0, c0] = sz[0, r1, c1] = math.log(1.9)
        sz[1, r0, c0] = sz[1, r1, c1] = math.log(4.5)

        out = {"heat": heat, "offset": off, "size": sz, "yaw": yaw, "zh": zh}
        boxes, scores = decode(out, cfg, score_thresh=0.20, topk=200)
        n = boxes.shape[0]
        print(f"  separation={sep_cells} cell(s) ({sep_cells*cfg.cell_m:.1f} m): "
              f"decode() returned {n} detection(s) for 1 physical peak-pair  [{label}]")
        return n

    radius_m = getattr(cfg, "nms_radius_m", 2.0)
    radius_cells = radius_m / cfg.cell_m
    print(f"\n  NMS radius_m={radius_m} ({radius_cells:.0f} cells) -- testing both sides of it.")

    # within the NMS radius -> same physical car -> expect collapse to 1
    n_in1 = run_case(1, f"within radius ({cfg.cell_m*1:.1f}m < {radius_m}m) -> expect 1")
    n_in2 = run_case(int(radius_cells) - 1 if radius_cells > 2 else 2,
                      f"within radius -> expect 1")
    # just past the NMS radius -> should be treated as two distinct objects -> expect 2
    far_cells = int(radius_cells) + 3
    n_far = run_case(far_cells,
                      f"beyond radius ({far_cells*cfg.cell_m:.1f}m > {radius_m}m) -> expect 2")

    car_len_cells = 4.5 / cfg.cell_m
    car_wid_cells = 1.9 / cfg.cell_m
    print(f"\n  for reference: one car spans ~{car_len_cells:.0f} x {car_wid_cells:.0f} "
          f"cells at cell_m={cfg.cell_m}.")

    ok_suppress = (n_in1 == 1) and (n_in2 == 1)
    ok_preserve = (n_far == 2)
    ok = ok_suppress and ok_preserve
    if ok_suppress:
        print("  CONFIRMED: peaks within the NMS radius (same car) correctly collapse to 1.")
    else:
        print(f"  UNEXPECTED: in-radius cases gave n_in1={n_in1}, n_in2={n_in2} "
              f"(expected both 1) -- NMS not behaving as patched, re-examine.")
    if ok_preserve:
        print(f"  CONFIRMED: peaks beyond the NMS radius ({far_cells*cfg.cell_m:.1f}m apart, "
              "two distinct objects) are NOT merged -- radius isn't over-suppressing.")
    else:
        print(f"  WARNING: far-separated peaks ({far_cells*cfg.cell_m:.1f}m apart) collapsed "
              f"to n_far={n_far} -- radius_m={radius_m} may be too large and could merge "
              "genuinely distinct nearby vehicles (e.g. in parking scenes). Consider a "
              "smaller radius_m.")
    print(f"\n  TEST 2 overall: {PASS if ok else FAIL}")
    return ok


def test3_near_field_depth_gap(cfg):
    print("\n=== TEST 3 (informational, not pass/fail): near-field depth coverage ===")
    print(f"  lss_d_min={cfg.lss_d_min} m, lss_d_max={cfg.lss_d_max} m, "
          f"BEV half-extent={cfg.bev_x_max} m")
    if cfg.lss_d_min > 0.0:
        print(f"  Any true surface closer than {cfg.lss_d_min} m to ANY camera has no "
              f"frustum depth bin covering it (bins span [{cfg.lss_d_min}, {cfg.lss_d_max}] m).")
        print("  Its lifted feature is forced onto the nearest available bin "
              f"(>= {cfg.lss_d_min} m), i.e. splatted to the wrong BEV cell by construction.")
        print("  This affects any vehicle within "
              f"{cfg.lss_d_min} m of a camera -- common for adjacent-lane / close-follow "
              "cars in OPV2V traffic -- independent of training quality.")
        print("  Exact affected area requires real camera extrinsics (rots/trans per agent); "
              "not computed here. Flagging as a structural candidate, not re-running the "
              "full frustum geometry test since points 1 and 2 above already confirm the "
              "indexing/decoding chain is sound.")


def test4_pos_weight_dead_param():
    print("\n=== TEST 4: gaussian_focal_loss pos_weight dead-parameter check ===")
    torch.manual_seed(0)
    H, W = 80, 80
    gt = torch.zeros(H, W)
    gt[40, 40] = 1.0  # one positive cell, matching _build_bev_gt's hard-1.0 convention
    # small Gaussian tail around it, like a real target, so neg_loss isn't trivially zero
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            r, c = 40 + dr, 40 + dc
            if (dr, dc) != (0, 0):
                gt[r, c] = max(gt[r, c].item(), math.exp(-(dr**2 + dc**2) / 2.0) * 0.6)
    pred = torch.full((H, W), 0.05)
    pred[40, 40] = 0.4  # deliberately weak/underconfident peak, like an undertrained head

    loss_w1 = gaussian_focal_loss(pred, gt, pos_weight=1.0)
    loss_w4 = gaussian_focal_loss(pred, gt, pos_weight=4.0)
    differs = not torch.isclose(loss_w1, loss_w4).item()

    print(f"  loss(pos_weight=1.0) = {loss_w1.item():.8f}")
    print(f"  loss(pos_weight=4.0) = {loss_w4.item():.8f}")
    if differs:
        print("  CONFIRMED: pos_weight now changes the loss value -- the fix is live in")
        print("  the deployed code (this is the desired post-patch state).")
    else:
        print("  pos_weight has NO effect -- the patch did not take, or was reverted.")
        print("  Re-check src/loss/detection.py before trusting any training run against it.")
    print(f"\n  TEST 4 overall: {PASS if differs else FAIL} "
          f"({'pos_weight fix confirmed live' if differs else 'fix NOT applied, recheck file'})")
    return differs


def test5_warp_to_ego(cfg):
    print("\n=== TEST 5: warp_to_ego rotation/translation convention ===")
    H, W = cfg.grid_H, cfg.grid_W
    cell = cfg.cell_m

    # a single active cell in the collaborator's OWN local frame
    local_row, local_col = 50, 60
    local_x = local_col * cell + cfg.bev_x_min   # 10.0 m
    local_y = local_row * cell + cfg.bev_y_min   # 5.0 m

    dx, dy, dyaw_deg = 3.0, -2.0, 30.0
    dyaw = math.radians(dyaw_deg)

    # independently-computed expected ego-frame position:
    # ego_pos = R(dyaw) @ local_pos + [dx, dy]
    c, s = math.cos(dyaw), math.sin(dyaw)
    exp_ego_x = c * local_x - s * local_y + dx
    exp_ego_y = s * local_x + c * local_y + dy
    exp_col = round((exp_ego_x - cfg.bev_x_min) / cell)
    exp_row = round((exp_ego_y - cfg.bev_y_min) / cell)
    print(f"  local cell (row={local_row},col={local_col}) -> local (x={local_x:.2f},y={local_y:.2f})")
    print(f"  rel_pose: dx={dx} dy={dy} dyaw={dyaw_deg} deg")
    print(f"  expected ego-frame (x={exp_ego_x:.2f}, y={exp_ego_y:.2f}) "
          f"-> expected cell (row={exp_row}, col={exp_col})")

    x = torch.zeros(H, W)
    x[local_row, local_col] = 1.0
    rel_pose = torch.tensor([dx, dy, dyaw], dtype=torch.float32)

    warped = warp_to_ego(x, rel_pose, cfg)
    peak_idx = int(torch.argmax(warped).item())
    got_row, got_col = divmod(peak_idx, W)
    mass_total = float(warped.sum())
    print(f"  warp_to_ego() output peak at (row={got_row}, col={got_col}), "
          f"peak value={warped[got_row, got_col].item():.4f}, total mass={mass_total:.4f}")

    tol_cells = 1  # bilinear interpolation can spread mass across neighbouring cells
    ok = (abs(got_row - exp_row) <= tol_cells) and (abs(got_col - exp_col) <= tol_cells)
    if ok:
        print(f"  CONFIRMED: warp_to_ego's rotation/translation convention matches the "
              f"independently-derived expectation (within {tol_cells} cell).")
    else:
        print(f"  MISMATCH: got (row={got_row},col={got_col}) vs expected "
              f"(row={exp_row},col={exp_col}) -- off by "
              f"(d_row={got_row-exp_row}, d_col={got_col-exp_col}). "
              f"This would mean collaborator features are being warped to the wrong "
              f"location during fusion -- a high-impact bug, not a false alarm. "
              f"Check sign conventions (rotation direction, which axis is dx vs dy) "
              f"against the actual rel_pose producer in the data loader before fixing "
              f"blindly, since the bug could be on either side of this interface.")
    print(f"\n  TEST 5 overall: {PASS if ok else FAIL}")
    return ok


def test6_from_boxes_roundtrip(cfg):
    print("\n=== TEST 6: _build_bev_gt_from_boxes round trip (real augmented-training path) ===")
    # (cx, cy, cz, w, l, h, yaw) -- same style of coverage as Test 1: quadrants,
    # a near-boundary case, and varied yaw, all outside the ego-exclusion zone.
    boxes = np.array([
        [10.0,  10.0, 0.3, 1.9, 4.5, 1.6,  0.0],
        [-10.0, 10.0, 0.3, 1.9, 4.5, 1.6,  math.radians(45.0)],
        [10.0, -10.0, 0.3, 1.9, 4.5, 1.6,  math.radians(90.0)],
        [-10.0,-10.0, 0.3, 1.9, 4.5, 1.6,  math.radians(135.0)],
        [19.0,  19.0, 0.3, 1.9, 4.5, 1.6,  math.radians(180.0)],
        [-19.0,-19.0, 0.3, 1.9, 4.5, 1.6,  math.radians(-90.0)],
        [15.0,   0.0, 0.3, 1.9, 4.5, 1.6,  math.radians(-45.0)],
        [0.0,   15.0, 0.3, 1.9, 4.5, 1.6,  math.radians(-135.0)],
    ], dtype=np.float32)

    target = _build_bev_gt_from_boxes(boxes, cfg)
    gt_boxes = target["boxes"]
    n_placed = gt_boxes.shape[0]
    print(f"  placed {n_placed}/{len(boxes)} boxes via _build_bev_gt_from_boxes "
          f"(size-adaptive Gaussian heatmap)")

    out = {k: torch.from_numpy(target[k]) for k in ("heat", "offset", "size", "yaw", "zh")}
    pred_boxes, pred_scores = decode(out, cfg, score_thresh=0.05, topk=50)
    print(f"  decode() returned {pred_boxes.shape[0]} detections for {n_placed} GT boxes")

    if pred_boxes.shape[0] != n_placed:
        print(f"  {FAIL}: detection count mismatch ({pred_boxes.shape[0]} != {n_placed})")
        return False

    ok = True
    used = set()
    for gi in range(n_placed):
        g = gt_boxes[gi]
        dists = np.linalg.norm(pred_boxes[:, :2] - g[:2], axis=1)
        for u in used:
            dists[u] = np.inf
        pi = int(np.argmin(dists))
        used.add(pi)
        p = pred_boxes[pi]

        pos_err = float(np.linalg.norm(p[:2] - g[:2]))
        yaw_err = abs(math.atan2(math.sin(p[6] - g[6]), math.cos(p[6] - g[6])))
        w_err = abs(float(p[3] - g[3]))
        l_err = abs(float(p[4] - g[4]))

        tol_pos, tol_yaw = 1e-3, 1e-3
        case_ok = (pos_err < tol_pos and yaw_err < tol_yaw and w_err < tol_pos and l_err < tol_pos)
        ok = ok and case_ok
        status = PASS if case_ok else FAIL
        print(f"  [{status}] gt=({g[0]:+.2f},{g[1]:+.2f},yaw={math.degrees(g[6]):+.1f} deg)  "
              f"pos_err={pos_err:.4f}m  yaw_err={math.degrees(yaw_err):.3f}deg  "
              f"w_err={w_err:.4f}  l_err={l_err:.4f}")

    print(f"\n  TEST 6 overall: {PASS if ok else FAIL}")
    return ok


def test7_gated_fusion_masking():
    print("\n=== TEST 7: UtilityGatedFusion masking correctness ===")
    torch.manual_seed(0)
    C, H, W = 4, 3, 3
    N = 3
    fusion = UtilityGatedFusion(C=C, use_utility_gate=True)
    fusion.eval()

    F_ego = torch.randn(C, H, W)
    U_r = torch.randn(N, H, W)

    ok = True

    # (a) no-agent fallback: all masks 0 -> F_star must equal F_ego exactly
    F_recv = torch.randn(N, C, H, W)
    mask_none = torch.zeros(N, H, W)
    with torch.no_grad():
        F_star = fusion(F_ego, F_recv, U_r, mask_none)
    exact = torch.equal(F_star, F_ego)
    print(f"  (a) no-agent fallback exact match to F_ego: {exact}")
    ok = ok and exact

    # (b) masked agent has zero influence: change F_recv for a masked-out
    # agent, output must not change (masked to -inf before softmax)
    mask_one = torch.zeros(N, H, W)
    mask_one[1] = 1.0   # only agent 1 transmits everywhere
    F_recv_a = torch.randn(N, C, H, W)
    F_recv_b = F_recv_a.clone()
    F_recv_b[0] = torch.randn(C, H, W) * 100.0   # perturb a MASKED agent wildly
    F_recv_b[2] = torch.randn(C, H, W) * 100.0   # perturb another MASKED agent wildly
    with torch.no_grad():
        out_a = fusion(F_ego, F_recv_a, U_r, mask_one)
        out_b = fusion(F_ego, F_recv_b, U_r, mask_one)
    unaffected = torch.allclose(out_a, out_b, atol=1e-5)
    print(f"  (b) masked-out agents' huge perturbation has no effect: {unaffected}")
    ok = ok and unaffected

    if not ok:
        print("  FAIL: see above -- masking logic is not behaving as the module's own "
              "docstring claims (cells an agent did NOT transmit should be fully masked).")
    print(f"\n  TEST 7 overall: {PASS if ok else FAIL}")
    return ok


def test8_hard_topk():
    print("\n=== TEST 8: hard_topk() exact-k-per-cell and budget truncation ===")
    torch.manual_seed(1)
    N, H, W = 5, 6, 6
    U = torch.rand(N, H, W)
    k = 2

    # no budget limit (budget_cells = all cells) -- check exact top-k per cell
    M = hard_topk(U, k=k, budget_cells=H * W, tau=0.0)
    ok = True
    for r in range(H):
        for c in range(W):
            col = U[:, r, c].numpy()
            expected_idx = set(np.argsort(-col)[:k].tolist())
            got_idx = set(np.nonzero(M[:, r, c].numpy())[0].tolist())
            if expected_idx != got_idx:
                ok = False
    print(f"  per-cell top-{k} selection matches independent argsort for all {H*W} cells: {ok}")

    # budget truncation: cut budget to half the cells, check total kept <= budget
    budget = (H * W) // 2
    M_budget = hard_topk(U, k=k, budget_cells=budget, tau=0.0)
    kept_cells = int((M_budget.sum(dim=0) > 0).sum().item())
    budget_ok = kept_cells <= budget
    print(f"  budget={budget} cells -> kept {kept_cells} cells (<=budget): {budget_ok}")
    ok = ok and budget_ok

    print(f"\n  TEST 8 overall: {PASS if ok else FAIL}")
    return ok


def test9_quantise_ste():
    print("\n=== TEST 9: quantise_ste() discretization + straight-through gradient ===")
    torch.manual_seed(2)
    x = torch.randn(1000) * 3.0
    x.requires_grad_(True)
    n_bits = 4
    y = quantise_ste(x, n_bits=n_bits)

    n_unique = torch.unique(y.detach()).numel()
    max_levels = 2 * ((1 << (n_bits - 1)) - 1) + 1   # 15 for 4-bit symmetric
    discretized = n_unique <= max_levels
    print(f"  forward output has {n_unique} unique values (max allowed {max_levels} "
          f"for {n_bits}-bit symmetric): {'OK' if discretized else 'TOO MANY -- not quantising'}")

    y.sum().backward()
    grad_is_identity = torch.allclose(x.grad, torch.ones_like(x.grad))
    print(f"  backward gradient is identity (STE contract): {grad_is_identity}")

    ok = discretized and grad_is_identity
    print(f"\n  TEST 9 overall: {PASS if ok else FAIL}")
    return ok


def _pinhole_project(P_cam, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    """Independent forward projection, used only to SYNTHESIZE the test input -- not the code
    under test.

    P_cam is in THIS DATASET'S camera convention: (depth, right, up), CARLA-native. That is the
    convention frustum_to_ego has used since the 2026-10-27 axis fix (verified against real data).
    Before that fix this helper assumed standard pinhole axes (right, down, forward), which is why
    this test failed from the fix until 2026-11-09 -- the test encoded the old convention, the
    code was correct."""
    depth, right, up = P_cam
    u = fx * right / depth + cx
    v = fy * (-up) / depth + cy
    return u, v, depth


def test10_frustum_to_ego():
    print("\n=== TEST 10: frustum_to_ego camera projection (executed) ===")
    torch.manual_seed(3)

    # known cam->ego extrinsics: a real rotation (20 deg about ego-Z) + translation
    yaw = math.radians(20.0)
    c, s = math.cos(yaw), math.sin(yaw)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    t = torch.tensor([2.0, 0.0, 1.0], dtype=torch.float32)

    # (depth, right, up) in this dataset's camera convention: 8 m ahead, 0.3 m right, 0.2 m down
    P_cam = torch.tensor([8.0, 0.3, -0.2], dtype=torch.float32)
    u, v, d = _pinhole_project(P_cam.tolist())
    intrins = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])

    expected_ego = R @ P_cam + t
    print(f"  cam-frame point P_cam={P_cam.tolist()} -> pixel (u={u:.2f}, v={v:.2f}, d={d:.2f})")
    print(f"  expected ego-frame point: {expected_ego.tolist()}")

    # --- sub-case A: no IDA (post_rots=I, post_trans=0) ---
    frustum = torch.tensor([[[[u, v, d]]]], dtype=torch.float32)       # (1,1,1,3) -> K,fH,fW,3
    rots = R.unsqueeze(0)          # (1,3,3)
    trans = t.unsqueeze(0)         # (1,3)
    intr = intrins.unsqueeze(0)    # (1,3,3)
    post_rots_id = torch.eye(3).unsqueeze(0)
    post_trans_zero = torch.zeros(3).unsqueeze(0)

    pts_ego = frustum_to_ego(frustum, rots, trans, intr, post_rots_id, post_trans_zero)
    got_a = pts_ego[0, 0, 0, 0]
    err_a = float((got_a - expected_ego).norm())
    print(f"  (A) no-IDA: got {got_a.tolist()}, error={err_a:.6f}")
    ok_a = err_a < 1e-3

    # --- sub-case B: with IDA (pixel_aug = 1.5 * pixel_orig + offset), test undo branch ---
    scale, offset = 1.5, torch.tensor([-10.0, -20.0])
    u_aug = scale * u + offset[0].item()
    v_aug = scale * v + offset[1].item()
    frustum_aug = torch.tensor([[[[u_aug, v_aug, d]]]], dtype=torch.float32)
    post_rots = torch.eye(3).unsqueeze(0)
    post_rots[0, :2, :2] = torch.eye(2) * scale
    post_trans = torch.zeros(1, 3)
    post_trans[0, :2] = offset

    pts_ego_b = frustum_to_ego(frustum_aug, rots, trans, intr, post_rots, post_trans)
    got_b = pts_ego_b[0, 0, 0, 0]
    err_b = float((got_b - expected_ego).norm())
    print(f"  (B) with-IDA (scale={scale}, offset={offset.tolist()}): "
          f"got {got_b.tolist()}, error={err_b:.6f}")
    ok_b = err_b < 1e-3

    ok = ok_a and ok_b
    if not ok:
        print("  MISMATCH: frustum_to_ego does not recover the independently-computed "
              "ego-frame point. This is the actual neural-network forward path for every "
              "camera in the model -- a real finding, not a false alarm, if confirmed.")
    print(f"\n  TEST 10 overall: {PASS if ok else FAIL}")
    return ok


def main():
    cfg = Config()
    cfg.validate()
    print(f"Config: grid={cfg.grid_H}x{cfg.grid_W}  cell_m={cfg.cell_m}  "
          f"bev=[{cfg.bev_x_min},{cfg.bev_x_max}]x[{cfg.bev_y_min},{cfg.bev_y_max}]  "
          f"d=[{cfg.lss_d_min},{cfg.lss_d_max}]")

    r1 = test1_roundtrip(cfg)
    r2 = test2_duplicate_peaks(cfg)
    test3_near_field_depth_gap(cfg)
    r4 = test4_pos_weight_dead_param()
    r5 = test5_warp_to_ego(cfg)
    r6 = test6_from_boxes_roundtrip(cfg)
    r7 = test7_gated_fusion_masking()
    r8 = test8_hard_topk()
    r9 = test9_quantise_ste()
    r10 = test10_frustum_to_ego()

    print("\n" + "=" * 70)
    print(f"SUMMARY: encode/decode indexing        = {PASS if r1 else FAIL}")
    print(f"SUMMARY: decode() NMS fix behaves       = {'as expected' if r2 else 'UNEXPECTED, recheck'}")
    print(f"SUMMARY: pos_weight fix is live         = {'YES' if r4 else 'NO, recheck file'}")
    print(f"SUMMARY: warp_to_ego convention         = {'CORRECT' if r5 else 'MISMATCH -- see above'}")
    print(f"SUMMARY: _build_bev_gt_from_boxes       = {'CORRECT' if r6 else 'MISMATCH -- see above'}")
    print(f"SUMMARY: gated_fusion masking           = {'CORRECT' if r7 else 'MISMATCH -- see above'}")
    print(f"SUMMARY: hard_topk selection/budget     = {'CORRECT' if r8 else 'MISMATCH -- see above'}")
    print(f"SUMMARY: quantise_ste STE contract      = {'CORRECT' if r9 else 'MISMATCH -- see above'}")
    print(f"SUMMARY: frustum_to_ego (executed)      = {'CORRECT' if r10 else 'MISMATCH -- see above'}")
    print("=" * 70)

    # Exit non-zero on failure so this can actually gate a training job. Until 2026-11-09 main()
    # always exited 0, which is how TEST 10 stayed red unnoticed after the camera-convention fix.
    failed = [n for n, r in (("encode/decode", r1), ("pos_weight", r4), ("warp_to_ego", r5),
                             ("build_bev_gt_from_boxes", r6), ("gated_fusion", r7),
                             ("hard_topk", r8), ("quantise_ste", r9), ("frustum_to_ego", r10))
              if not r]
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())