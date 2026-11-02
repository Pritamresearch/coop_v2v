"""
Qualitative visualization: render predicted vs. ground-truth 3D boxes.

Two views:
    1. BEV (bird's-eye view): oriented rectangles on the 200x200 grid,
       color-coded TP (matched) / FP (predicted, unmatched) / FN (missed GT).
       This is the standard qualitative figure in cooperative-perception papers.
    2. Camera view: project the SAME boxes into the ego's camera images so a
       reader can sanity-check that a BEV detection corresponds to something
       real in the pixel data.

Usage (after a model is trained):

    from plots.visualize_boxes import render_bev, render_camera_overlay
    render_bev(out, scene, cfg, save_path="plots/figures/qual_bev_0.png")
    render_camera_overlay(scene, cfg, save_path="plots/figures/qual_cam_0.png")

Both functions are also runnable standalone via make_qualitative_figures()
which grabs a few scenes from a trained checkpoint and renders a grid.
"""
from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from src.metrics import decode, rotated_iou


# ------------------------------------------------------------------
# BEV rectangle geometry (shared with metrics.rotated_iou convention)
# ------------------------------------------------------------------
def _box_corners_xy(cx, cy, w, l, yaw):
    dx = np.array([ l/2,  l/2, -l/2, -l/2])
    dy = np.array([ w/2, -w/2, -w/2,  w/2])
    c, s = math.cos(yaw), math.sin(yaw)
    xs = cx + c * dx - s * dy
    ys = cy + s * dx + c * dy
    return np.stack([xs, ys], axis=-1)   # (4, 2)


def _match_for_viz(pred_boxes, pred_scores, gt_boxes, iou_t: float = 0.3):
    """
    Greedy match (same rule as Evaluator) so the qualitative figure's
    colors are consistent with the reported AP numbers.
    Returns:
        pred_status : (Np,) array of 'TP' / 'FP'
        gt_status   : (Ng,) array of 'TP' / 'FN'
    """
    Np, Ng = len(pred_boxes), len(gt_boxes)
    pred_status = np.array(["FP"] * Np, dtype=object)
    gt_status = np.array(["FN"] * Ng, dtype=object)
    matched_gt = np.zeros(Ng, dtype=bool)

    order = np.argsort(-pred_scores) if Np > 0 else np.array([], dtype=int)
    for k in order:
        d = pred_boxes[k]
        d_ = np.array([d[0], d[1], d[3], d[4], d[6]])
        best_i, best_iou = -1, 0.0
        for j in range(Ng):
            if matched_gt[j]:
                continue
            g = gt_boxes[j]
            g_ = np.array([g[0], g[1], g[3], g[4], g[6]])
            iou = rotated_iou(d_, g_)
            if iou > best_iou:
                best_iou, best_i = iou, j
        if best_i >= 0 and best_iou >= iou_t:
            matched_gt[best_i] = True
            pred_status[k] = "TP"
            gt_status[best_i] = "TP"
    return pred_status, gt_status


# ------------------------------------------------------------------
# BEV render
# ------------------------------------------------------------------
def render_bev(out: dict, scene: dict, cfg, save_path: str,
               score_thresh: float = 0.15, iou_match: float = 0.3,
               title: str | None = None):
    """
    out   : model forward-pass output dict (must contain 'heat','offset','size','yaw','zh')
    scene : the scene dict used to produce `out` (for gt boxes + oof flags)
    """
    pred_boxes, pred_scores = decode(out, cfg, score_thresh=score_thresh)
    gt_boxes = scene["gt"]["boxes"].cpu().numpy()
    gt_oof = scene["gt"]["oof"].cpu().numpy()

    pred_status, gt_status = _match_for_viz(pred_boxes, pred_scores, gt_boxes, iou_match)

    fig, ax = plt.subplots(figsize=(7, 7))

    # ego position + heading marker at origin
    ax.scatter([0], [0], marker="^", s=140, c="black", zorder=5, label="ego")

    # GT boxes: solid outline, red for FN (missed), green for TP (matched)
    # dashed for out-of-FoV (only visible to a collaborator)
    for j, g in enumerate(gt_boxes):
        cx, cy, cz, w, l, h, yaw = g
        corners = _box_corners_xy(cx, cy, w, l, yaw)
        color = "#2ca02c" if gt_status[j] == "TP" else "#d62728"
        style = "--" if gt_oof[j] else "-"
        poly = MplPolygon(corners, closed=True, fill=False,
                          edgecolor=color, linewidth=2.0, linestyle=style)
        ax.add_patch(poly)
        # heading tick
        tip = corners[0]  # front-right-ish corner
        ax.plot([cx, (corners[0, 0] + corners[1, 0]) / 2],
                [cy, (corners[0, 1] + corners[1, 1]) / 2], color=color, linewidth=1.5)

    # predicted boxes: blue outline for TP, orange for FP
    for k, d in enumerate(pred_boxes):
        cx, cy, cz, w, l, h, yaw = d
        corners = _box_corners_xy(cx, cy, w, l, yaw)
        color = "#1f77b4" if pred_status[k] == "TP" else "#ff7f0e"
        poly = MplPolygon(corners, closed=True, fill=False,
                          edgecolor=color, linewidth=1.4, linestyle=":")
        ax.add_patch(poly)

    R = cfg.bev_x_max
    ax.set_xlim(cfg.bev_x_min, cfg.bev_x_max)
    ax.set_ylim(cfg.bev_y_min, cfg.bev_y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, ego forward)")
    ax.set_ylabel("y (m, ego left)")
    ax.grid(alpha=0.25)

    # legend (manual proxy artists)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#2ca02c", lw=2, label="GT matched (TP)"),
        Line2D([0], [0], color="#d62728", lw=2, label="GT missed (FN)"),
        Line2D([0], [0], color="#d62728", lw=2, ls="--", label="GT out-of-FoV"),
        Line2D([0], [0], color="#1f77b4", lw=1.4, ls=":", label="Pred matched (TP)"),
        Line2D([0], [0], color="#ff7f0e", lw=1.4, ls=":", label="Pred unmatched (FP)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    n_tp = int((pred_status == "TP").sum())
    ax.set_title(title or f"BEV detections  (GT={n_gt}, Pred={n_pred}, TP={n_tp})")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------
# Camera-space overlay (project BEV boxes into the ego's own cameras)
# ------------------------------------------------------------------
def _project_box_to_cam(corners_xy_z, R_c2e, t_c2e, K, img_H, img_W):
    """
    corners_xy_z: (8, 3) box corners in ego frame (4 bottom + 4 top)
    Returns list of (u, v) or None if behind camera.
    """
    Rt = R_c2e.T
    pts_cam = (corners_xy_z - t_c2e[None, :]) @ Rt.T
    z = pts_cam[:, 0]
    uv = []
    for i in range(len(pts_cam)):
        if z[i] <= 0.3:
            uv.append(None)
            continue
        u = K[0, 0] * pts_cam[i, 1] / z[i] + K[0, 2]
        v = K[1, 1] * (-pts_cam[i, 2]) / z[i] + K[1, 2]
        uv.append((float(u), float(v)))
    return uv


def _box_corners_3d(cx, cy, cz, w, l, h, yaw):
    """8 corners: 4 bottom + 4 top, in ego frame (x fwd, y left, z up)."""
    xy = _box_corners_xy(cx, cy, w, l, yaw)          # (4,2)
    bottom = np.concatenate([xy, np.full((4, 1), cz - h / 2)], axis=-1)
    top = np.concatenate([xy, np.full((4, 1), cz + h / 2)], axis=-1)
    return np.concatenate([bottom, top], axis=0)      # (8,3)


_EDGES = [(0,1),(1,2),(2,3),(3,0),   # bottom
         (4,5),(5,6),(6,7),(7,4),   # top
         (0,4),(1,5),(2,6),(3,7)]  # verticals


def render_camera_overlay(scene: dict, cfg, boxes: np.ndarray, box_colors: list,
                          save_path: str, agent_idx: int = 0):
    """
    Draw `boxes` (N,7) onto the agent's n_cams images as a single stitched panel.
    boxes are in EGO frame; for agent_idx != 0 the caller must pre-transform them.
    """
    images = scene["images"][agent_idx].cpu().numpy()     # (n_cams, 3, H, W)
    intrins = scene["intrins"][agent_idx].cpu().numpy()
    rots = scene["rots"][agent_idx].cpu().numpy()
    trans = scene["trans"][agent_idx].cpu().numpy()
    n_cams = images.shape[0]

    fig, axs = plt.subplots(1, n_cams, figsize=(4 * n_cams, 4))
    if n_cams == 1:
        axs = [axs]
    cam_names = ["front", "right", "back", "left"][:n_cams]

    for cam in range(n_cams):
        img = np.transpose(images[cam], (1, 2, 0)).clip(0, 1)
        axs[cam].imshow(img)
        axs[cam].set_title(cam_names[cam] if cam < len(cam_names) else f"cam{cam}")
        axs[cam].axis("off")
        H_img, W_img = img.shape[:2]

        for bi, b in enumerate(boxes):
            cx, cy, cz, w, l, h, yaw = b
            corners3d = _box_corners_3d(cx, cy, cz, w, l, h, yaw)
            uv = _project_box_to_cam(corners3d, rots[cam], trans[cam], intrins[cam], H_img, W_img)
            color = box_colors[bi] if bi < len(box_colors) else "yellow"
            for (a, bidx) in _EDGES:
                if uv[a] is None or uv[bidx] is None:
                    continue
                ua, va = uv[a]; ub, vb = uv[bidx]
                if not (0 <= ua <= W_img * 1.5 and 0 <= ub <= W_img * 1.5):
                    continue
                axs[cam].plot([ua, ub], [va, vb], color=color, linewidth=1.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------
# Multi-method qualitative grid (mirrors JigsawComm Appendix F Fig. 2:
# rows = scenes, columns = methods, GT green / predictions red)
# ------------------------------------------------------------------
def render_multi_method_grid(cfg, models: dict, dataset, save_path: str,
                             n_scenes: int = 3, score_thresh: float = 0.15):
    """
    models : dict[name -> (cfg_for_that_model, trained_model)]
             Each model may have its own cfg (different flags), but all must
             share the same grid_H/grid_W/cell_m/bev_x_min/... geometry so
             boxes are comparable across columns.
    dataset: a dataset whose __getitem__ returns the scene dict (SAME scenes
             are reused across all methods/columns for a fair comparison).

    Convention (matches JigsawComm's qualitative figure):
        GT boxes        -> solid GREEN
        predicted boxes -> dotted RED (no TP/FP color-split here -- this
                           figure is for visual comparison across methods,
                           not a metric-consistent figure; use render_bev
                           for the TP/FP/FN-colored single-method version)
    """
    import torch
    method_names = list(models.keys())
    n_methods = len(method_names)

    fig, axs = plt.subplots(n_scenes, n_methods,
                            figsize=(3.6 * n_methods, 3.6 * n_scenes),
                            squeeze=False)

    for row in range(n_scenes):
        scene = dataset[row]
        gt_boxes = scene["gt"]["boxes"].cpu().numpy()

        for col, name in enumerate(method_names):
            cfg_m, model_m = models[name]
            device = next(model_m.parameters()).device
            scene_dev = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in scene.items()}
            scene_dev["gt"] = {k: v.to(device) for k, v in scene["gt"].items()}
            model_m.eval()
            with torch.no_grad():
                out = model_m(scene_dev, hard=True, budget_frac=cfg_m.budget_frac)
            pred_boxes, pred_scores = decode(out, cfg_m, score_thresh=score_thresh)

            ax = axs[row, col]
            ax.scatter([0], [0], marker="^", s=90, c="black", zorder=5)

            for g in gt_boxes:
                cx, cy, cz, w, l, h, yaw = g
                corners = _box_corners_xy(cx, cy, w, l, yaw)
                poly = MplPolygon(corners, closed=True, fill=False,
                                  edgecolor="#2ca02c", linewidth=1.8)
                ax.add_patch(poly)

            for d in pred_boxes:
                cx, cy, cz, w, l, h, yaw = d
                corners = _box_corners_xy(cx, cy, w, l, yaw)
                poly = MplPolygon(corners, closed=True, fill=False,
                                  edgecolor="#d62728", linewidth=1.4, linestyle=":")
                ax.add_patch(poly)

            ax.set_xlim(cfg_m.bev_x_min, cfg_m.bev_x_max)
            ax.set_ylim(cfg_m.bev_y_min, cfg_m.bev_y_max)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(name, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"scene {row}", fontsize=10)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#2ca02c", lw=1.8, label="Ground truth"),
        Line2D([0], [0], color="#d62728", lw=1.4, ls=":", label="Prediction"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10,
              bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Qualitative comparison across methods", y=1.06, fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------
# One-call convenience: grab N scenes, render BEV + camera panels
# ------------------------------------------------------------------
def make_qualitative_figures(cfg, model, dataset, n_scenes: int = 5, out_dir: str = "plots/figures"):
    """
    For each of n_scenes scenes:
        1) BEV figure — GT (solid/dashed) vs predictions (TP blue / FP orange / FN red)
        2) Camera overlay — true 3D wireframe cuboids projected into each camera image
        3) boxes_<i>.json — machine-readable 3D boxes for downstream analysis

    Model is assumed to have already been loaded from best.pth by the caller.
    """
    import os
    import json
    import torch
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    paths = []

    for i in range(min(n_scenes, len(dataset))):
        scene = dataset[i]
        scene_dev = {k: (v.to(device) if hasattr(v, "to") else v)
                     for k, v in scene.items() if k != "gt"}
        scene_dev["gt"] = {k: v.to(device) for k, v in scene["gt"].items()}

        with torch.no_grad():
            out = model(scene_dev, hard=True, budget_frac=cfg.budget_frac)

        # 1) BEV figure (GT vs predictions, colour-coded TP/FP/FN)
        bev_path = f"{out_dir}/qual_bev_{i}.png"
        render_bev(out, scene, cfg, bev_path, score_thresh=0.15)
        paths.append(bev_path)

        # 2) Camera 3D-box overlay
        pred_boxes, pred_scores = decode(out, cfg, score_thresh=0.15)
        gt_boxes   = scene["gt"]["boxes"].cpu().numpy()
        gt_status, _ = _match_for_viz(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.3)
        colors = ["#1f77b4" if (i < len(gt_status) and gt_status[i] == "TP")
                  else "#ff7f0e" for i in range(len(pred_boxes))]

        cam_path = f"{out_dir}/qual_cam_{i}.png"
        render_camera_overlay(scene, cfg, pred_boxes, colors, cam_path, agent_idx=0)
        paths.append(cam_path)

        # 3) JSON: machine-readable 3D bounding boxes
        boxes_json = {
            "scene_idx":  i,
            "n_agents":   int(scene["n_agents"]),
            "gt_boxes":   gt_boxes.tolist(),               # list of [cx,cy,cz,w,l,h,yaw]
            "pred_boxes": pred_boxes.tolist() if hasattr(pred_boxes, "tolist")
                          else [b.tolist() for b in pred_boxes],
            "pred_scores": pred_scores.tolist() if hasattr(pred_scores, "tolist")
                           else [float(s) for s in pred_scores],
            "kb_transmitted": float(out.get("n_cells", 0)) * cfg.per_cell_bytes / 1024.0,
        }
        json_path = f"{out_dir}/boxes_{i}.json"
        with open(json_path, "w") as f:
            json.dump(boxes_json, f, indent=2)
        paths.append(json_path)

        print(f"  scene {i:2d} | GT={len(gt_boxes):3d}  Pred={len(pred_boxes):3d}  "
              f"KB={boxes_json['kb_transmitted']:.1f}")

    print(f"Wrote {len(paths)} outputs to {out_dir}/")
    return paths
