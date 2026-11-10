"""
Evaluation utilities — full diagnostic suite for journal-quality analysis.

Every call to evaluate() returns ~35 metrics covering:
    - Detection: AP@0.3/0.5/0.7, precision, recall, TP/FP/FN
    - Localisation: centre error, width/length/yaw RMSE (matched boxes only)
    - IoU: mean, median, histogram (10 bins)
    - Heatmap: max, mean, std, positive-pixel mean, negative-pixel mean
    - Decode: n_peaks, score max/mean/min, confidence histogram
    - Communication: KB/frame, n_cells, compression ratio
    - OOF recall
    - Runtime

All metrics saved to CSV every epoch via train.py.
"""
from __future__ import annotations
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .loss.detection import detection_loss

try:
    from shapely.geometry import Polygon
    _HAS_SHAPELY = True
except Exception:
    _HAS_SHAPELY = False


# ------------------------------------------------------------------ #
# Decoder                                                              #
# ------------------------------------------------------------------ #

def _peaks(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    pad = kernel // 2
    hmax = F.max_pool2d(heat.unsqueeze(0).unsqueeze(0),
                        kernel, stride=1, padding=pad).squeeze()
    return (heat == hmax).float() * heat


def _nms_by_centre(boxes, scores, radius_m: float = 2.0):
    """Greedy centre-distance NMS. boxes: (M,7) cx,cy,cz,w,l,h,yaw (numpy),
    already sorted by score descending (decode()'s topk guarantees this).
    Suppresses a lower-scored box if its centre is within radius_m of an
    already-kept higher-scored box. radius_m is a hyperparameter -- not
    derived from data -- and should be swept, not trusted as-is."""
    import numpy as np
    keep = []
    for i in range(boxes.shape[0]):
        ci = boxes[i, :2]
        if all(np.linalg.norm(ci - boxes[k, :2]) > radius_m for k in keep):
            keep.append(i)
    return boxes[keep], scores[keep]


def decode(out: dict, cfg, score_thresh: float = 0.05, topk: int = 50):
    """
    Returns numpy array (M, 7): (cx, cy, cz, w, l, h, yaw) in ego metres,
    and (M,) scores.
    """
    heat   = out["heat"].detach()
    off    = out["offset"].detach()
    sz     = out["size"].detach()
    yaw    = out.get("yaw")
    zh     = out.get("zh")

    peaks  = _peaks(heat)
    # Suppress ego-body footprint: the ego vehicle hood/body creates
    # consistent camera features that cause ghost detections near origin.
    # Zero out peaks within 3m x 2m of ego to match GT exclusion in loader.
    H_p, W_p = peaks.shape
    _x_min = getattr(cfg, "bev_x_min", -51.2)
    _y_min = getattr(cfg, "bev_y_min", -51.2)
    _cell  = cfg.cell_m
    _cols_t = torch.arange(W_p, device=peaks.device, dtype=torch.float32)
    _rows_t = torch.arange(H_p, device=peaks.device, dtype=torch.float32)
    _cx_t = _cols_t * _cell + _x_min   # (W_p,)
    _cy_t = _rows_t * _cell + _y_min   # (H_p,)
    _ego_mask = ((_cx_t.abs() < 3.0).unsqueeze(0) &
                 (_cy_t.abs() < 2.0).unsqueeze(1))  # (H_p, W_p)
    peaks = peaks.masked_fill(_ego_mask, 0.0)
    flat   = peaks.view(-1)
    n_valid = int((flat > score_thresh).sum().item())
    if n_valid == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    n = min(n_valid, topk)
    scores, idx = flat.topk(n)
    H, W  = heat.shape
    rows  = (idx // W).long()
    cols  = (idx %  W).long()

    dx    = off[0, rows, cols]
    dy    = off[1, rows, cols]
    log_w = sz[0, rows, cols]
    log_l = sz[1, rows, cols]

    cx = (cols.float() + dx) * cfg.cell_m + cfg.bev_x_min
    cy = (rows.float() + dy) * cfg.cell_m + cfg.bev_y_min
    w  = torch.exp(log_w).clamp(0.3, 15.0)
    l  = torch.exp(log_l).clamp(0.3, 30.0)

    if yaw is not None:
        s = yaw[0, rows, cols]; c_cos = yaw[1, rows, cols]
        yaw_val = torch.atan2(s, c_cos)
    else:
        yaw_val = torch.zeros_like(cx)

    if zh is not None:
        cz = zh[0, rows, cols]
        h  = zh[1, rows, cols].clamp(0.3, 5.0)
    else:
        cz = torch.zeros_like(cx)
        h  = torch.full_like(cx, 1.5)

    boxes = torch.stack([cx, cy, cz, w, l, h, yaw_val], dim=-1)
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    boxes_np, scores_np = _nms_by_centre(boxes_np, scores_np,
                                          radius_m=getattr(cfg, "nms_radius_m", 2.0))
    return boxes_np, scores_np


# ------------------------------------------------------------------ #
# IoU                                                                  #
# ------------------------------------------------------------------ #

def _box_polygon(cx, cy, w, l, yaw):
    dx = np.array([ l/2,  l/2, -l/2, -l/2])
    dy = np.array([ w/2, -w/2, -w/2,  w/2])
    c, s = math.cos(yaw), math.sin(yaw)
    xs = cx + c * dx - s * dy
    ys = cy + s * dx + c * dy
    return list(zip(xs.tolist(), ys.tolist()))


def rotated_iou(a: np.ndarray, b: np.ndarray) -> float:
    """a, b: (cx, cy, w, l, yaw). Returns BEV IoU in [0, 1]."""
    if _HAS_SHAPELY:
        try:
            pa = Polygon(_box_polygon(a[0], a[1], a[2], a[3], a[4]))
            pb = Polygon(_box_polygon(b[0], b[1], b[2], b[3], b[4]))
            if not pa.is_valid or not pb.is_valid:
                return 0.0
            inter = pa.intersection(pb).area
            union = pa.union(pb).area
            return inter / union if union > 0 else 0.0
        except Exception:
            pass
    # fallback: axis-aligned
    def aabb(x):
        c, s = math.cos(x[4]), math.sin(x[4])
        ew = abs(c) * x[3] + abs(s) * x[2]
        eh = abs(s) * x[3] + abs(c) * x[2]
        return x[0]-ew/2, x[1]-eh/2, x[0]+ew/2, x[1]+eh/2
    ax0,ay0,ax1,ay1 = aabb(a); bx0,by0,bx1,by1 = aabb(b)
    ix0,iy0 = max(ax0,bx0), max(ay0,by0)
    ix1,iy1 = min(ax1,bx1), min(ay1,by1)
    inter = max(0.0,ix1-ix0)*max(0.0,iy1-iy0)
    union = (ax1-ax0)*(ay1-ay0)+(bx1-bx0)*(by1-by0)-inter
    return inter/union if union > 0 else 0.0


# ------------------------------------------------------------------ #
# AP                                                                   #
# ------------------------------------------------------------------ #

def _ap_from_matches(scores: list, tp: list, n_gt: int) -> float:
    """Continuous (all-points / COCO-style) AP -- exact area under the
    monotonic precision envelope, not 11-point interpolation."""
    if n_gt == 0 or len(scores) == 0:
        return 0.0
    order     = np.argsort(-np.array(scores))
    tp        = np.array(tp)[order]
    fp        = 1 - tp
    tp_cum    = np.cumsum(tp)
    fp_cum    = np.cumsum(fp)
    recall    = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    precision = precision.copy()
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision])
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))
    return ap


# ------------------------------------------------------------------ #
# Rich Evaluator                                                        #
# ------------------------------------------------------------------ #

class Evaluator:
    def __init__(self, cfg, iou_thrs=(0.3, 0.5, 0.7)):
        self.cfg      = cfg
        self.iou_thrs = iou_thrs

        # AP state
        self.scores = {t: [] for t in iou_thrs}
        self.tp     = {t: [] for t in iou_thrs}
        self.n_gt   = 0

        # Detection counts
        self.n_pred = 0

        # OOF
        self.n_oof_gt       = 0
        self.oof_recall_num = 0

        # KB / cells
        self.kb_list    = []
        self.cells_list = []

        # Heatmap statistics (accumulated)
        # Validation loss (same detection_loss used in training, for a
        # direct train-vs-val comparison)
        self.val_loss_list = []
        self.heat_max_list  = []
        self.heat_mean_list = []
        self.heat_std_list  = []
        self.heat_pos_list  = []   # mean at GT centre cells
        self.heat_neg_list  = []   # mean everywhere else

        # Decode statistics
        self.score_max_list  = []
        self.score_mean_list = []
        self.n_peaks_list    = []
        self.conf_hist       = np.zeros(10, dtype=np.float64)  # 0.0-1.0 in 10 bins

        # Localisation errors (matched detections only, IoU@0.3)
        self.iou_list        = []   # all IoUs for matched pairs
        self.centre_err_list = []   # euclidean centre error (m)
        self.w_err_list      = []   # |pred_w - gt_w|
        self.l_err_list      = []   # |pred_l - gt_l|
        self.yaw_err_list    = []   # |pred_yaw - gt_yaw| (deg)

    # ----------------------------------------------------------------
    def _collect_heatmap_stats(self, out: dict, gt_mask: np.ndarray):
        """Collect heatmap diagnostics (non-blocking, uses raw tensors)."""
        heat = out["heat"].detach().cpu().float()
        hmax  = float(heat.max())
        hmean = float(heat.mean())
        hstd  = float(heat.std())
        self.heat_max_list.append(hmax)
        self.heat_mean_list.append(hmean)
        self.heat_std_list.append(hstd)

        # positive = GT centre cells (mask=1)
        mask_t = torch.from_numpy(gt_mask).bool()
        if mask_t.any():
            self.heat_pos_list.append(float(heat[mask_t].mean()))
        neg_vals = heat[~mask_t]
        if len(neg_vals) > 0:
            self.heat_neg_list.append(float(neg_vals.mean()))

    def _collect_score_stats(self, scores: np.ndarray):
        if len(scores) == 0:
            self.n_peaks_list.append(0)
            self.score_max_list.append(0.0)
            self.score_mean_list.append(0.0)
            return
        self.n_peaks_list.append(len(scores))
        self.score_max_list.append(float(scores.max()))
        self.score_mean_list.append(float(scores.mean()))
        # confidence histogram
        bins = np.clip((scores * 10).astype(int), 0, 9)
        for b in bins:
            self.conf_hist[b] += 1
    

    def _collect_localisation_fast(self, det_boxes, det_scores, gt_boxes, iou_mat):
        n_pred, n_gt = iou_mat.shape
        if n_pred == 0 or n_gt == 0:
            return
        matched_gt = np.zeros(n_gt, dtype=bool)
        order = np.argsort(-det_scores)
        for k in order:
            best_iou = 0.0; best_j = -1
            for j in range(n_gt):
                if matched_gt[j]:
                    continue
                if iou_mat[k, j] > best_iou:
                    best_iou = iou_mat[k, j]; best_j = j
            if best_j >= 0 and best_iou >= 0.3:
                matched_gt[best_j] = True
                d = det_boxes[k]; g = gt_boxes[best_j]
                self.iou_list.append(best_iou)
                self.centre_err_list.append(
                    math.sqrt((d[0]-g[0])**2 + (d[1]-g[1])**2))
                self.w_err_list.append(abs(d[3] - g[3]))
                self.l_err_list.append(abs(d[4] - g[4]))
                dyaw = math.degrees(abs(d[6] - g[6])) % 360
                if dyaw > 180:
                    dyaw = 360 - dyaw
                self.yaw_err_list.append(dyaw)





    def _collect_localisation(self, det_boxes, det_scores, gt_boxes):
        """For each prediction matched to a GT at IoU@0.3, record errors."""
        if len(det_boxes) == 0 or len(gt_boxes) == 0:
            return
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)
        order = np.argsort(-det_scores)
        for k in order:
            d  = det_boxes[k]
            d_ = np.array([d[0], d[1], d[3], d[4], d[6]])
            best_iou = 0.0; best_j = -1
            for j, g in enumerate(gt_boxes):
                if matched_gt[j]:
                    continue
                g_ = np.array([g[0], g[1], g[3], g[4], g[6]])
                iou = rotated_iou(d_, g_)
                if iou > best_iou:
                    best_iou = iou; best_j = j
            if best_j >= 0 and best_iou >= 0.3:
                matched_gt[best_j] = True
                g = gt_boxes[best_j]
                self.iou_list.append(best_iou)
                self.centre_err_list.append(
                    math.sqrt((d[0]-g[0])**2 + (d[1]-g[1])**2))
                self.w_err_list.append(abs(d[3] - g[3]))
                self.l_err_list.append(abs(d[4] - g[4]))
                # yaw error in degrees, wrapped to [-180, 180]
                dyaw = math.degrees(abs(d[6] - g[6])) % 360
                if dyaw > 180:
                    dyaw = 360 - dyaw
                self.yaw_err_list.append(dyaw)

    # ----------------------------------------------------------------
    
    
    def update(self, det_boxes, det_scores, gt_boxes_np, gt_oof_np, kb: float,
               out: dict = None, n_cells: int = 0):
        self.kb_list.append(kb)
        self.cells_list.append(n_cells)
        self.n_gt   += len(gt_boxes_np)
        self.n_pred += len(det_boxes)

        n_pred = len(det_boxes)
        n_gt   = len(gt_boxes_np)

        # Compute the full IoU matrix ONCE (was computed 4x redundantly before)
        iou_mat = np.zeros((n_pred, n_gt), dtype=np.float32)
        if n_pred > 0 and n_gt > 0:
            gt_5 = np.stack([gt_boxes_np[:, 0], gt_boxes_np[:, 1],
                             gt_boxes_np[:, 3], gt_boxes_np[:, 4],
                             gt_boxes_np[:, 6]], axis=1)
            for k in range(n_pred):
                d = det_boxes[k]
                d_ = np.array([d[0], d[1], d[3], d[4], d[6]])
                for j in range(n_gt):
                    iou_mat[k, j] = rotated_iou(d_, gt_5[j])

        order = (np.argsort(-det_scores) if n_pred > 0 else np.array([], dtype=int))

        # AP matching for all thresholds using the same precomputed matrix
        for t in self.iou_thrs:
            matched = np.zeros(n_gt, dtype=bool)
            for k in order:
                best_i = -1; best_iou = 0.0
                for j in range(n_gt):
                    if matched[j]:
                        continue
                    if iou_mat[k, j] > best_iou:
                        best_iou = iou_mat[k, j]; best_i = j
                if best_i >= 0 and best_iou >= t:
                    matched[best_i] = True
                    self.scores[t].append(float(det_scores[k]))
                    self.tp[t].append(1)
                else:
                    self.scores[t].append(float(det_scores[k]))
                    self.tp[t].append(0)

        # OOF recall — reuse iou_mat, just take max over predictions per GT
        if len(gt_oof_np) > 0 and n_gt > 0:
            oof_idx = np.where(gt_oof_np)[0]
            found = 0
            for j in oof_idx:
                best = iou_mat[:, j].max() if n_pred > 0 else 0.0
                if best >= 0.3:
                    found += 1
            self.oof_recall_num += found
            self.n_oof_gt       += len(oof_idx)

        # Score stats
        self._collect_score_stats(det_scores)

        # Localisation errors — reuse iou_mat instead of recomputing
        self._collect_localisation_fast(det_boxes, det_scores, gt_boxes_np, iou_mat)

        # Heatmap stats (optional, only if out provided)
        if out is not None and "heat" in out:
            from src.data.opv2v import OPV2VDataset   # avoid circular import
            
            
            
            
            
            
            
            H, W = out["heat"].shape[-2], out["heat"].shape[-1]
            dummy_mask = np.zeros((H, W), dtype=np.float32)
            self._collect_heatmap_stats(out, dummy_mask)

    # ----------------------------------------------------------------
    def compute(self) -> dict:
        r = {}

        # AP
        for t in self.iou_thrs:
            r[f"ap{int(t*100):02d}"] = _ap_from_matches(
                self.scores[t], self.tp[t], self.n_gt)

        # Detection counts
        tp50 = int(sum(self.tp[0.5]))
        fp50 = len(self.tp[0.5]) - tp50
        fn50 = max(0, self.n_gt - tp50)
        r["n_gt"]       = int(self.n_gt)
        r["n_pred"]     = int(self.n_pred)
        r["tp50"]       = tp50
        r["fp50"]       = fp50
        r["fn50"]       = fn50
        r["precision50"] = tp50 / max(tp50 + fp50, 1)
        r["recall50"]    = tp50 / max(self.n_gt,   1)

        # OOF
        r["oof_recall"] = (self.oof_recall_num / self.n_oof_gt
                           if self.n_oof_gt > 0 else 0.0)
        r["n_oof_gt"]   = int(self.n_oof_gt)

        # KB / cells
        r["kb"]            = float(np.mean(self.kb_list)) if self.kb_list else 0.0
        r["n_cells_mean"]  = float(np.mean(self.cells_list)) if self.cells_list else 0.0

        # Heatmap
        r["heat_max"]      = float(np.mean(self.heat_max_list))  if self.heat_max_list  else 0.0
        r["heat_mean"]     = float(np.mean(self.heat_mean_list)) if self.heat_mean_list else 0.0
        r["heat_std"]      = float(np.mean(self.heat_std_list))  if self.heat_std_list  else 0.0
        r["heat_pos_mean"] = float(np.mean(self.heat_pos_list))  if self.heat_pos_list  else 0.0
        r["heat_neg_mean"] = float(np.mean(self.heat_neg_list))  if self.heat_neg_list  else 0.0
        r["val_loss_det"]  = float(np.mean(self.val_loss_list))  if self.val_loss_list  else 0.0

        # Decode
        r["n_peaks_mean"]  = float(np.mean(self.n_peaks_list))   if self.n_peaks_list  else 0.0
        r["score_max"]     = float(np.mean(self.score_max_list))  if self.score_max_list else 0.0
        r["score_mean"]    = float(np.mean(self.score_mean_list)) if self.score_mean_list else 0.0

        # Confidence histogram (as string for CSV)
        total = self.conf_hist.sum()
        if total > 0:
            hist_norm = (self.conf_hist / total * 100).round(1)
        else:
            hist_norm = self.conf_hist
        r["conf_hist"] = "|".join(f"{v:.1f}" for v in hist_norm)

        # Localisation errors
        if self.iou_list:
            ious = np.array(self.iou_list)
            r["iou_mean"]   = float(ious.mean())
            r["iou_median"] = float(np.median(ious))
            r["iou_std"]    = float(ious.std())
            # IoU histogram (10 bins 0.0-1.0)
            iou_hist, _ = np.histogram(ious, bins=10, range=(0, 1))
            r["iou_hist"] = "|".join(str(int(v)) for v in iou_hist)
        else:
            r["iou_mean"] = r["iou_median"] = r["iou_std"] = 0.0
            r["iou_hist"] = "|".join(["0"]*10)

        if self.centre_err_list:
            r["centre_rmse"] = float(np.sqrt(np.mean(np.array(self.centre_err_list)**2)))
            r["w_rmse"]      = float(np.sqrt(np.mean(np.array(self.w_err_list)**2)))
            r["l_rmse"]      = float(np.sqrt(np.mean(np.array(self.l_err_list)**2)))
            r["yaw_rmse"]    = float(np.sqrt(np.mean(np.array(self.yaw_err_list)**2)))
        else:
            r["centre_rmse"] = r["w_rmse"] = r["l_rmse"] = r["yaw_rmse"] = 0.0

        return r


# ------------------------------------------------------------------ #
# Byte accounting                                                       #
# ------------------------------------------------------------------ #

def bytes_kb_from_out(out: dict, cfg) -> float:
    if cfg.no_fusion:
        return 0.0
    n = out.get("n_cells", 0)
    cell_bytes = n * cfg.per_cell_bytes
    # The dense control map is sent by EACH collaborator, so it scales with the number of senders.
    # It was previously counted once regardless of agent count, which under-reported multi-agent
    # bandwidth; it is also now included in the differentiable soft_B (coop_detector), so the two
    # byte figures agree.
    n_senders = max(int(out.get("n_senders", 1)), 0)
    util_map_bytes = (cfg.grid_H * cfg.grid_W * cfg.b_util * n_senders
                      if getattr(cfg, "count_control_bytes", True) else 0)
    return (cell_bytes + util_map_bytes) / 1024.0


# ------------------------------------------------------------------ #
# Main evaluate function                                               #
# ------------------------------------------------------------------ #

@torch.no_grad()
def evaluate(model, loader, cfg, hard: bool = True,
             tau: float | None = None,
             budget_frac: float | None = None) -> dict:
    model.eval()
    ev     = Evaluator(cfg)
    device = next(model.parameters()).device
    t0     = time.time()

    for scene in tqdm(loader, desc="  validation", leave=False):
        scene  = _to_device(scene, device)
        out    = model(scene, hard=hard, gumbel_T=cfg.gumbel_T1,
                       tau=tau, budget_frac=budget_frac)
        _val_loss, _ = detection_loss(out, scene["gt"],
                                      hm_norm=getattr(cfg, "hm_norm", "sepnorm"),
                                      hm_pos_weight=getattr(cfg, "hm_pos_weight", 4.0))
        ev.val_loss_list.append(float(_val_loss.item()))
        boxes, scores = decode(out, cfg, score_thresh=getattr(cfg, "eval_score_thresh", 0.20), topk=200)
        gt_boxes = scene["gt"]["boxes"].cpu().numpy()
        gt_oof   = scene["gt"]["oof"].cpu().numpy()
        kb       = bytes_kb_from_out(out, cfg)
        n_cells  = int(out.get("n_cells", 0))

        # Heatmap stats
        heat = out["heat"].detach().cpu().float()
        ev.heat_max_list.append(float(heat.max()))
        ev.heat_mean_list.append(float(heat.mean()))
        ev.heat_std_list.append(float(heat.std()))

        # positive / negative heat
        gt_mask = scene["gt"]["mask"].cpu().numpy().astype(bool)
        if gt_mask.any():
            heat_np = heat.numpy()
            ev.heat_pos_list.append(float(heat_np[gt_mask].mean()))
            ev.heat_neg_list.append(float(heat_np[~gt_mask].mean()))

        ev._collect_score_stats(scores)
        ev.update(boxes, scores, gt_boxes, gt_oof, kb=kb, n_cells=n_cells)

    r = ev.compute()
    r["val_time_s"] = float(time.time() - t0)
    return r


# ------------------------------------------------------------------ #
# Device helper                                                         #
# ------------------------------------------------------------------ #

def _to_device(scene, device):
    def _mv(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: _mv(v) for k, v in x.items()}
        return x
    return _mv(scene)
    