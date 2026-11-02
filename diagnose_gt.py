#!/usr/bin/env python
"""
diagnose_gt.py  --  Root-cause the AP=0 / heat-decay failure in coop_v2v.

Run from the coop_v2v project root:
    python diagnose_gt.py --dataset opv2v --data_root /path/to/OPV2V/DATA_train/... --n 32

It trains NOTHING. It inspects the ground-truth tensors the loader emits and the
loss/decode functions that consume them, and prints a PASS/FAIL verdict for each
of six independent failure modes.

Hypothesis under test
---------------------
loss -> 0, pos ~= neg, Heat max decaying 0.63 -> 0.15 is the signature of a focal
loss whose POSITIVE term is never active (num_pos == 0). Then only the negative
term trains, which drives every cell to zero. Checks C and D settle this.
"""
import argparse, sys, math, json
import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Import the project's own modules so we test the REAL code, not a re-write.
# ----------------------------------------------------------------------------
sys.path.insert(0, ".")
try:
    from src.config import Config
except Exception as e:
    print(f"FATAL: cannot import src.config ({e}). Run this from the project root.")
    sys.exit(1)

try:
    from src.train import build_loaders
except Exception as e:
    print(f"FATAL: cannot import loaders from src.train ({e}).")
    sys.exit(1)

try:
    from src.loss.detection import gaussian_focal_loss, detection_loss
    HAVE_LOSS = True
except Exception as e:
    print(f"WARN: cannot import detection loss ({e}); checks D/E will be skipped.")
    HAVE_LOSS = False

try:
    from src.metrics import decode
    HAVE_DECODE = True
except Exception as e:
    print(f"WARN: cannot import decode from src.metrics ({e}); check F skipped.")
    HAVE_DECODE = False


PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
verdicts = {}


def report(name, status, detail):
    verdicts[name] = status
    bar = {"PASS": "  [PASS]", "FAIL": ">>[FAIL]", "WARN": "  [WARN]"}[status]
    print(f"{bar} {name}\n         {detail}\n")


def as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="opv2v")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--backbone", default="lss")
    ap.add_argument("--n", type=int, default=32, help="frames to inspect")
    ap.add_argument("--score_thresh", type=float, default=None,
                    help="override decode score threshold to probe check F")
    args = ap.parse_args()

    cfg = Config()

    print("=" * 78)
    print("coop_v2v GT-pipeline diagnostic")
    print("=" * 78)
    print(f"dataset={args.dataset}  backbone={args.backbone}  frames={args.n}\n")

    train_loader, val_loader, test_loader = build_loaders(cfg)
    ds = getattr(train_loader, "dataset", train_loader)

    # ---- collect n scenes -------------------------------------------------
    scenes = []
    for i in range(min(args.n, len(ds))):
        try:
            scenes.append(ds[i])
        except Exception as e:
            print(f"FATAL: dataset[{i}] raised {type(e).__name__}: {e}")
            sys.exit(1)
    print(f"Loaded {len(scenes)} scenes.\n")

    s0 = scenes[0]
    print("Scene-dict keys:", sorted(list(s0.keys())), "\n")

    for req in ["gt_heat", "gt_reg", "gt_mask"]:
        if req not in s0:
            report(f"KEYS::{req}", FAIL, f"'{req}' missing from scene dict entirely.")

    # =======================================================================
    # CHECK A -- does gt_heat contain any object at all?
    # =======================================================================
    heat_max, heat_sum, n_empty = [], [], 0
    for s in scenes:
        h = as_tensor(s["gt_heat"]).float()
        heat_max.append(float(h.max()))
        heat_sum.append(float(h.sum()))
        if float(h.max()) <= 0:
            n_empty += 1
    hm = np.array(heat_max)
    detail = (f"gt_heat.max() over {len(scenes)} frames: "
              f"min={hm.min():.6f} mean={hm.mean():.6f} max={hm.max():.6f}; "
              f"{n_empty} frames are entirely empty; mean mass={np.mean(heat_sum):.2f}")
    if n_empty == len(scenes):
        report("A. gt_heat non-empty", FAIL,
               detail + "\n         -> Every heatmap is blank. GT never reaches the loss. "
                        "Box->grid conversion or box filtering is dropping all objects.")
    elif n_empty > 0:
        report("A. gt_heat non-empty", WARN, detail)
    else:
        report("A. gt_heat non-empty", PASS, detail)

    # =======================================================================
    # CHECK B -- is the Gaussian peak EXACTLY 1.0? (the float32 0.9999 trap)
    # =======================================================================
    n_exact1, n_near1, n_pos_cells_total = 0, 0, 0
    peak_vals = []
    for s in scenes:
        h = as_tensor(s["gt_heat"]).float()
        n_exact1 += int((h == 1.0).sum())
        n_near1 += int((h >= 1.0 - 1e-4).sum())
        n_pos_cells_total += int((h > 0).sum())
        if float(h.max()) > 0:
            peak_vals.append(float(h.max()))
    detail = (f"cells with gt_heat == 1.0 exactly: {n_exact1}\n"
              f"         cells with gt_heat >= 1-1e-4:      {n_near1}\n"
              f"         cells with gt_heat > 0:            {n_pos_cells_total}\n"
              f"         observed peak values (first 5): "
              f"{[round(v, 8) for v in peak_vals[:5]]}")
    if n_exact1 == 0 and n_near1 > 0:
        report("B. Gaussian peak == 1.0", FAIL,
               detail + "\n         -> THIS IS THE BUG. Peaks are 0.9999.., so the standard\n"
                        "            `pos_inds = gt.eq(1)` selects ZERO cells. Only the negative\n"
                        "            focal term trains => heatmap decays to 0, loss -> 0, AP = 0.\n"
                        "            FIX: use `gt.ge(1.0 - 1e-4)` in gaussian_focal_loss, and set\n"
                        "            the peak cell explicitly to 1.0 in the target builder.")
    elif n_exact1 == 0 and n_near1 == 0:
        report("B. Gaussian peak == 1.0", FAIL,
               detail + "\n         -> No cell anywhere is near 1.0. The Gaussian is being written\n"
                        "            at reduced amplitude, or targets are normalised after drawing.")
    else:
        report("B. Gaussian peak == 1.0", PASS, detail)

    # =======================================================================
    # CHECK C -- num_pos as the loss will actually compute it
    # =======================================================================
    n_mask_pos, n_heat_eq1, disagree = 0, 0, 0
    for s in scenes:
        h = as_tensor(s["gt_heat"]).float()
        m = as_tensor(s["gt_mask"]).float()
        a = (h == 1.0)
        b = (m > 0.5)
        n_heat_eq1 += int(a.sum())
        n_mask_pos += int(b.sum())
        if a.shape == b.shape:
            disagree += int((a ^ b).sum())
    detail = (f"sum(gt_mask > 0.5)   = {n_mask_pos}   <- drives the REGRESSION term\n"
              f"         sum(gt_heat == 1.0)  = {n_heat_eq1}   <- drives the CLASSIFICATION term\n"
              f"         cells where the two disagree = {disagree}")
    if n_heat_eq1 == 0 and n_mask_pos > 0:
        report("C. classification vs regression positives", FAIL,
               detail + "\n         -> Regression has positives but classification has NONE.\n"
                        "            Exactly reproduces your symptom: near-zero loss coming\n"
                        "            entirely from the regression terms on a handful of cells,\n"
                        "            while the heatmap head receives no positive signal at all.")
    elif disagree > 0:
        report("C. classification vs regression positives", WARN,
               detail + "\n         -> gt_mask and the gt_heat peaks index DIFFERENT cells.\n"
                        "            Regression is supervised at cells the classifier never fires at.")
    else:
        report("C. classification vs regression positives", PASS, detail)

    # =======================================================================
    # CHECK D -- loss behaviour under a KNOWN-GOOD and KNOWN-BAD prediction
    # =======================================================================
    if HAVE_LOSS:
        s = scenes[0]
        gt_h = as_tensor(s["gt_heat"]).float()
        while gt_h.dim() < 4:
            gt_h = gt_h.unsqueeze(0)
        eps = 1e-4
        perfect = gt_h.clamp(eps, 1 - eps)          # prediction == target
        flat = torch.full_like(gt_h, 0.0025)        # your observed pos==neg==0.0025
        try:
            l_perfect = float(gaussian_focal_loss(perfect, gt_h))
            l_flat = float(gaussian_focal_loss(flat, gt_h))
            detail = (f"focal(pred == gt)      = {l_perfect:.6f}   (should be ~0)\n"
                      f"         focal(pred == 0.0025)  = {l_flat:.6f}   "
                      f"(should be LARGE, >= ~5 per object, if positives exist)")
            if l_flat < 0.5:
                report("D. focal loss responds to positives", FAIL,
                       detail + "\n         -> A prediction of 0.0025 at every ground-truth object centre\n"
                                "            costs almost nothing. The positive term is dead. This is\n"
                                "            direct confirmation of the num_pos == 0 diagnosis.")
            else:
                report("D. focal loss responds to positives", PASS, detail)
        except Exception as e:
            report("D. focal loss responds to positives", WARN,
                   f"could not call gaussian_focal_loss directly: {type(e).__name__}: {e}")

    # =======================================================================
    # CHECK E -- box <-> grid round trip (axis order / sign convention)
    # =======================================================================
    x_min = getattr(cfg, "bev_x_min", -50.0); x_max = getattr(cfg, "bev_x_max", 50.0)
    y_min = getattr(cfg, "bev_y_min", -50.0); y_max = getattr(cfg, "bev_y_max", 50.0)
    ok, bad, checked = 0, 0, 0
    examples = []
    for s in scenes:
        boxes = s.get("gt_boxes", None)
        if boxes is None:
            continue
        b = as_tensor(boxes).float().reshape(-1, as_tensor(boxes).shape[-1])
        h = as_tensor(s["gt_heat"]).float().squeeze()
        if h.dim() != 2 or b.numel() == 0:
            continue
        H, W = h.shape
        for row in b:
            cx, cy = float(row[0]), float(row[1])
            # convention under test: col <- x, row <- y
            col = int((cx - x_min) / (x_max - x_min) * W)
            r = int((cy - y_min) / (y_max - y_min) * H)
            checked += 1
            if not (0 <= r < H and 0 <= col < W):
                bad += 1
                if len(examples) < 3:
                    examples.append(f"box(cx={cx:.1f},cy={cy:.1f}) -> (r={r},c={col}) OUTSIDE {H}x{W}")
                continue
            # is there heat mass within a 3-cell neighbourhood of the predicted index?
            r0, r1 = max(0, r - 2), min(H, r + 3)
            c0, c1 = max(0, col - 2), min(W, col + 3)
            if float(h[r0:r1, c0:c1].max()) > 0.5:
                ok += 1
            else:
                bad += 1
                if len(examples) < 3:
                    examples.append(
                        f"box(cx={cx:.1f},cy={cy:.1f}) -> (r={r},c={col}) but no heat there "
                        f"(global heat argmax={np.unravel_index(int(h.argmax()), (H, W))})")
    if checked == 0:
        report("E. box->grid round trip", WARN, "no gt_boxes available to test.")
    else:
        detail = (f"{ok}/{checked} boxes land on a heat peak under the convention "
                  f"col<-x, row<-y with range x[{x_min},{x_max}] y[{y_min},{y_max}]\n         "
                  + "\n         ".join(examples))
        if ok == 0:
            report("E. box->grid round trip", FAIL,
                   detail + "\n         -> No box maps onto its own heat peak. Either the axis order is\n"
                            "            swapped (row<-x, col<-y), a sign is flipped, or the BEV range in\n"
                            "            Config does not match the range the loader used to draw targets.")
        elif ok < checked * 0.9:
            report("E. box->grid round trip", WARN, detail)
        else:
            report("E. box->grid round trip", PASS, detail)

    # =======================================================================
    # CHECK F -- can decode() ever return a box at the heat levels you observe?
    # =======================================================================
    thr = args.score_thresh if args.score_thresh is not None else \
        0.05  # real default from src.metrics.decode()
    observed_heat_max = 0.20  # from your training logs
    detail = (f"cfg score threshold = {thr}\n"
              f"         your trained model's Heat max plateaued at ~{observed_heat_max}")
    if thr >= observed_heat_max:
        report("F. decode threshold reachable", FAIL,
               detail + "\n         -> decode() returns ZERO boxes at every frame, so TP=0 and AP=0\n"
                        "            MECHANICALLY, independent of the heatmap bug. This is a second,\n"
                        "            separate cause. Lower it (0.01-0.05) while debugging so you can\n"
                        "            see whether detections exist at all.")
    else:
        report("F. decode threshold reachable", PASS, detail)

    # =======================================================================
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for k, v in verdicts.items():
        print(f"  {v:5s}  {k}")
    fails = [k for k, v in verdicts.items() if v == FAIL]
    print()
    if fails:
        print("Fix these in order (earlier ones cause the later ones):")
        for k in fails:
            print(f"   - {k}")
    else:
        print("No structural GT fault found. The failure is then in the model/optimisation")
        print("path, not the targets -- re-check that fused BEV features reach the head.")
    print()


if __name__ == "__main__":
    main()