"""
Cheap, non-disruptive diagnostic: does the current checkpoint's AP recover
meaningfully with a stricter decode() score_thresh, WITHOUT any retraining?

Context: train_log.csv shows n_peaks_mean~87 (vs ~10-17 real GT/frame) and
score_mean~0.82 even at the already-in-use score_thresh=0.20 -- a lot of
high-confidence false positives are surviving. heat_pos_mean (0.65-0.70) is
still meaningfully higher than heat_neg_mean (0.37-0.46), so there may be
real separation being swamped by too permissive a threshold at this stage
of training, rather than (or in addition to) a genuine loss-calibration
problem.

Method: run the model forward ONCE per real validation frame (exactly like
the real evaluate() in src/metrics.py does: hard=True, gumbel_T=cfg.gumbel_T1,
tau=cfg.tau, budget_frac=cfg.budget_frac), cache the raw `out` dict and GT,
then sweep score_thresh through the REAL decode() function against those
cached outputs -- no retraining, no GPU job queue, no re-running the model.

Run on the login node (CPU; EfficientNet-B0 forward on ~40 real multi-agent,
multi-camera frames will take a few minutes):

    python score_thresh_sweep.py <checkpoint_path> <preset_name>

Example:
    python score_thresh_sweep.py results/checkpoints/no_fusion_camfix/last.pth no_fusion
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

import torch

from src.config import Config
from src.baselines import make
from src.train import build_loaders
from src.models.coop_detector import CoopDetector
from src.metrics import decode, Evaluator, _to_device, bytes_kb_from_out


def main():
    if len(sys.argv) < 3:
        print("Usage: python score_thresh_sweep.py <checkpoint_path> <preset_name>")
        sys.exit(1)
    ckpt_path = sys.argv[1]
    preset = sys.argv[2]

    base = Config()
    base.validate()
    cfg = make(preset, base)

    print(f"Loading checkpoint {ckpt_path} ...")
    ck = torch.load(ckpt_path, map_location="cpu")
    model = CoopDetector(cfg)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"Checkpoint epoch: {ck.get('epoch')}  best_ap50: {ck.get('best_ap50')}")

    print("Building real data loaders ...")
    tr, va, te = build_loaders(cfg)

    n_frames = 10
    cached = []   # list of (out_dict, gt_dict)
    it = iter(va)
    print(f"Running forward pass on {n_frames} real validation frames (CPU, may take a few minutes) ...")
    device = torch.device("cpu")
    with torch.no_grad():
        for i in range(n_frames):
            try:
                scene = next(it)
            except StopIteration:
                break
            scene = _to_device(scene, device)
            out = model(scene, hard=True, gumbel_T=cfg.gumbel_T1,
                        tau=cfg.tau, budget_frac=cfg.budget_frac)
            cached.append((out, scene["gt"]))
            if (i + 1) % 10 == 0:
                print(f"  cached {i+1}/{n_frames}")

    print(f"\nCached {len(cached)} real frames. Sweeping score_thresh ...\n")
    for thresh in [0.05, 0.20, 0.40, 0.60, 0.80, 0.90]:
        ev = Evaluator(cfg)
        total_peaks = 0
        for out, gt in cached:
            boxes, scores = decode(out, cfg, score_thresh=thresh, topk=200)
            total_peaks += len(scores)
            gt_boxes = gt["boxes"].cpu().numpy()
            gt_oof = gt["oof"].cpu().numpy()
            kb = bytes_kb_from_out(out, cfg)
            n_cells = int(out.get("n_cells", 0))
            ev.update(boxes, scores, gt_boxes, gt_oof, kb=kb, n_cells=n_cells)

        r = ev.compute()
        mean_peaks = total_peaks / max(len(cached), 1)
        print(f"score_thresh={thresh:.2f}  mean_peaks/frame={mean_peaks:.1f}  "
              f"AP30={r.get('ap30', float('nan')):.4f}  AP50={r.get('ap50', float('nan')):.4f}  "
              f"AP70={r.get('ap70', float('nan')):.4f}  recall50={r.get('recall50', float('nan')):.4f}")


if __name__ == "__main__":
    main()