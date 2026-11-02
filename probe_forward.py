"""
probe_forward.py -- one real forward pass through the ACTUAL model + ACTUAL loss.
"""
import sys
sys.path.insert(0, ".")
import torch
from src.config import Config
from src.train import build_loaders
from src.models.coop_detector import CoopDetector
from src.loss.detection import detection_loss, gaussian_focal_loss

def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}\n")

    tr, va, te = build_loaders(cfg)
    batch = next(iter(tr))

    model = CoopDetector(cfg).to(device)
    model.eval()

    try:
        from src.metrics import _to_device
        batch = _to_device(batch, device)
    except Exception as e:
        print(f"WARN: could not use _to_device ({e}); attempting manual move")

    with torch.no_grad():
        out = model(batch, hard=True)

    print("out keys:", list(out.keys()))
    heat = out["heat"]
    print(f"out['heat'] shape={tuple(heat.shape)} dtype={heat.dtype} "
          f"min={float(heat.min()):.6f} max={float(heat.max()):.6f} mean={float(heat.mean()):.6f}")

    gt = batch["gt"] if "gt" in batch else None
    if gt is None:
        print("FATAL: batch has no 'gt' key at this level -- check collate_fn structure.")
        return

    gt_heat = gt["heat"]
    print(f"gt['heat']  shape={tuple(gt_heat.shape)} dtype={gt_heat.dtype} "
          f"min={float(gt_heat.min()):.6f} max={float(gt_heat.max()):.6f}")

    h2 = heat
    g2 = gt_heat
    print(f"\nshape match: pred{tuple(h2.shape)} vs gt{tuple(g2.shape)}  "
          f"{'OK' if h2.shape == g2.shape else '<<< MISMATCH -- likely the real bug'}")

    if h2.shape != g2.shape:
        print("STOPPING spatial check: shapes differ, cannot index like-for-like.")
    else:
        pos_mask = g2.eq(1.0)
        n_pos = int(pos_mask.sum())
        print(f"\nn_pos cells in this batch: {n_pos}")
        if n_pos > 0:
            pred_at_pos = h2[pos_mask]
            print(f"pred value AT true positive cells: "
                  f"min={float(pred_at_pos.min()):.6f} "
                  f"mean={float(pred_at_pos.mean()):.6f} "
                  f"max={float(pred_at_pos.max()):.6f}")
            print(f"pred global max (anywhere in the map): {float(h2.max()):.6f}")

            flat_pred = h2.reshape(h2.shape[0], -1) if h2.dim() > 2 else h2.reshape(1, -1)
            flat_gt = g2.reshape(g2.shape[0], -1) if g2.dim() > 2 else g2.reshape(1, -1)
            for b in range(min(3, flat_pred.shape[0])):
                pred_argmax = int(flat_pred[b].argmax())
                gt_argmax = int(flat_gt[b].argmax())
                W = h2.shape[-1]
                print(f"  batch[{b}]: pred peak at flat-idx {pred_argmax} "
                      f"(row={pred_argmax // W}, col={pred_argmax % W}), "
                      f"gt peak at flat-idx {gt_argmax} "
                      f"(row={gt_argmax // W}, col={gt_argmax % W})  "
                      f"{'SAME CELL' if pred_argmax == gt_argmax else 'DIFFERENT -- spatial mismatch'}")

        try:
            hm_loss = gaussian_focal_loss(heat, gt_heat)
            print(f"\ngaussian_focal_loss(out['heat'], gt['heat']) = {float(hm_loss):.6f}")
        except Exception as e:
            print(f"\ngaussian_focal_loss call failed: {type(e).__name__}: {e}")

    try:
        total, parts = detection_loss(out, gt)
        print(f"\ndetection_loss total = {float(total):.6f}")
        print("parts:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in parts.items()})
    except Exception as e:
        print(f"\ndetection_loss call failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
