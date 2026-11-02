"""
precompute_depth_prior.py

One-time offline precomputation of monocular depth priors for all OPV2V
camera images using ZoeDepth (ZoeD_NK variant -- best match to CARLA
synthetic imagery, confirmed against real LiDAR: 90.8% vs 91.4% in-range).

Output: for every *_camera{0,1,2,3}.png, writes a corresponding
        *_camera{0,1,2,3}_depth_prior.npy file containing a (fH, fW) int16
        array of depth-bin indices in {0..depth_K-1}, -1 = out of range.

        fH = IMG_H // lss_downsample  (e.g. 600//8 = 75)
        fW = IMG_W // lss_downsample  (e.g. 800//8 = 100)

This is fully resumable: already-processed files are skipped.
Run as a single GPU SLURM job -- see slurm/precompute_depth.sh.

Usage:
    python precompute_depth_prior.py \
        --splits train val test \
        --d_min 4.0 --d_max 50.0 --depth_K 64 \
        --downsample 8
"""
import argparse
import os
import glob
import time
import numpy as np
import torch
from PIL import Image

DATA_ROOTS = {
    "train": "/mnt/scratch2/users/2641309/datasets/OPV2V/DATA_train/train_chunks/train",
    "val":   "/mnt/scratch2/users/2641309/datasets/OPV2V/DATA_validate/validate_chunks/validate",
    "test":  "/mnt/scratch2/users/2641309/datasets/OPV2V/DATA_test/test_chunks/test",
}
N_CAMS = 4


def build_file_list(splits):
    paths = []
    for split in splits:
        root = DATA_ROOTS[split]
        for cam in range(N_CAMS):
            found = sorted(glob.glob(
                os.path.join(root, "*", "*", f"*_camera{cam}.png")))
            paths.extend(found)
    return paths


def depth_to_bins(depth_m: np.ndarray, d_min: float, d_max: float,
                  depth_K: int, downsample: int) -> np.ndarray:
    """
    depth_m   : (H, W) float32, metric depth in metres from ZoeDepth
    Returns   : (fH, fW) int16, bin index in [0, depth_K-1], -1 = invalid
    """
    H, W = depth_m.shape
    fH, fW = H // downsample, W // downsample

    # Downsample depth by averaging (more stable than nearest for supervision)
    depth_ds = depth_m.reshape(fH, downsample, fW, downsample).mean(axis=(1, 3))

    edges = np.linspace(d_min, d_max, depth_K + 1, dtype=np.float32)
    bins = np.digitize(depth_ds, edges) - 1          # 0-indexed

    # Clip and mask out-of-range
    valid = (depth_ds >= d_min) & (depth_ds <= d_max)
    bins = np.clip(bins, 0, depth_K - 1).astype(np.int16)
    bins[~valid] = -1
    return bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=["train", "val", "test"])
    ap.add_argument("--d_min",    type=float, default=4.0)
    ap.add_argument("--d_max",    type=float, default=50.0)
    ap.add_argument("--depth_K",  type=int,   default=64)
    ap.add_argument("--downsample", type=int, default=8)
    ap.add_argument("--batch",    type=int,   default=1,
                    help="Images per GPU batch (1 is safe; increase if VRAM allows)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading ZoeDepth ZoeD_NK...")
    model = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK",
                           pretrained=True, trust_repo=True)
    model = model.to(device)
    model.eval()
    print("ZoeDepth loaded.")

    all_paths = build_file_list(args.splits)
    total = len(all_paths)
    print(f"Total images to process: {total}")

    # Count already done
    done = sum(1 for p in all_paths
               if os.path.exists(p.replace(".png", "_depth_prior.npy")))
    print(f"Already processed: {done} / {total}  (skipping these)")

    t0 = time.time()
    processed = 0
    errors = 0

    for i, img_path in enumerate(all_paths):
        out_path = img_path.replace(".png", "_depth_prior.npy")
        if os.path.exists(out_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            with torch.no_grad():
                depth_m = model.infer_pil(img)   # (H, W) numpy float32, metres

            bins = depth_to_bins(depth_m, args.d_min, args.d_max,
                                 args.depth_K, args.downsample)
            np.save(out_path, bins)
            processed += 1

        except Exception as e:
            print(f"ERROR on {img_path}: {e}")
            errors += 1
            continue

        if processed % 100 == 0 and processed > 0:
            elapsed = time.time() - t0
            rate = processed / elapsed
            remaining = (total - done - processed) / max(rate, 1e-6)
            print(f"  [{processed}/{total-done}]  "
                  f"{rate:.2f} img/s  "
                  f"ETA: {remaining/3600:.1f}h  "
                  f"errors={errors}")

    elapsed = time.time() - t0
    print(f"\nDone. Processed {processed} images in {elapsed/3600:.2f}h  "
          f"({errors} errors)")
    print(f"Output format: (fH={600//args.downsample}, "
          f"fW={800//args.downsample}) int16 depth-bin arrays")


if __name__ == "__main__":
    main()