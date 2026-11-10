"""Does per-pixel depth entropy actually flag depth ERRORS, or is the model confidently wrong?

Before moving the uncertainty measurement from BEV into image space, check that image-space entropy
is informative at all. On LiDAR-labelled val pixels, per distance band:
  - Spearman(normalised per-pixel entropy, |depth error|)
  - AUROC of entropy as a detector of |depth error| > 2 m
If entropy is uninformative (AUROC ~ 0.5), the model is confidently wrong and an entropy-based
reliability term cannot work however it is pooled.

    python entropy_calibration.py <ckpt> <n_frames>
"""
import sys, os
sys.path.insert(0, "/mnt/scratch2/users/2641309/projects/coop_v2v")
from dataclasses import replace
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
from src.config import Config
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.lss_backbone import LSSBackbone
from src.metrics import _to_device
from src.depth_bins import depth_bin_edges

ckpt, n_frames = sys.argv[1], int(sys.argv[2])
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
cfg = replace(Config(), device="cpu", augment=False, depth_source="lidar",
              depth_loss_weight="inv_sqrt_freq")          # arm B
backbone = LSSBackbone(cfg)
sd = torch.load(ckpt, map_location="cpu", weights_only=False)["model"]
backbone.load_state_dict({k[9:]: v for k, v in sd.items() if k.startswith("backbone.")})
backbone.eval()
print(f"checkpoint {ckpt}")

edges = depth_bin_edges(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K, getattr(cfg, "depth_bins", "linear"))
ctr = (edges[:-1] + edges[1:]) / 2
pts = torch.linspace(cfg.lss_d_min, cfg.lss_d_max, cfg.depth_K)

ds = OPV2VDataset(cfg.val_dir, cfg, max_agents=1, use_gt_pose=cfg.use_gt_pose)
E, ERR_AM, ERR_EX, TRUE = [], [], [], []
with torch.no_grad():
    for i in np.linspace(0, len(ds) - 1, n_frames).astype(int):
        s = _to_device(opv2v_collate([ds[int(i)]]), torch.device("cpu"))
        _, _, depth_img = backbone._forward_agent(
            s["images"][0], s["intrins"][0], s["rots"][0], s["trans"][0], None, None)
        lab = s["depth_gt_img"][0]                      # (n_cams, fH, fW) bin index, -1 = no label
        m = lab >= 0
        if not m.any():
            continue
        p = F.softmax(depth_img.float(), dim=1)         # (n_cams, K, fH, fW)
        ent = (-(p * (p + 1e-9).log()).sum(1) / np.log(cfg.depth_K))      # (n_cams, fH, fW)
        am = pts[p.argmax(1)]
        ex = (p * pts.view(1, -1, 1, 1)).sum(1)
        true = torch.tensor(ctr)[lab.clamp(min=0)]
        E.append(ent[m].numpy()); ERR_AM.append((am[m] - true[m]).abs().numpy())
        ERR_EX.append((ex[m] - true[m]).abs().numpy()); TRUE.append(true[m].numpy())

E = np.concatenate(E); EA = np.concatenate(ERR_AM); EX = np.concatenate(ERR_EX); T = np.concatenate(TRUE)
print(f"{len(E)} LiDAR-labelled pixels from {n_frames} val frames\n")


def auroc(score, pos):
    """P(score[positive] > score[negative]) via the rank identity; positives = error > 2 m."""
    pos = pos.astype(bool)
    if pos.all() or not pos.any():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


print(f"{'band':>12} {'n':>8} {'entropy':>9} {'|err| am':>9} {'|err| ex':>9} "
      f"{'rho(ent,|err|)':>15} {'AUROC>2m':>9} {'frac>2m':>8}")
rows = {}
for lo, hi in [(4, 90), (4, 20), (20, 35), (35, 51.2), (51.2, 90)]:
    s = (T >= lo) & (T < hi)
    if s.sum() < 50:
        continue
    rho = spearmanr(E[s], EA[s])[0]
    a = auroc(E[s], EA[s] > 2.0)
    lab = f"{lo:g}-{hi:g}"
    rows[lab] = (rho, a, int(s.sum()))
    print(f"{lab:>12} {s.sum():>8} {E[s].mean():>9.4f} {EA[s].mean():>9.2f} {EX[s].mean():>9.2f} "
          f"{rho:>15.4f} {a:>9.4f} {np.mean(EA[s] > 2.0):>8.3f}")

rho_all, a_all, _ = rows["4-90"]
mid = [rows[k][1] for k in ("20-35", "35-51.2") if k in rows]
print(f"\nDECISION RULE: AUROC overall >= 0.65 AND > 0.5 at 20-51 m")
print(f"  overall AUROC = {a_all:.4f}  ->  {'MET' if a_all >= 0.65 else 'NOT MET'}")
for k in ("20-35", "35-51.2"):
    if k in rows:
        print(f"  {k} m AUROC = {rows[k][1]:.4f}  ->  {'MET' if rows[k][1] > 0.5 else 'NOT MET'}")
ok = a_all >= 0.65 and all(x > 0.5 for x in mid)
print(f"\nVERDICT: {'PROCEED with image-space reliability' if ok else 'STOP -- entropy does not flag errors'}")
