"""Gate: does the ego's receiver 'need' map go HIGH in BEV cells the ego cannot see?

`need` drives receiver-centric selection (coop_detector: need = 1 - ego_vis = D_ego, the normalised
entropy of the ego's splatted depth distribution). If a cell receives no frustum point at all, its
splatted depth column is all zeros. Should the entropy of that empty column come out LOW, the ego
would request nothing exactly where it is blind -- inverting the whole point of receiver-centric
selection. This measures `need` on real frames, split by whether the cell got any geometric coverage.

Coverage is pure geometry (frustum_to_ego -> _bev_coords), so the uncovered-cell result does not
depend on the weights: an empty column is empty whatever the encoder predicts. Pass a checkpoint to
make the covered-cell numbers realistic too.

    python scripts/check_need_uncovered.py [--ckpt PATH] [--frames 6]

Exits non-zero unless uncovered cells have need >= 0.95 (they should be exactly 1.0: a uniform
distribution over depth bins is maximum entropy).
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace
import numpy as np, torch
from src.config import Config
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.lss_backbone import LSSBackbone, frustum_to_ego
from src.models.depth_aware_utility import depth_uncertainty
from src.metrics import _to_device

BANDS = [(0, 20), (20, 35), (35, 51.2)]


def measure(cfg, backbone, ds, idxs, uncertainty_fn=depth_uncertainty):
    """Returns (need_uncovered, need_covered, per-band covered need, coverage fraction)."""
    unc, cov, bands = [], [], {b: [] for b in BANDS}
    frac_cov = []
    H, W, cell = cfg.grid_H, cfg.grid_W, cfg.cell_m
    # BEV cell centre distance from the ego, for the per-band split
    rr, cc = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cx = cc.float() * cell + cfg.bev_x_min + cell / 2
    cy = rr.float() * cell + cfg.bev_y_min + cell / 2
    dist = torch.hypot(cx, cy)

    with torch.no_grad():
        for i in idxs:
            scene = _to_device(opv2v_collate([ds[int(i)]]), torch.device("cpu"))
            # ---- ego only: the receiver 'need' is built from agent 0 ----
            img = scene["images"][0]
            pr = scene["post_rots"][0] if scene.get("post_rots") is not None else None
            pt = scene["post_trans"][0] if scene.get("post_trans") is not None else None
            _, depth_bev, _ = backbone._forward_agent(
                img, scene["intrins"][0], scene["rots"][0], scene["trans"][0], pr, pt,
                depth_scale=float(scene.get("depth_scale", 1.0)))
            need = uncertainty_fn(torch.log(depth_bev + 1e-9).unsqueeze(0))[0]   # (H, W)

            # ---- coverage, from the same geometry the lift uses (weight-independent) ----
            pts = frustum_to_ego(backbone.frustum, scene["rots"][0], scene["trans"][0],
                                 scene["intrins"][0], pr, pt,
                                 depth_scale=float(scene.get("depth_scale", 1.0)))
            row, col, valid = backbone._bev_coords(pts)
            covered = torch.zeros(H, W, dtype=torch.bool)
            r, c = row.reshape(-1)[valid.reshape(-1)], col.reshape(-1)[valid.reshape(-1)]
            covered[r.long(), c.long()] = True

            unc.append(need[~covered].numpy())
            cov.append(need[covered].numpy())
            frac_cov.append(float(covered.float().mean()))
            for b in BANDS:
                m = covered & (dist >= b[0]) & (dist < b[1])
                if m.any():
                    bands[b].append(need[m].numpy())
    return (np.concatenate(unc), np.concatenate(cov),
            {b: np.concatenate(v) for b, v in bands.items() if v}, float(np.mean(frac_cov)))


def _auroc(score, pos):
    pos = pos.astype(bool)
    if pos.all() or not pos.any():
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def report(tag, u, c, bands, fc, thresh=0.95, verbose=True):
    ok = len(u) > 0 and float(np.median(u)) >= thresh and float(np.mean(u >= thresh)) >= 0.99
    # Is `need` in practice a COVERAGE indicator rather than a depth-uncertainty map? The
    # "ours_nodepth" preset is labelled "magnitude utility + coverage need" on the strength of this.
    score = np.concatenate([u, c])
    is_unc = np.concatenate([np.ones(len(u), bool), np.zeros(len(c), bool)])
    auroc_cov = _auroc(score, is_unc)
    frac_cov_low = float(np.mean(c < 0.01))
    if verbose:
        print(f"  [{tag}] mean covered fraction of the grid: {fc:.3f}")
        print(f"    UNCOVERED cells (n={len(u):8d}): need median {np.median(u):.4f}  mean {u.mean():.4f}  "
              f"min {u.min():.4f}  frac>={thresh}: {np.mean(u >= thresh):.4f}")
        print(f"    COVERED   cells (n={len(c):8d}): need median {np.median(c):.4f}  mean {c.mean():.4f}  "
              f"p05 {np.quantile(c, 0.05):.4f}")
        for b, v in bands.items():
            print(f"      covered {b[0]:>4}-{b[1]:<5} m (n={len(v):7d}): need median {np.median(v):.4f}  mean {v.mean():.4f}")
        print(f"    need as a COVERAGE indicator: AUROC {auroc_cov:.4f}  "
              f"(covered cells with need<0.01: {frac_cov_low:.4f})")
        print(f"    {'PASS' if ok else 'FAIL'}: uncovered cells {'do' if ok else 'DO NOT'} request help "
              f"(need >= {thresh})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="optional checkpoint, to make covered-cell numbers realistic")
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()

    cfg = replace(Config(), device="cpu", augment=False, depth_source="none")
    cfg.validate()
    backbone = LSSBackbone(cfg)
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"]
        bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        backbone.load_state_dict(bb)
        print(f"loaded backbone weights from {a.ckpt}")
    else:
        print("random weights (uncovered-cell result is weight-independent)")
    backbone.eval()

    ds = OPV2VDataset(cfg.val_dir, cfg, max_agents=1, use_gt_pose=cfg.use_gt_pose)
    idxs = np.linspace(0, len(ds) - 1, a.frames).astype(int)

    print("\nreal need map:")
    ok = report("real", *measure(cfg, backbone, ds, idxs))

    # ---- negative control: a plausible broken variant, where an empty column reads as CERTAIN ----
    def broken(depth_logits, eps=1e-9):
        """What the code would do if the splatted column were normalised to zeros and its entropy
        taken directly, instead of going through log()+softmax (which turns a constant into a
        uniform distribution). An all-zero column then has zero entropy = fully certain."""
        p = depth_logits.exp()                       # undo the log: all-zero column stays all-zero
        K = p.size(1)
        logK = torch.log(torch.tensor(float(K)))
        return (-(p * (p + eps).log()).sum(dim=1) / logK).clamp(0.0, 1.0)

    print("\nnegative control (entropy of the un-softmaxed column; empty column reads as certain):")
    bad = report("broken", *measure(cfg, backbone, ds, idxs, uncertainty_fn=broken))

    print(f"\ngate: real={'PASS' if ok else 'FAIL'}  negative-control={'correctly FAILS' if not bad else 'PASSES -- GATE IS USELESS'}")
    return 0 if (ok and not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
