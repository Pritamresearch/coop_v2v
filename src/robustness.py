"""
Robustness protocols.

    * pose noise    : Gaussian on collaborator rel_pose (sigma 0 -> 0.6 m)
    * latency       : drop a collaborator's contribution with prob delay/500 ms
                      (models "message arrived too late to use this frame")
    * packet_loss   : drop a collaborator's contribution with an EXPLICIT
                      probability swept directly (0 -> 0.5), independent of
                      any latency interpretation -- models outright packet loss
                      on an unreliable V2V channel (as in JigsawComm Fig. 8/App. E)
    * agent count   : cap number of collaborators fused (2 -> max_agents)

Applied to the SAME trained model at eval time, so the outputs isolate
robustness from training effects.
"""
from __future__ import annotations
import csv
import os
import numpy as np
import torch

from .config import Config
from .metrics import evaluate, _to_device, Evaluator, decode, bytes_kb_from_out


def _perturb_pose(scene, sigma_m: float, sigma_yaw: float, rng: np.random.Generator):
    """Add Gaussian noise to collaborator rel_pose entries (row 0 = ego stays 0)."""
    scene = dict(scene)
    rp = scene["rel_pose"].clone()
    N = rp.shape[0]
    noise = torch.zeros_like(rp)
    if N > 1:
        noise[1:, 0] = torch.from_numpy(rng.normal(0.0, sigma_m,   N - 1).astype(np.float32))
        noise[1:, 1] = torch.from_numpy(rng.normal(0.0, sigma_m,   N - 1).astype(np.float32))
        noise[1:, 2] = torch.from_numpy(rng.normal(0.0, sigma_yaw, N - 1).astype(np.float32))
    scene["rel_pose"] = rp + noise.to(rp.device)
    return scene


def _drop_agents(scene, drop_prob: float, rng: np.random.Generator):
    """Simulate staleness/loss: randomly drop collaborators' images to zero."""
    scene = dict(scene)
    imgs = scene["images"].clone()
    N = imgs.shape[0]
    if N > 1 and drop_prob > 0:
        for i in range(1, N):
            if rng.random() < drop_prob:
                imgs[i].zero_()
    scene["images"] = imgs
    return scene


def _cap_agents(scene, max_collab: int):
    """Keep ego + first max_collab collaborators."""
    scene = dict(scene)
    N = scene["images"].shape[0]
    keep = min(N, max_collab + 1)
    for k in ("images", "intrins", "rots", "trans", "rel_pose"):
        scene[k] = scene[k][:keep]
    if scene.get("post_rots") is not None:
        scene["post_rots"] = scene["post_rots"][:keep]
    if scene.get("post_trans") is not None:
        scene["post_trans"] = scene["post_trans"][:keep]
    if "depth_gt_img" in scene:
        scene["depth_gt_img"] = scene["depth_gt_img"][:keep]
    scene["n_agents"] = keep
    return scene


@torch.no_grad()
def _eval_perturbed(model, loader, cfg, perturb_fn):
    model.eval()
    ev = Evaluator(cfg)
    device = next(model.parameters()).device
    for scene in loader:
        scene = _to_device(scene, device)
        scene_p = perturb_fn(scene)
        out = model(scene_p, hard=True, tau=cfg.tau, budget_frac=cfg.budget_frac)
        boxes, scores = decode(out, cfg)
        ev.update(boxes, scores,
                  scene["gt"]["boxes"].cpu().numpy(),
                  scene["gt"]["oof"].cpu().numpy(),
                  kb=bytes_kb_from_out(out, cfg))
    return ev.compute()


def run_robustness(cfg: Config, model, loader):
    out_dir = f"{cfg.root}/robustness"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{cfg.exp_name}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["axis", "x", "ap30", "ap50", "ap70", "oof_recall", "kb"])

        rng = np.random.default_rng(cfg.seed + 7)

        # ---- pose noise sigma sweep ----
        for sigma in [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]:
            r = _eval_perturbed(model, loader, cfg,
                                lambda s, sig=sigma: _perturb_pose(s, sig, math.radians(sig * 5.0), rng))
            w.writerow(["pose", sigma, r["ap30"], r["ap50"], r["ap70"], r["oof_recall"], r["kb"]])
            print(f"[robustness pose sigma={sigma:.2f}] AP70={r['ap70']:.3f}  OOF={r['oof_recall']:.3f}")

        # ---- latency (delay -> drop prob) ----
        for delay_ms in [0, 100, 200, 300, 500]:
            p = delay_ms / 500.0
            r = _eval_perturbed(model, loader, cfg,
                                lambda s, prob=p: _drop_agents(s, prob, rng))
            w.writerow(["latency", delay_ms, r["ap30"], r["ap50"], r["ap70"], r["oof_recall"], r["kb"]])
            print(f"[robustness latency delay={delay_ms}ms] AP70={r['ap70']:.3f}  OOF={r['oof_recall']:.3f}")

        # ---- packet loss (explicit probability, independent of any latency framing) ----
        for p_loss in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            r = _eval_perturbed(model, loader, cfg,
                                lambda s, prob=p_loss: _drop_agents(s, prob, rng))
            w.writerow(["packet_loss", p_loss, r["ap30"], r["ap50"], r["ap70"], r["oof_recall"], r["kb"]])
            print(f"[robustness packet_loss p={p_loss:.2f}] AP70={r['ap70']:.3f}  OOF={r['oof_recall']:.3f}")

        # ---- agent count ----
        for k in range(1, cfg.max_agents + 1):
            r = _eval_perturbed(model, loader, cfg,
                                lambda s, kk=k: _cap_agents(s, kk))
            w.writerow(["agents", k, r["ap30"], r["ap50"], r["ap70"], r["oof_recall"], r["kb"]])
            print(f"[robustness agents={k+1}] AP70={r['ap70']:.3f}  OOF={r['oof_recall']:.3f}")
    return out_path


import math  # placed at end because used only in run_robustness
