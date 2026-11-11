"""Gates for the baseline components, on REAL data, each verified to fail on broken code.

    python scripts/check_baseline_components.py [--frames 3]

Covers, per the 2x2 baseline design (sender/receiver x confidence/depth):
  1. need_source="confidence" really uses 1 - ego confidence, NOT the depth-based need, and the two
     differ materially. If they did not, the Where2comm-vs-ours comparison would be contaminated by
     our depth term leaking through the request map.
  2. fusion_type="max" is a true point-wise maximum over the ego and TRANSMITTED collaborator cells,
     with untransmitted cells excluded and the ego as fallback.
  3. utility_type="depth_entropy" equals 1 - D exactly (CoCa3D-style, no feature-magnitude term).
  4. Byte accounting: the differentiable soft_B and metrics.bytes_kb_from_out agree, both include
     the per-sender control map, and both scale with the number of senders.

Every gate is run twice: once on the real code, once on a deliberately broken variant that must
make it FAIL. A gate that cannot fail is worse than no gate (see the stale TEST 10 incident).
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dataclasses import replace
import numpy as np, torch
from src.config import Config
from src.baselines import make
from src.data.opv2v import OPV2VDataset, opv2v_collate
from src.models.coop_detector import CoopDetector
from src.models.gated_fusion import UtilityGatedFusion
from src.models.depth_aware_utility import UtilityHead, depth_uncertainty
from src.metrics import _to_device, bytes_kb_from_out

OK, BAD = "PASS", "FAIL"


def _scenes(cfg, n):
    ds = OPV2VDataset(cfg.val_dir, replace(cfg, augment=False, depth_source="none"),
                      max_agents=cfg.max_agents, use_gt_pose=cfg.use_gt_pose)
    out = []
    for i in np.linspace(0, len(ds) - 1, n * 4).astype(int):
        s = opv2v_collate([ds[int(i)]])
        if int(s["n_agents"]) >= 2:                   # the baselines only differ when fusing
            out.append(_to_device(s, torch.device("cpu")))
        if len(out) == n:
            break
    assert out, "no multi-agent val frames found"
    return out


def gate_need_source(scenes, verbose=True):
    """need_source='confidence' must use ego confidence, not the depth-based need."""
    base = make("ours_full", Config())
    diffs, corr = [], []
    for sc in scenes:
        needs = {}
        for src in ("depth", "confidence"):
            cfg = replace(base, device="cpu", need_source=src)
            torch.manual_seed(0)
            m = CoopDetector(cfg).eval()
            with torch.no_grad():
                F_all, depth_logits, _ = m.backbone(sc)
                U_all, D_all = m.utility(F_all, depth_logits)
                needs[src] = m._request_map(F_all, D_all)   # the REAL code path forward() uses
        a, b = needs["depth"].flatten(), needs["confidence"].flatten()
        diffs.append(float((a - b).abs().mean()))
        if a.std() > 0 and b.std() > 0:
            corr.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    md, mc = float(np.mean(diffs)), float(np.mean(corr)) if corr else float("nan")
    ok = md > 0.05 and abs(mc) < 0.9
    if verbose:
        print(f"  {OK if ok else BAD}  need_source: mean|depth-confidence| = {md:.4f}, "
              f"corr = {mc:.4f}  (want a real difference, |corr| < 0.9)")
    return ok


def gate_max_fusion(verbose=True):
    """fusion_type='max' must be a point-wise max over ego + TRANSMITTED cells only."""
    torch.manual_seed(0)
    C, H, W, N = 4, 3, 3, 2
    fus = UtilityGatedFusion(C=C, fusion_type="max").eval()
    F_ego = torch.randn(C, H, W)
    F_recv = torch.randn(N, C, H, W) + 5.0            # collaborators clearly larger
    mask = torch.zeros(N, H, W)
    mask[0, 0, 0] = 1.0                                # exactly one transmitted cell
    with torch.no_grad():
        out = fus(F_ego, F_recv, torch.zeros(N, H, W), mask)
    want_sent = torch.maximum(F_ego[:, 0, 0], F_recv[0, :, 0, 0])
    sent_ok = torch.allclose(out[:, 0, 0], want_sent)
    others = torch.ones(H, W, dtype=torch.bool); others[0, 0] = False
    untouched_ok = torch.allclose(out[:, others], F_ego[:, others])   # ego is the fallback
    ok = bool(sent_ok and untouched_ok)
    if verbose:
        print(f"  {OK if ok else BAD}  max fusion: transmitted cell = max(ego, collab) {sent_ok}, "
              f"untransmitted cells fall back to ego {untouched_ok}")
    return ok


def gate_depth_entropy(verbose=True):
    """utility_type='depth_entropy' must be exactly 1 - D, with no magnitude term."""
    torch.manual_seed(0)
    C, K, H, W = 8, 16, 4, 4
    head = UtilityHead(C=C, utility_type="depth_entropy").eval()
    F_bev = torch.randn(1, C, H, W) * 7.0              # large magnitudes: must NOT matter
    logits = torch.randn(1, K, H, W)
    with torch.no_grad():
        U, D = head(F_bev, logits)
    exact = torch.allclose(U, 1.0 - D, atol=1e-6)
    with torch.no_grad():
        U2, _ = head(F_bev * 100.0, logits)            # scaling features must not change U
    invariant = torch.allclose(U, U2, atol=1e-6)
    ok = bool(exact and invariant)
    if verbose:
        print(f"  {OK if ok else BAD}  depth_entropy: U == 1-D {exact}, "
              f"invariant to feature magnitude {invariant}")
    return ok


def gate_img_entropy(scenes, verbose=True):
    """The CoCa3D arm must use IMAGE-space entropy, not the degenerate splatted-histogram entropy.

    Calls the production backbone and UtilityHead -- no re-derivation. Checks, against the
    independently measured image-space statistics (median ~0.57, p5 ~0.11, p95 ~0.90):
      * covered cells are NOT degenerate (the BEV-histogram version has D < 0.01 in 97% of them),
      * uncovered cells are exactly 1.0 (maximum entropy), so an unseen cell is never "confident",
      * the median over covered cells is in a plausible band around the image-space median.
    """
    cfg = replace(make("coca3d_cam", Config()), device="cpu")
    meds, frac_low, unc_ok = [], [], []
    for sc in scenes:
        torch.manual_seed(0)
        m = CoopDetector(cfg).eval()
        with torch.no_grad():
            F_all, depth_logits, extra = m.backbone(sc)
            U, D = m.utility(F_all, depth_logits, extra.get("ent_bev"))   # production path
        ego = D[0]
        unc = ego >= 1.0 - 1e-6
        cov = ~unc
        if cov.sum() < 100:
            continue
        meds.append(float(ego[cov].median()))
        frac_low.append(float((ego[cov] < 0.01).float().mean()))
        unc_ok.append(bool(torch.allclose(ego[unc], torch.ones_like(ego[unc]))) if unc.any() else True)
        assert torch.allclose(U, 1.0 - D, atol=1e-6), "img_depth_entropy must be U = 1 - D"
    med, low = float(np.mean(meds)), float(np.mean(frac_low))
    ok = bool(low < 0.25 and 0.2 < med < 0.95 and all(unc_ok))
    if verbose:
        print(f"  {OK if ok else BAD}  img entropy: median over covered cells {med:.3f} "
              f"(image-space reference ~0.57), degenerate fraction (<0.01) {low:.3f} "
              f"(BEV-histogram version: 0.97), uncovered == 1.0 {all(unc_ok)}")
    return ok


def gate_bytes(scenes, verbose=True):
    """soft_B and bytes_kb_from_out must agree, include control bytes, and scale with senders."""
    cfg = replace(make("ours_full", Config()), device="cpu")
    sc = scenes[0]
    torch.manual_seed(0)
    m = CoopDetector(cfg).eval()
    with torch.no_grad():
        out = m(sc, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)
    kb = bytes_kb_from_out(out, cfg)
    agree = abs(float(out["soft_B"]) / 1024.0 - kb) < 1e-3
    n_send = int(out["n_senders"])
    ctrl = cfg.grid_H * cfg.grid_W * cfg.b_util * n_send
    has_ctrl = kb * 1024.0 > out["n_cells"] * cfg.per_cell_bytes + ctrl * 0.99
    cfg_off = replace(cfg, count_control_bytes=False)
    with torch.no_grad():
        out_off = m(sc, hard=True, gumbel_T=cfg.gumbel_T1, tau=cfg.tau, budget_frac=cfg.budget_frac)
    drops = bytes_kb_from_out(out_off, cfg_off) < kb
    ok = bool(agree and has_ctrl and drops and n_send >= 1)
    if verbose:
        print(f"  {OK if ok else BAD}  bytes: soft_B/1024 == bytes_kb {agree} "
              f"({float(out['soft_B'])/1024:.2f} vs {kb:.2f} KB), control map for {n_send} sender(s) "
              f"included {has_ctrl}, removable {drops}")
    return ok


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--frames", type=int, default=3)
    a = ap.parse_args()
    scenes = _scenes(replace(Config(), device="cpu"), a.frames)
    print(f"real data: {len(scenes)} multi-agent val frames\n[real code]")
    real = {"need_source": gate_need_source(scenes), "max fusion": gate_max_fusion(),
            "depth_entropy": gate_depth_entropy(), "img_entropy": gate_img_entropy(scenes),
            "bytes": gate_bytes(scenes)}

    # ---------------- negative controls: each must FAIL ----------------
    print("\n[negative controls -- each gate must FAIL on broken code]")
    import src.models.depth_aware_utility as DAU
    import src.models.gated_fusion as GF
    import src.metrics as MET
    broken = {}

    orig_fwd = DAU.UtilityHead.forward
    def bad_entropy(self, F_bev, depth_logits):
        U, D = orig_fwd(self, F_bev, depth_logits)
        return (U * F_bev.norm(dim=1) if self.utility_type == "depth_entropy" else U), D
    DAU.UtilityHead.forward = bad_entropy
    broken["depth_entropy"] = gate_depth_entropy(verbose=False)
    DAU.UtilityHead.forward = orig_fwd

    orig_gf = GF.UtilityGatedFusion.forward
    def bad_max(self, F_ego, F_recv, U_r, mask):
        if self.fusion_type == "max":
            return torch.cat([F_ego.unsqueeze(0), F_recv], 0).max(0).values   # ignores the mask
        return orig_gf(self, F_ego, F_recv, U_r, mask)
    GF.UtilityGatedFusion.forward = bad_max
    broken["max fusion"] = gate_max_fusion(verbose=False)
    GF.UtilityGatedFusion.forward = orig_gf

    orig_b = MET.bytes_kb_from_out
    MET.bytes_kb_from_out = lambda out, cfg: out.get("n_cells", 0) * cfg.per_cell_bytes / 1024.0
    globals()["bytes_kb_from_out"] = MET.bytes_kb_from_out
    broken["bytes"] = gate_bytes(scenes, verbose=False)
    MET.bytes_kb_from_out = orig_b
    globals()["bytes_kb_from_out"] = orig_b

    # need_source: break it so "confidence" silently falls back to the DEPTH need -- exactly the
    # contamination this option exists to remove. Patching the method forward() actually calls.
    _orig_req = CoopDetector._request_map
    CoopDetector._request_map = lambda self, F_all, D_all: 1.0 - self._ego_visibility(D_all[0])
    broken["need_source"] = gate_need_source(scenes, verbose=False)
    CoopDetector._request_map = _orig_req

    # img entropy: break it so the arm falls back to the DEGENERATE splatted-histogram entropy --
    # exactly the strawman this implementation exists to avoid. Patch the production path.
    import src.models.lss_backbone as _LB
    _orig_bb = _LB.LSSBackbone.forward
    def _bad_bb(self, scene):
        F_out, dl, extra = _orig_bb(self, scene)
        from src.models.depth_aware_utility import depth_uncertainty
        extra["ent_bev"] = depth_uncertainty(dl)        # BEV-histogram entropy, the degenerate one
        return F_out, dl, extra
    _LB.LSSBackbone.forward = _bad_bb
    broken["img_entropy"] = gate_img_entropy(scenes, verbose=False)
    _LB.LSSBackbone.forward = _orig_bb

    for k in ("need_source", "depth_entropy", "img_entropy", "max fusion", "bytes"):
        print(f"  broken {k:>14}: {'correctly FAILS' if not broken[k] else 'STILL PASSES -- gate is useless'}")

    all_ok = all(real.values()) and not any(broken.values())
    print(f"\nreal: {real}")
    print(f"overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
