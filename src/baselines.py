"""
Baselines expressed as FLAG PRESETS on the SAME model.

The playbook requires that efficient baselines be re-run on the identical
camera setup, not cited from LiDAR numbers. This module gives one Config
per baseline that changes ONLY the ablation flags; the backbone, head,
data and evaluation are shared.

Presets:
    no_fusion           : ego-only lower bound
    dense               : transmit every covered cell, plain attention (upper bound, bandwidth-blind)
    where2comm_cam      : confidence-only utility + gated fusion + receiver-centric (Where2comm on camera)
    jigsawcomm_cam      : top-1 per cell, magnitude utility, plain attention (JigsawComm on camera)
    ours_full           : depth-aware + top-2 + utility-gated + receiver-centric  (the paper's method)
"""
from __future__ import annotations
from dataclasses import replace
from .config import Config


def make(name: str, base: Config) -> Config:
    if name == "no_fusion":
        return replace(base, exp_name="no_fusion",
                       no_fusion=True, dense_transmit=False,
                       use_encoder=False)
    if name == "dense":
        return replace(base, exp_name="dense",
                       no_fusion=False, dense_transmit=True,
                       use_utility_gate=False, use_encoder=False,
                       receiver_centric=False, utility_type="magnitude_only")
    if name == "where2comm_cam":
        return replace(base, exp_name="where2comm_cam",
                       no_fusion=False, dense_transmit=False,
                       utility_type="confidence_only",
                       use_utility_gate=True, receiver_centric=True,
                       k=1, use_encoder=True)
    if name == "jigsawcomm_cam":
        return replace(base, exp_name="jigsawcomm_cam",
                       no_fusion=False, dense_transmit=False,
                       utility_type="magnitude_only",
                       use_utility_gate=False, receiver_centric=False,
                       k=1, use_encoder=True)
    if name == "ours_nodepth":
        # "No depth term" ablation: identical to ours_full except the sender utility drops
        # exp(-alpha*D) and becomes ||F|| alone. The receiver need is unchanged -- and since D is
        # ~0 in 97% of covered cells and exactly 1 in uncovered ones (measured 2026-11-10), that
        # need is in practice a COVERAGE map, not a depth-uncertainty map. So this preset is
        # "magnitude utility + coverage need". See scripts/check_need_uncovered.py, which gates
        # that reading, and diagnostics/.../bev-depth-entropy-degenerate notes.
        return replace(base, exp_name="ours_nodepth",
                       no_fusion=False, dense_transmit=False,
                       utility_type="magnitude_only",
                       use_utility_gate=True, receiver_centric=True,
                       k=2, use_encoder=True)
    if name == "ours_full":
        return replace(base, exp_name="ours_full",
                       no_fusion=False, dense_transmit=False,
                       utility_type="depth_aware",
                       use_utility_gate=True, receiver_centric=True,
                       k=2, use_encoder=True)
    raise ValueError(f"unknown baseline preset: {name}")


ALL = ["no_fusion", "dense", "where2comm_cam", "jigsawcomm_cam", "ours_full", "ours_nodepth"]
