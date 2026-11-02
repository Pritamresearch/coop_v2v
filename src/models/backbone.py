"""
Backbone selector — always returns LSSBackbone for OPV2V real data.
"""
from __future__ import annotations


def build_backbone(cfg):
    if cfg.backbone == "lss":
        from .lss_backbone import LSSBackbone
        return LSSBackbone(cfg)
    raise ValueError(f"Unknown backbone: {cfg.backbone!r}. Only 'lss' is supported.")
