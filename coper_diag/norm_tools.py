"""BatchNorm diagnostics and fixes for batch-size-1 training.

What is actually at stake with batch 1 (agents x 4 cameras):
  * The image backbone sees 4 images per step, so its BN batch statistics are
    computed over 4 images: noisy but not degenerate.
  * BEV encoder / head BN sees ONE sample per step; statistics come only from
    spatial positions of that one BEV map (80x80). Running statistics then
    average many single-scene estimates, and eval-mode (running stats) can
    differ systematically from train-mode (batch stats).
  * This is an eval-path mismatch, not overfitting. It is testable WITHOUT
    retraining: evaluate the same checkpoint (a) in eval mode, (b) with BN
    using per-sample batch stats, (c) after re-estimating running stats.
    If (b) or (c) >> (a), BN is a real cause. If all equal, rule it out.

Fixes (each a single, reportable change): GroupNorm in BEV encoder + head,
frozen BN in the pretrained image backbone, or larger true batch. Gradient
accumulation does NOT fix BN statistics.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


@contextlib.contextmanager
def bn_use_batch_stats(model: nn.Module):
    """Evaluate with BN in train mode (batch stats) without touching running
    stats: momentum=0 makes the running update a no-op. Dropout stays off."""
    saved = []
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            saved.append((m, m.training, m.momentum))
            m.train()
            m.momentum = 0.0
    try:
        yield model
    finally:
        for m, tr, mom in saved:
            m.train(tr)
            m.momentum = mom


@torch.no_grad()
def recalibrate_bn(model: nn.Module, batches, forward_fn, n_batches=500):
    """'Precise BN': reset running stats and re-estimate them as a true
    cumulative average over n_batches training batches (no augmentation),
    then return the model in eval mode. forward_fn(model, batch) runs a
    forward pass. Use a COPY of the checkpoint."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average
    model.eval()
    for m in bns:
        m.train()
    for i, b in enumerate(batches):
        if i >= n_batches:
            break
        forward_fn(model, b)
    model.eval()
    return model


def bn_to_gn(module: nn.Module, num_groups=32, skip=lambda name: False) -> nn.Module:
    """Replace BatchNorm2d with GroupNorm (affine weights copied). Use on the
    BEV encoder and detection head, which are trained from scratch. For the
    pretrained image backbone prefer `freeze_bn` (keeps ImageNet statistics).
    Must be applied BEFORE building the optimizer."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d) and not skip(name):
            C = child.num_features
            g = num_groups
            while C % g:
                g //= 2
            gn = nn.GroupNorm(g, C, affine=child.affine)
            if child.affine:
                gn.weight.data.copy_(child.weight.data)
                gn.bias.data.copy_(child.bias.data)
            setattr(module, name, gn)
        else:
            bn_to_gn(child, num_groups, skip)
    return module


def freeze_bn(module: nn.Module) -> nn.Module:
    """Keep BN in eval mode forever (fixed running stats, affine still
    trainable). Call after every model.train(), or wrap train() as below."""
    bns = [m for m in module.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    orig_train = module.train

    def train(mode=True):
        orig_train(mode)
        for m in bns:
            m.eval()
        return module

    module.train = train
    module.train(module.training)
    return module


def count_bn(module: nn.Module) -> int:
    return sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in module.modules())
