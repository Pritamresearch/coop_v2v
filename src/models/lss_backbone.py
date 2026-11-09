"""
Lift-Splat-Shoot backbone — production implementation
======================================================
Philion & Fidler, ECCV 2020. Adapted for depth-aware V2V co-design.

Image encoder (cfg.lss_encoder):
    'efficientnet'  EfficientNet-B0, no pretrained weights needed
    'resnet50'      ResNet-50
    'small'         Lightweight CNN for CPU debugging only

VERIFIED EfficientNet-B0 output sizes for input 256×704:
    features[0]: (32, 128, 352)   stride ×2
    features[1]: (16, 128, 352)
    features[2]: (24,  64, 176)   stride ×4
    features[3]: (40,  32,  88)   stride ×8   <- use this for downsample=8
    features[4]: (80,  16,  44)   stride ×16  <- use this for downsample=16

So for cfg.lss_downsample=8:
    trunk = features[:4]   trunk_ch = 40
    fH = 256/8  = 32
    fW = 704/8  = 88

For cfg.lss_downsample=16:
    trunk = features[:5]   trunk_ch = 80
    fH = 256/16 = 16
    fW = 704/16 = 44

Input scene dict keys used:
    images      (N, n_cams, 3, H, W)   float32 ImageNet-normalised
    intrins     (N, n_cams, 3, 3)
    rots        (N, n_cams, 3, 3)   cam -> LiDAR rotation
    trans       (N, n_cams, 3)      cam origin in LiDAR frame (m)
    post_rots   (N, n_cams, 3, 3) | None
    post_trans  (N, n_cams, 3)    | None

Returns:
    F              (N, C, grid_H, grid_W)
    depth_logits   (N, K, grid_H, grid_W)   log-probs (BEV splatted)
    extra          {"depth_img": (N, n_cams, K, fH, fW)}
"""
from __future__ import annotations
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

class _ConvBnRelu(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride=s, padding=p, bias=False),
            nn.GroupNorm(min(32, cout), cout),
            nn.ReLU(inplace=True),
        )


# ------------------------------------------------------------------ #
# Image encoders                                                       #
# ------------------------------------------------------------------ #

class SmallEncoder(nn.Module):
    """
    Lightweight CNN for CPU debugging only.
    downsample=8  -> 3 stride-2 blocks
    downsample=16 -> 4 stride-2 blocks
    """
    def __init__(self, K: int, C: int, downsample: int = 8):
        super().__init__()
        assert downsample in (8, 16)
        n_down = int(math.log2(downsample))   # 3 or 4
        chs = [32, 64, 128, 256][:n_down]
        layers, cin = [], 3
        for cout in chs:
            layers += [_ConvBnRelu(cin, cout, s=2), _ConvBnRelu(cout, cout)]
            cin = cout
        self.trunk = nn.Sequential(*layers)
        self.head  = nn.Conv2d(cin, K + C, 1)
        self.K = K

    def forward(self, x):
        h = self.head(self.trunk(x))
        return h[:, :self.K], h[:, self.K:]


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B0 image encoder.

    Verified output for 256x704 input:
        downsample=8  -> features[:4], trunk_ch=40, fH=32,  fW=88
        downsample=16 -> features[:5], trunk_ch=80, fH=16,  fW=44
    """
    _SPLIT = {8: (4, 40), 16: (5, 80)}

    def __init__(self, K: int, C: int, downsample: int = 8,
                 pretrained: bool = False):
        super().__init__()
        assert downsample in self._SPLIT, \
            f"EfficientNetEncoder: downsample must be 8 or 16, got {downsample}"

        n_blocks, trunk_ch = self._SPLIT[downsample]

        # Load ImageNet-pretrained weights from the local torch cache
        # (~/.cache/torch/hub/checkpoints/) when pretrained=True -- no
        # internet needed on compute nodes as long as the cache is already
        # populated (shared $HOME across login/compute nodes on this cluster).
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        _weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        eff = efficientnet_b0(weights=_weights)
        trunk = nn.Sequential(*list(eff.features)[:n_blocks])
        # torch.compile gives ~20-30% speedup on V100 (PyTorch 2.x)
        try:
            self.trunk = torch.compile(trunk)
        except Exception:
            self.trunk = trunk
        self.head  = nn.Conv2d(trunk_ch, K + C, 1)
        self.K = K

    def forward(self, x):
        h = self.head(self.trunk(x))
        return h[:, :self.K], h[:, self.K:]


class ResNet50Encoder(nn.Module):
    """
    ResNet-50 image encoder.
    layer2 -> ×8  downsampling, ch=512
    layer3 -> ×16 downsampling, ch=1024
    """
    _SPLIT = {8: 2, 16: 3}

    def __init__(self, K: int, C: int, downsample: int = 8,
                 pretrained: bool = False):
        super().__init__()
        assert downsample in self._SPLIT

        from torchvision.models import resnet50
        rn = resnet50(weights=None)
        n_layers = self._SPLIT[downsample]
        trunk_ch = {2: 512, 3: 1024}[n_layers]

        blocks = [rn.conv1, rn.bn1, rn.relu, rn.maxpool,
                  rn.layer1, rn.layer2]
        if n_layers == 3:
            blocks.append(rn.layer3)
        self.trunk = nn.Sequential(*blocks)
        self.head  = nn.Conv2d(trunk_ch, K + C, 1)
        self.K = K

    def forward(self, x):
        h = self.head(self.trunk(x))
        return h[:, :self.K], h[:, self.K:]


def build_encoder(K: int, C: int, cfg) -> nn.Module:
    enc = getattr(cfg, "lss_encoder", "small")
    ds  = cfg.lss_downsample
    pre = getattr(cfg, "lss_pretrained", False)   # default False = no internet needed
    if enc == "efficientnet":
        return EfficientNetEncoder(K, C, downsample=ds, pretrained=pre)
    if enc == "resnet50":
        return ResNet50Encoder(K, C, downsample=ds, pretrained=pre)
    return SmallEncoder(K, C, downsample=ds)


# ------------------------------------------------------------------ #
# Frustum                                                              #
# ------------------------------------------------------------------ #

def create_frustum(fH: int, fW: int, H_img: int, W_img: int,
                   d_min: float, d_max: float, K: int,
                   spacing: str = "linear") -> torch.Tensor:
    """
    Build (K, fH, fW, 3) frustum.
    Each point is (pixel_u, pixel_v, depth_metres) in ORIGINAL image coords.
    fH, fW are the ACTUAL encoder output spatial dims (not H//downsample).
    spacing: "linear" or "log" depth bins (see src/depth_bins.py).
    """
    if spacing == "log":
        from src.depth_bins import depth_bin_points
        d = torch.tensor(depth_bin_points(d_min, d_max, K, spacing), dtype=torch.float32)
    else:
        d = torch.linspace(d_min, d_max, K)          # unchanged: bit-identical to earlier runs
    depths = d.view(K, 1, 1).expand(K, fH, fW)
    us     = torch.linspace(0, W_img - 1, fW).view(1, 1, fW).expand(K, fH, fW)
    vs     = torch.linspace(0, H_img - 1, fH).view(1, fH, 1).expand(K, fH, fW)
    return torch.stack((us, vs, depths), dim=-1)   # (K, fH, fW, 3)


# ------------------------------------------------------------------ #
# Geometry: frustum → ego 3-D                                         #
# ------------------------------------------------------------------ #

def frustum_to_ego(
    frustum:   torch.Tensor,          # (K, fH, fW, 3)
    rots:      torch.Tensor,          # (n_cams, 3, 3)  cam->ego
    trans:     torch.Tensor,          # (n_cams, 3)
    intrins:   torch.Tensor,          # (n_cams, 3, 3)
    post_rots:  Optional[torch.Tensor],
    post_trans: Optional[torch.Tensor],
    depth_scale: float = 1.0,         # BEV scale augmentation (see src/data/bev_augment.py)
) -> torch.Tensor:                    # (n_cams, K, fH, fW, 3)
    n_cams = rots.shape[0]
    pts = frustum.unsqueeze(0).expand(n_cams, -1, -1, -1, -1).clone()

    # A true scene scale needs the frustum depths scaled as well as the translations: K^-1 applied
    # to (u*d, v*d, d) is linear in d, so scaling d scales the camera-frame point exactly.
    if depth_scale != 1.0:
        pts[..., 2] = pts[..., 2] * depth_scale

    # undo image augmentation
    if post_rots is not None and post_trans is not None:
        pts = pts - post_trans.view(n_cams, 1, 1, 1, 3)
        pts = torch.einsum("nij,nkhwj->nkhwi",
                           torch.inverse(post_rots), pts)

    # un-project via K^{-1}: (u*d, v*d, d) -> cam 3D (standard-pinhole axes:
    # X=right, Y=down, Z=forward/depth -- NOT this dataset's real convention,
    # see permutation below)
    d       = pts[..., 2:3]
    uv1     = torch.cat([pts[..., :2] * d, d], dim=-1)
    K_inv   = torch.inverse(intrins)
    pts_cam_std = torch.einsum("nij,nkhwj->nkhwi", K_inv, uv1)

    # Permute into this dataset's REAL camera-local convention (confirmed
    # empirically against real OPV2V data, 2026-10-27, n=791 GT boxes:
    # CARLA-native, x=forward/depth, y=right, z=up -- matches
    # _check_oof/_build_depth_gt in src/data/opv2v.py). The un-projection
    # above is algebraically correct for a standard pinhole camera; this
    # dataset's rots/trans/intrins just don't use that axis layout.
    #   new_x (depth) = std Z
    #   new_y (right) = std X
    #   new_z (up)    = -std Y
    pts_cam = torch.stack(
        [pts_cam_std[..., 2], pts_cam_std[..., 0], -pts_cam_std[..., 1]],
        dim=-1,
    )

    # cam -> ego
    pts_ego = (torch.einsum("nij,nkhwj->nkhwi", rots, pts_cam)
               + trans.view(n_cams, 1, 1, 1, 3))
    return pts_ego


# ------------------------------------------------------------------ #
# BEV scatter-add                                                      #
# ------------------------------------------------------------------ #

def bev_pool(feats: torch.Tensor,   # (M, C)
             coords: torch.Tensor,  # (M, 2) long (row, col)
             H: int, W: int) -> torch.Tensor:
    C   = feats.shape[1]
    bev = feats.new_zeros(C, H * W)
    r, c = coords[:, 0], coords[:, 1]
    ok   = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    if ok.sum() == 0:
        return bev.view(C, H, W)
    idx = (r[ok] * W + c[ok]).long()
    bev.index_add_(1, idx, feats[ok].t())
    return bev.view(C, H, W)


# ------------------------------------------------------------------ #
# LSS Backbone                                                         #
# ------------------------------------------------------------------ #

class LSSBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enc = build_encoder(cfg.depth_K, cfg.feat_C, cfg)

        self.bev_refine = nn.Sequential(
            nn.Conv2d(cfg.feat_C, cfg.feat_C, 3, padding=1, bias=False),
            nn.GroupNorm(32, cfg.feat_C),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.feat_C, cfg.feat_C, 3, padding=1, bias=False),
            nn.GroupNorm(32, cfg.feat_C),
            nn.ReLU(inplace=True),
        )

        # Compute ACTUAL encoder output spatial size by dry-running the encoder
        self._fH, self._fW = self._probe_encoder_size()

        frustum = create_frustum(
            self._fH, self._fW,
            cfg.lss_img_H, cfg.lss_img_W,
            cfg.lss_d_min, cfg.lss_d_max,
            cfg.depth_K,
            getattr(cfg, "depth_bins", "linear"),
        )
        self.register_buffer("frustum", frustum, persistent=False)
        print(f"[LSSBackbone] encoder={cfg.lss_encoder}  "
              f"input={cfg.lss_img_H}x{cfg.lss_img_W}  "
              f"feature_map={self._fH}x{self._fW}  "
              f"downsample=x{cfg.lss_img_H // self._fH}")

    def _probe_encoder_size(self) -> Tuple[int, int]:
        """Dry-run encoder on a dummy image to get actual fH, fW."""
        cfg = self.cfg
        dummy = torch.zeros(1, 3, cfg.lss_img_H, cfg.lss_img_W)
        with torch.no_grad():
            _, ctx = self.enc(dummy)
        return ctx.shape[2], ctx.shape[3]

    def _bev_coords(self, pts_ego: torch.Tensor):
        cfg = self.cfg
        x   = pts_ego[..., 0]
        y   = pts_ego[..., 1]
        col = ((x - cfg.bev_x_min) / cfg.cell_m).long()
        row = ((y - cfg.bev_y_min) / cfg.cell_m).long()
        valid = (col >= 0) & (col < cfg.grid_W) & \
                (row >= 0) & (row < cfg.grid_H)
        return row, col, valid

    def _forward_agent(
        self,
        images:     torch.Tensor,    # (n_cams, 3, H, W)
        intrins:    torch.Tensor,
        rots:       torch.Tensor,
        trans:      torch.Tensor,
        post_rots:  Optional[torch.Tensor],
        post_trans: Optional[torch.Tensor],
        depth_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg

        # ---- encode ----
        depth_logits_img, context = self.enc(images)
        # depth_logits_img : (n_cams, K, fH, fW)
        # context          : (n_cams, C, fH, fW)
        depth_prob = F.softmax(depth_logits_img, dim=1)

        # ---- lift: outer product -> (n_cams, C, K, fH, fW) ----
        feats_5d = context.unsqueeze(2) * depth_prob.unsqueeze(1)

        # ---- all dims from ACTUAL tensor shapes ----
        n_cams_t, C_t, K_t, fH_t, fW_t = feats_5d.shape
        M = n_cams_t * K_t * fH_t * fW_t

        # ---- frustum geometry ----
        pts_ego = frustum_to_ego(
            self.frustum, rots, trans, intrins, post_rots, post_trans,
            depth_scale=depth_scale,
        )
        row, col, valid = self._bev_coords(pts_ego)

        # ---- splat features ----
        feats_flat = feats_5d.permute(0, 2, 3, 4, 1).reshape(M, C_t)
        rc_flat    = torch.stack([row.reshape(M), col.reshape(M)], dim=-1)
        valid_flat = valid.reshape(M)

        F_agent = bev_pool(feats_flat[valid_flat], rc_flat[valid_flat],
                           cfg.grid_H, cfg.grid_W)
        F_agent = self.bev_refine(F_agent.unsqueeze(0)).squeeze(0)

        # ---- splat depth distribution ----
        k_idx = (torch.arange(K_t, device=images.device)
                       .view(1, K_t, 1, 1)
                       .expand(n_cams_t, K_t, fH_t, fW_t)
                       .reshape(M))
        prob_flat   = depth_prob.reshape(M)
        eye_K       = torch.eye(K_t, device=images.device, dtype=images.dtype)
        depth_feats = eye_K[k_idx] * prob_flat.unsqueeze(-1)   # (M, K)

        depth_bev = bev_pool(depth_feats[valid_flat], rc_flat[valid_flat],
                             cfg.grid_H, cfg.grid_W)            # (K, H, W)
        depth_bev = depth_bev / (depth_bev.sum(0, keepdim=True) + 1e-6)

        return F_agent, depth_bev, depth_logits_img

    def forward(self, scene: dict):
        images     = scene["images"]
        intrins    = scene["intrins"]
        rots       = scene["rots"]
        trans      = scene["trans"]
        post_rots  = scene.get("post_rots")
        post_trans = scene.get("post_trans")

        N = images.shape[0]
        F_list, depth_list, dimg_list = [], [], []

        for i in range(N):
            pr = post_rots[i]  if post_rots  is not None else None
            pt = post_trans[i] if post_trans is not None else None
            Fi, Di, dimg = self._forward_agent(
                images[i], intrins[i], rots[i], trans[i], pr, pt,
                depth_scale=float(scene.get("depth_scale", 1.0)),
            )
            F_list.append(Fi)
            depth_list.append(Di)
            dimg_list.append(dimg)

        F_out            = torch.stack(F_list,     dim=0)
        depth_bev        = torch.stack(depth_list, dim=0)
        depth_logits_bev = torch.log(depth_bev + 1e-9)
        depth_img        = torch.stack(dimg_list,  dim=0)

        return F_out, depth_logits_bev, {"depth_img": depth_img}