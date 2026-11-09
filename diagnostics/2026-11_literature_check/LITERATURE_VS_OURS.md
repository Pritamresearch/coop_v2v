# Literature and reference-implementation check (2026-11-08)

Done before writing the image-augmentation code, to copy proven parameter ranges rather than invent them.
Every row below is sourced. **Nothing here has been applied to the code** — any change must still pass our
real-data checks (`roundtrip_test.py`, `e2e_geometry_integration_test.py`, `scripts/check_collab_warp.py`).

Sources: BEVDet [paper](https://arxiv.org/abs/2112.11790), [repo](https://github.com/HuangJunJie2017/BEVDet) ·
BEVDepth [paper](https://arxiv.org/abs/2206.10092), [repo](https://github.com/Megvii-BaseDetection/BEVDepth) ·
CoBEVT [paper](https://arxiv.org/abs/2207.02202), [repo](https://github.com/DerrickXuNu/CoBEVT) ·
[OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) · CoCa3D [paper](https://arxiv.org/abs/2303.13560),
[repo](https://github.com/MediaBrain-SJTU/CoCa3D) · [OPV2V paper](https://arxiv.org/abs/2109.07644)

## 1. Image-view augmentation (BEVDet `configs/bevdet/bevdet-r50.py`)

| Setting | BEVDet | Ours |
|---|---|---|
| `resize` | `(-0.06, 0.11)` added to the base scale | none |
| `rot` | `(-5.4, 5.4)` degrees | none |
| `flip` | `True` (50% horizontal) | none |
| `crop_h` | `(0.0, 0.0)` → fixed bottom-aligned vertical crop; horizontal crop random | none |
| Photometric / colour jitter | **none** — only `mmlabNormalize` | none |
| Test-time | resize only, centre crop, no flip/rot | n/a |

**How consistency is kept:** the camera intrinsics are *not* modified. The whole 2D augmentation lives in a
2×2 `post_rot` and 2-vector `post_tran` mapping original pixels to augmented pixels, composed in the order
**resize → crop → flip → rotate**. The view transformer applies the inverse so BEV features do not move
(paper Eq. 1–2).

**Our code already has this hook:** [`frustum_to_ego`](../../src/models/lss_backbone.py#L212) undoes
`post_rots`/`post_trans` when they are not `None`, and the loader currently always passes `None`. So
image augmentation can be added without touching intrinsics or geometry.

⚠️ **Our plan included photometric jitter; BEVDet does not use any.** Worth treating as a separate arm
rather than bundling it, so a geometric-only result stays interpretable.

## 2. BEV-view augmentation

| Setting | BEVDet (`bda_aug_conf`) | Ours |
|---|---|---|
| Rotation | **±22.5°** | **±180°** |
| Scale | 0.95–1.05 | none |
| Flip x / y | 0.5 / 0.5 | none |

BEVDet applies these to the view-transformer output *and* the detection targets together. The paper argues
BEV-space augmentation is the one that matters once a BEV encoder is present, and warns that image-view
augmentation alone "even degrades the performance" in that setting.

**Relevant to us:** our BEV rotation is 8× wider than BEVDet's, and we have neither scale nor flip. Adding
scale+flip is cheaper than image augmentation and is the configuration BEVDet actually validated.

OpenCOOD [issue #31](https://github.com/DerrickXuNu/OpenCOOD/issues/31): the maintainer says augmentation is
normally done at the point-cloud level and is "hard to achieve" for intermediate fusion, because GT boxes
must move with the data. That is exactly the class of bug we hit (commit `a624570`), and upstream appears
to avoid BEV augmentation for intermediate fusion altogether.

## 3. Depth supervision and discretisation

| | BEVDepth | CoCa3D (OPV2V+) | Ours |
|---|---|---|---|
| Range / bins | 2–58 m, 112 bins | 0–280 m, 50 bins | 4–90 m, 128 bins |
| Spacing | **uniform** | linear-increasing (LID)¹ | uniform (arm A tried log) |
| Loss | **binary** cross-entropy per bin, weight **3.0** | CaDDN categorical depth loss | softmax CE, weight 1.0 |
| Multiple returns per cell | min-pool over 16×16 → nearest | not stated | nearest (matches) |
| Camera-aware depth net | **yes** — fx, fy, cx, cy + augmentation + sensor-to-ego through MLP + SE gate | no | **no** |
| Far-range re-weighting | **none** | none | arm B (`inv_sqrt_freq`) |

¹ CoCa3D §4.1 says linear-increasing, §6.2 says uniform; the paper contradicts itself.

**Strong external support for our depth finding:** BEVDepth Table 1 — swapping learned depth for
ground-truth depth moves mAP **0.282 → 0.470** and mATE 0.768 → 0.393 on nuScenes. Depth is the dominant
error source in LSS-style detectors, which matches what we measured.

**Our arm B re-weighting is not something any of these papers do.** It is a defensible departure (it
doubled far-range depth mass near truth in our A/B), but it is ours, not standard practice.

**Not done by us, done by BEVDepth:** feeding camera parameters into the depth net. With 4 cameras of
identical intrinsics per agent this matters less for us than for nuScenes, but the extrinsics differ per
camera, so it may still help.

## 4. Cooperative camera methods

- **CoBEVT does not use lift-splat at all** — cross-attention from image to BEV with camera-aware positional
  encoding, ResNet-34, 100 m × 100 m at 0.39 m. Its OPV2V camera track reports **BEV segmentation IoU, not
  detection AP**, so its numbers are not comparable to ours.
- **CoCa3D shares *depth*, not just BEV features.** Each agent transmits a selected depth distribution plus
  voxel features (gated by per-pixel depth entropy); the receiver scores multi-view consistency and refines
  its own depth. This directly attacks the problem we have — a car far from the ego is near a collaborator,
  where depth is reliable. **Our architecture shares BEV features only.**
  - CoCa3D ablation (Table 2, OPV2V+): Co-FL feature sharing alone AP@0.5 **0.7191**; with ground-truth depth
    **0.8805**; their collaborative depth gets **0.8260**. So most of the remaining headroom was depth.
- Neither CoBEVT nor CoCa3D describes any image or BEV augmentation.
- CoBEVT [removes scenario `2021_09_09_13_20_58`](https://github.com/DerrickXuNu/CoBEVT/tree/main/opv2v) for
  buggy camera data. **Checked: that scenario is not in our copy of the dataset**, so nothing to exclude.
- OpenCOOD issues [#33](https://github.com/DerrickXuNu/OpenCOOD/issues/33),
  [#67](https://github.com/DerrickXuNu/OpenCOOD/issues/67): OPV2V LiDAR coordinates are **left-handed**, not
  the OpenPCDet right-hand convention. Our camera convention was verified empirically against real data, so
  this is a note, not an action.

## 5. Benchmark numbers and evaluation protocol

**There is no published camera-only detection AP on the original OPV2V.** The OPV2V paper benchmarks LiDAR
only; CoBEVT's camera track is segmentation IoU; CoCa3D's camera AP is on **OPV2V+**, an extended variant.

| Protocol | Range | Metric |
|---|---|---|
| OPV2V official (LiDAR) | x ∈ [−140, 140], y ∈ [−40, 40] m | AP@IoU 0.5 / 0.7, no global sort |
| CoBEVT camera track | 100 m × 100 m, 0.39 m | BEV segmentation IoU |
| CoCa3D OPV2V+ | 281.6 m × 80 m | AP@IoU 0.3 / 0.5 / 0.7 / 0.8 |
| **Ours** | **±51.2 m square, 0.4 m** | **AP@IoU 0.3 / 0.5 / 0.7, rotated BEV IoU** |

⚠️ **Our evaluation is not directly comparable to any published number**: narrower in x than OPV2V official
(±51.2 vs ±140) and wider in y (±51.2 vs ±40).

**Closest reference point for "what is good"** — CoCa3D on OPV2V+, camera-only:

| | AP@0.5 | AP@0.7 |
|---|---|---|
| No collaboration (single agent) | **0.2041** | 0.0853 |
| Best cooperative baseline (Where2comm) | 0.7089 | 0.4741 |
| CoCa3D | 0.8260 | 0.6675 |
| **Ours, ego-only, 120 epochs (±51.2 m)** | **0.0279** | ~0.00 |

Different dataset variant, range and protocol, so not a like-for-like gap — but roughly 7× below the
single-agent camera-only reference, which says the remaining gap is large and structural, not a tuning gap.

---

# CoCa3D deep dive (2026-11-09) — our "first depth-uncertainty utility" claim

Read: [paper](https://arxiv.org/html/2303.13560v1) in full, and the official code
([repo](https://github.com/MediaBrain-SJTU/CoCa3D), branch `master`) — in particular
`opencood/models/sub_modules/homography.py` (the Co-Depth implementation),
`opencood/models/lift_splat_shoot_colla.py`, and
`opencood/hypes_yaml/opv2v/camera/opv2v_max_ms_fusion_w_single_wo_gt_depth_w_colla_LID.yaml`.

## The claim does NOT survive as stated

CoCa3D §3.3.1 defines `U_i(h,w) = H(D_i(h,w))` — the entropy of a per-pixel depth distribution — and
thresholds it (`U_i < u_thre`) to decide what an agent transmits. That is a depth-uncertainty-derived
transmission utility, published at CVPR 2023. **An unqualified "first depth-uncertainty-based utility for
cooperative perception" is refuted on its face.** The threshold value is given nowhere in the paper.

## What the released code actually does (differs from its own paper)

- `uncertainty()` in `homography.py` is fed the tensor **after** `voxel_pooling`, i.e. BEV space, not the
  image space the paper describes.
- That tensor has a channel dim of **1** (one splatted scalar per BEV cell), so `entropy.sum(dim=-3)`
  reduces over a length-1 axis: it computes `-p·log(p+1e-6)` of a single scalar, **not the entropy of a
  distribution**.
- **The mask is dead code.** `u_masks` is computed, the line applying it is commented out, and
  `cost_volume(...)` is called with `masked_depth_probs=None`. The depth-uncertainty gate has no effect on
  the released forward pass. `u_thre=10` is a hardcoded default appearing in no config (max possible
  entropy with 48 bins is ln 48 ≈ 3.87, so it would select everything anyway).
- The live receiver-side gate is `valid_depth_thre=0.1` on depth *probability*, not entropy.
- `c_thre` (the detection-confidence gate for the feature message) is **absent from the code entirely**;
  fusion is plain `MaxFusion` over all agents, dense, unmasked.

## Where we genuinely differ (defensible, each verified)

| Axis | CoCa3D paper | CoCa3D code | Ours |
|---|---|---|---|
| Uncertainty space | 2D image feature map | BEV (but mask unused) | BEV, after splatting |
| Real D-bin entropy? | yes | **no** (length-1 axis) | yes |
| Polarity | selects **low** entropy | n/a (inert) | selects high utility — opposite |
| Selection rule | swept threshold | none | **top-k under a fixed byte budget** |
| What is transmitted | per-voxel depth confidence + voxel features, **plus** a separate BEV feature message | dense everything | **BEV features only** |
| What gates the feature message | **detection confidence** (`c_thre`, i.e. Where2comm's criterion) | nothing | **the depth-uncertainty utility itself** |
| Fusion | point-wise **max** (argues explicitly against attention) | `MaxFusion` | **attention, gated by the utility** |
| Does collaboration improve depth? | **yes** — the paper's core contribution | yes | **no, by design** |

## Suggested framing (replaces the "first" claim)

> Prior work uses depth uncertainty to decide which *depth* cues to share and refines depth by multi-view
> consistency (CoCa3D, CVPR'23). We instead derive a per-BEV-cell utility from the splatted depth
> distribution's entropy and use it, under a fixed byte budget, to select and gate the *feature* message,
> without sharing depth or altering any agent's depth estimate.

## Bandwidth numbers are not comparable, and this is checkable

The paper specifies only "message size by byte in log scale with base 2" and "no extra compression" — no
formula, no per-element width, no statement of whether the depth and feature messages are summed. **The
released code contains no byte accounting at all**: no `comm_rate`/`comm_vol`, and `inference.py` rejects
the `intermediate_with_comm` mode its own README documents. Our per-cell byte accounting therefore cannot
be put on their axis. The defensible move is to state our byte model explicitly and note that CoCa3D's
published axis is not reproducible from the official release.

## Running CoCa3D as a baseline: high cost, likely not worth it

No pretrained weights (repo is 4.8 MB, `has_downloads: false`). Needs OPV2V+ (a *re-simulation* with extra
camera+depth agents, not derivable from vanilla OPV2V) from the CoPerception+ page; `use_depth_gt: True` so
depth-sensor GT is mandatory; undocumented `valid_data_info.pkl` index files must be regenerated; configs
carry the authors' absolute cluster paths; legacy environment (python 3.7, torch 1.10.1+cu111, spconv
v1.2.1 from source); full training from scratch (50 + 20 epochs); and the message selection and byte
accounting would have to be implemented by us.
