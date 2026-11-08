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
