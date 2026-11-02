# coop_v2v

Camera-only V2V cooperative 3D vehicle detection on [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/).
Each agent lifts its 4 cameras into a bird's-eye-view (BEV) grid with an LSS backbone, scores BEV cells with a
depth-aware utility, transmits only the top-k cells, and the ego vehicle fuses what it receives before a
CenterPoint-style detection head.

## Layout

| Path | Contents |
|---|---|
| `src/models/` | `lss_backbone.py` (EfficientNet-B0 LSS lift + BEV pooling), `depth_aware_utility.py`, `topk_scheduler.py`, `gated_fusion.py`, `coop_detector.py` (full model + heads) |
| `src/data/` | `opv2v.py` (loader, BEV ground truth, depth labels), `bev_augment.py` (BEV rotation augmentation, Gaussian targets) |
| `src/loss/` | `detection.py` (heatmap focal loss + box regression, **the one training uses**), `coop_loss.py` (detection + bandwidth + depth terms); `*_sepnorm.py` / `detection_fixed.py` are earlier variants |
| `src/config.py`, `src/baselines.py` | all hyperparameters; method presets (`no_fusion`, `dense`, `where2comm_cam`, `jigsawcomm_cam`, `ours_full`) |
| `src/train.py`, `src/metrics.py` | training loop; decoding, NMS, rotated-IoU AP evaluation |
| `run_*.py`, `slurm/` | entry points and SLURM job scripts |
| `precompute_depth_prior.py` | offline ZoeDepth depth-label generation (see known issues) |
| root `check_*.py`, `diagnose_*.py`, `patch_*.py`, ... | one-off diagnostics and patch scripts from debugging |
| `diagnostics/` | dated investigations with their scripts and raw results |

Not in the repo (kept on the cluster): `results/` (checkpoints, train logs), `logs/`, model weights, and the dataset.

## Setup and running

```bash
bash scripts/setup_env.sh            # conda env "coop_v2v"; torch cu118 installed separately (see requirements.txt)
python run_baselines.py --only no_fusion --epochs 120      # or submit slurm/no_fusion_camfix.sh
```

Paths to the OPV2V splits are set in `src/config.py`. Outputs go to `results/checkpoints/<exp_name>/` and
`results/logs/<exp_name>/train_log.csv`.

## Status (2026-11-02)

Detection AP is still near zero (AP50 ≈ 0.001–0.004 on val). Fixed and verified so far: BEV grid range
(±51.2 m, 0.4 m cells), decode NMS, camera-axis convention in `frustum_to_ego`, depth bin count.

Open findings, details in [`diagnostics/2026-11_heatmap_loss_and_depth/`](diagnostics/2026-11_heatmap_loss_and_depth/):

1. **Heatmap loss is miscalibrated.** `gaussian_focal_loss` normalises the negative term by the number of negative
   cells (not positives, as in CenterNet), making a flat ~0.6 heatmap loss-optimal (~2,600 peaks/frame). A
   one-epoch A/B with the standard loss fixes the calibration but **does not improve AP or ranking**, so it is not
   the main bottleneck. Proposed patch: `proposed_heatmap_fix.diff` (not applied).
2. **Depth supervision is wrong.** The ZoeDepth labels (`*_depth_prior.npy`, `depth_source="zoedepth"`) are about
   1.7–2.8× the true depth (median +12 m vs LiDAR and vs GT box distances), and were encoded with the old
   64-bin / 4–50 m layout while training now decodes them as 128 bins over 4–90 m. This is the current leading
   explanation for the weak BEV features; a LiDAR-depth oracle experiment (`oracle_depth.py`) is pending.
3. Eval-time fixes (score threshold, heatmap smoothing, NMS radius) cannot recover AP; verified on real outputs.
