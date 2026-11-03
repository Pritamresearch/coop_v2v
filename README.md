# coop_v2v

Camera-only V2V cooperative 3D vehicle detection on [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/).

Each connected vehicle (agent) lifts its 4 camera images into a bird's-eye-view (BEV) feature grid with an
LSS backbone. It scores every BEV cell with a depth-aware utility map, transmits only the most useful cells
under a bandwidth budget (top-k), and the ego vehicle fuses what it receives before a CenterPoint-style head
predicts 3D boxes. Baselines (no fusion, dense fusion, Where2comm-style, JigsawComm-style) run through the
same code as presets.

> **Status:** the pipeline trains end to end but detection AP is still near zero. See
> [Known issues](#known-issues-2026-11-03) before launching long runs.

---

## Repository contents

### `src/`: the model, data and training code

| File | What it contains |
|---|---|
| `config.py` | `Config` dataclass holding **every** hyperparameter and the dataset/results paths. Each run saves its config to `results/<exp_name>/config.json`. |
| `baselines.py` | `make(name, base_cfg)` returns the config for a method preset (see [Presets](#presets)). |
| `train.py` | Training loop (`train(cfg, ...)`), data-loader builder, LR/λ/temperature schedules, checkpointing, `train_log.csv` writing. |
| `metrics.py` | Heatmap decoding + NMS (`decode`), rotated BEV IoU, AP@0.3/0.5/0.7, the `Evaluator` with per-epoch diagnostics, `evaluate()`. |
| `inference.py` | Load `best.pth`, accuracy-vs-bandwidth sweep over budgets. |
| `robustness.py` | Eval-time robustness sweeps: pose noise, latency, packet loss, number of collaborators. |
| `data/opv2v.py` | OPV2V dataset: multi-agent images, camera calibration, BEV ground truth (heatmap, offsets, size, yaw), depth labels (ZoeDepth priors or LiDAR), out-of-view flags. |
| `data/bev_augment.py` | Training-time BEV rotation augmentation and size-adaptive Gaussian heatmap targets. |
| `models/lss_backbone.py` | EfficientNet-B0 / ResNet-50 encoder, frustum → ego-frame geometry (CARLA camera convention), depth-weighted lift and BEV pooling. |
| `models/depth_aware_utility.py` | Per-cell utility map (depth-aware, confidence-only, magnitude-only or learned variants). |
| `models/topk_scheduler.py` | Bandwidth-budgeted cell selection (true top-k or Gumbel-sigmoid relaxation). |
| `models/gated_fusion.py` | Utility-gated, receiver-centric fusion of received features. |
| `models/coop_detector.py` | Full model: backbone → utility → selection → fusion → detection heads. |
| `loss/coop_loss.py` | Total loss = detection + λ·bandwidth + γ·depth cross-entropy. |
| `loss/detection.py` | Heatmap focal loss + box regression. **This is the version training uses.** |
| `loss/detection_sepnorm.py`, `loss/coop_loss_sepnorm.py`, `loss/detection_fixed.py` | Earlier loss variants kept for reference; not used by training. |
| `ablation/run_ablations.py` | Ablation matrix A1–A6 (one config flag changed per variant). |

### Entry-point scripts (repository root)

| Script | Purpose |
|---|---|
| `run_baselines.py` | Train one or more presets, then budget sweep, qualitative figures, robustness, and all plots. |
| `run_test.py` | Test-set evaluation from saved `best.pth` (never retrains); boxes JSON, figures, curves. |
| `run_ablations.py` | Run ablations A1–A4 and write `results/ablation/*.csv`. |
| `run_lidar_oracle.py` | Same model trained with real LiDAR depth supervision (upper bound; LiDAR subset only). |
| `precompute_depth_prior.py` / `precompute_depth.sh` | One-off ZoeDepth depth-label generation (writes `*_depth_prior.npy` next to each image). |
| `run_overfit_test.py`, `overfit_test.py` | Sanity check: can the model overfit 8 real frames? |
| `score_thresh_sweep.py` | Re-decode a checkpoint at several score thresholds. |
| `scripts/setup_env.sh` | Create the `coop_v2v` conda environment. |
| `scripts/check_opv2v.py` | Data sanity check: run this first on a new machine or dataset copy. |
| `scripts/opv2v_restore/` | Download OPV2V archives from the official UCLA Box share and extract only selected file types (see [Data](#2-data)). |

### Other folders

| Path | What it contains |
|---|---|
| `slurm/` | SLURM job scripts used on the cluster (training, resume, oracle, tests). Use `slurm/no_fusion_camfix.sh` as the template. |
| `plots/` | Figure generation (`make_plots.py`, box / utility / feature visualisers); figures are written to `plots/figures/`. |
| `diagnostics/` | Dated investigations: scripts, raw outputs and conclusions (see Known issues). |
| `coper_diag/` | Reusable diagnostic toolkit (geometry, depth audit, visibility, BN checks, report generation). |
| `tests_diag/`, `tests/` | Diagnostic test suite. |
| root `check_*.py`, `diagnose_*.py`, `verify_*.py`, `patch_*.py`, `quick_*.py`, `e2e_geometry_integration_test.py`, … | One-off debugging and patch scripts from development; each documents itself in its docstring. |

**Not in the repository** (kept on the cluster): the OPV2V dataset, `results/` (checkpoints, logs, configs),
`logs/`, and model weights (`*.pth`). The ImageNet EfficientNet-B0 weights are downloaded automatically
by torchvision on first use into `~/.cache/torch/hub/checkpoints/` (pre-download them there if your
compute nodes have no internet access).

---

## Setup

### 1. Environment

```bash
bash scripts/setup_env.sh          # conda env "coop_v2v": python 3.10, torch (CUDA 11.8), numpy, opencv, shapely, ...
conda activate coop_v2v
```

On the Kelvin2 cluster, activate conda the way the SLURM scripts do:

```bash
source ~/.local/share/flight/env/conda+opencood/etc/profile.d/conda.sh
conda activate coop_v2v
export PYTHONPATH=$PWD:$PYTHONPATH     # run everything from the repository root
```

### 2. Data

Download OPV2V from the official UCLA Box share, https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V
(the Google Drive link in older OpenCOOD docs no longer works). The OPV2V archives are `train-003.zip`,
`validate-002.zip` and `test-012.zip`; `train-002.zip` and `test-001.zip` contain a different 2022 dataset.

To fetch an archive on the cluster and extract only some file types into an existing copy (never overwriting):

```bash
# e.g. restore LiDAR for the train split, only into scenario folders that already exist locally
sbatch -J restore_train scripts/opv2v_restore/fetch_and_extract.sh \
    f_1621828681754 train-003.zip <OPV2V>/DATA_train/train_chunks .pcd --only-existing-dirs
# Box file ids: train-003 f_1621828681754, validate-002 f_1621863671123, test-012 f_1621736219709
```

The script needs a Python with `requests` (`RESTORE_PY`) and keeps the full archives in `RESTORE_ARCHIVES`
as a backup. Then point the paths at the top of `src/config.py` to the data:

```python
_SCRATCH = "/mnt/scratch2/users/2641309"           # change for your machine
_OPV2V   = f"{_SCRATCH}/datasets/OPV2V"
TRAIN_DIR    = f"{_OPV2V}/DATA_train/train_chunks/train"
VAL_DIR      = f"{_OPV2V}/DATA_validate/validate_chunks/validate"
TEST_DIR     = f"{_OPV2V}/DATA_test/test_chunks/test"
RESULTS_ROOT = f"{_SCRATCH}/projects/coop_v2v/results"
```

Expected layout (standard OPV2V): `<split>/<scenario>/<agent_id>/<frame>.yaml`, `<frame>_camera{0..3}.png`,
optionally `<frame>.pcd` (LiDAR). Then verify the data:

```bash
python scripts/check_opv2v.py
```

### 3. Depth labels

Training supervises depth with `cfg.depth_source` (validation and test never load depth labels):

- `"lidar"` (**recommended**) projects each vehicle's `<frame>.pcd` into its cameras and keeps the nearest
  return per feature cell. Training frames whose ego vehicle has no `.pcd` are skipped, so restore the
  train LiDAR first (see above). The final model is still camera-only; LiDAR is only a training-time teacher.
- `"zoedepth"` (the original default) reads `<frame>_camera{i}_depth_prior.npy` from `precompute_depth.sh`.
  ⚠️ These labels are about 2–3× too far (see [Known issues](#known-issues-2026-11-03)) and are encoded for
  64 bins over 4–50 m, not the current 128 bins over 4–90 m.
- `"none"` disables depth labels.

---

## How to run

### Presets

| Preset | Method |
|---|---|
| `no_fusion` | Ego vehicle only (lower bound) |
| `dense` | Transmit every covered cell, plain fusion (bandwidth-blind upper bound) |
| `where2comm_cam` | Confidence-only utility + gated, receiver-centric fusion (Where2comm on cameras) |
| `jigsawcomm_cam` | Top-1 per cell, magnitude utility, plain attention (JigsawComm on cameras) |
| `ours_full` | Depth-aware utility + top-2 + utility gate + receiver-centric fusion (proposed method) |

### Current recommended run

`slurm/no_fusion_lidardepth.sh` trains `no_fusion` with LiDAR depth labels and the corrected heatmap loss.
It resumes automatically from its own `last.pth`, so chain it past the 3-day SLURM limit:

```bash
j1=$(sbatch --parsable slurm/no_fusion_lidardepth.sh)
j2=$(sbatch --parsable --dependency=afterany:$j1 slurm/no_fusion_lidardepth.sh)
sbatch --dependency=afterany:$j2 slurm/no_fusion_lidardepth.sh
```

### Train a single preset

Copy `slurm/no_fusion_lidardepth.sh` (or the older `slurm/no_fusion_camfix.sh`), change the preset,
`exp_name` and settings, and submit with `sbatch`. The core of it is:

```bash
python -u -c "
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders
base = Config(); base.validate()
tr, va, te = build_loaders(base)
cfg = replace(make('no_fusion', base), exp_name='no_fusion_myrun', epochs=120)
train(cfg, tr, va, te)
" 2>&1 | tee logs/no_fusion_myrun.log
```

Use a **new `exp_name`** for every run: outputs go to `results/checkpoints/<exp_name>/` and
`results/logs/<exp_name>/`, and a fresh run overwrites an existing log of the same name.

One epoch is roughly 1.5 h on a V100 (6,374 training frames, batch 1 with 4-step gradient accumulation);
validation runs every 2 epochs. A 3-day SLURM limit covers about 40 epochs, so long runs need resuming.

### Resume an interrupted run

Same command with `resume=True` and the **same** `exp_name` (see `slurm/resume_no_fusion_gridfix.sh`):

```python
cfg = replace(make('no_fusion', base), exp_name='no_fusion_myrun', epochs=120, resume=True)
```

This reloads `results/checkpoints/<exp_name>/last.pth` (model, optimizer, epoch, best AP) and appends to the
existing `train_log.csv`.

### Full pipeline (train all presets, then figures)

```bash
python run_baselines.py                                  # all 5 presets
python run_baselines.py --only ours_full,no_fusion --epochs 120 --warmup 40
```

After training it runs the budget sweep (`results/curves/`), qualitative figures, robustness sweeps
(for `ours_full`) and all plots (`plots/figures/`). Here `exp_name` equals the preset name, so this reuses
or overwrites `results/*/<preset>/`.

### Evaluate on the test set

```bash
python run_test.py --exp ours_full          # one experiment's best.pth
python run_test.py --all --n_qual 5         # every preset
```

`best.pth` is the checkpoint with the highest **validation AP50**.

### Ablations and the LiDAR oracle

```bash
python run_ablations.py --only A1,A2        # A1 top-k x gate, A2 utility type, A3 receiver-centric, A4 alpha
python run_lidar_oracle.py --epochs 120     # LiDAR depth supervision; trains on the LiDAR subset only
```

### Sanity checks

```bash
python run_overfit_test.py 2>&1 | tee overfit_test.log     # should drive 8 frames to near-perfect detection
python score_thresh_sweep.py results/checkpoints/no_fusion_camfix/last.pth no_fusion
```

---

## Outputs

| Path | Contents |
|---|---|
| `results/<exp_name>/config.json` | Exact config of the run (`Config.load(path)` restores it). |
| `results/checkpoints/<exp_name>/` | `last.pth` (every epoch), `best.pth` (best val AP50), `epoch_<N>.pth` every 20 epochs. |
| `results/logs/<exp_name>/train_log.csv` | One row per epoch (see below). |
| `results/curves/`, `results/ablation/`, `results/robustness/` | CSVs from sweeps, ablations and robustness runs. |
| `plots/figures/` | Paper figures and qualitative BEV / camera renderings. |

`train_log.csv` columns worth watching: `loss_hm`, `loss_depth` (training losses); `ap30/ap50/ap70`,
`recall50`, `precision50` (validation); `heat_pos_mean` vs `heat_neg_mean` (heatmap at object centres vs
background); `n_peaks_mean` (detections per frame after NMS, versus ~10–20 real vehicles); `centre_rmse`,
`yaw_rmse` (errors of matched boxes). Validation columns repeat on non-validation epochs.

## Key settings (`src/config.py`)

| Setting | Current value | Meaning |
|---|---|---|
| `bev_x/y_min/max`, `cell_m`, `grid_H/W` | ±51.2 m, 0.4 m, 256×256 | BEV detection range and resolution |
| `lss_img_H/W`, `lss_d_min/max`, `depth_K` | 384×512, 4–90 m, 128 bins | Camera input size and depth discretisation |
| `epochs`, `warmup_epochs`, `lr` | 60 (runs use 120), 40, 1e-4 | Schedule; the bandwidth penalty switches on after warm-up |
| `gamma_depth`, `depth_source` | 1.0, `"zoedepth"` (use `"lidar"`) | Depth-supervision weight and label source |
| `hm_norm`, `hm_pos_weight` | `"sepnorm"`, 4.0 (use `"centernet"`, 1.0) | Heatmap focal-loss normalisation and positive weight |
| `gauss_mode` | `"legacy"` (use `"centerpoint"`) | Heatmap target width; `centerpoint` uses the same target on train and val |
| `eval_score_thresh` | 0.20 (use 0.05) | Detection score cut-off in `evaluate()`; CenterNet-normalised heads score lower |
| `budget_frac`, `k`, `utility_type` | 0.4, 2, `"depth_aware"` | Transmission budget and selection |
| `max_agents`, `use_gt_pose`, `augment` | 5, True, True | Collaborators per frame, pose source, BEV rotation augmentation |

---

## Known issues (2026-11-03)

Runs with the original settings (`*_camfix`) reach validation AP50 of only about 0.001–0.004. Already fixed and
verified: BEV grid range, decode NMS, the camera-axis convention in `frustum_to_ego`, and the depth bin count.
Findings, with scripts and raw results in
[`diagnostics/2026-11_heatmap_loss_and_depth/`](diagnostics/2026-11_heatmap_loss_and_depth/):

1. **Depth supervision was wrong (leading explanation; fix being trained).** The ZoeDepth labels are about
   1.7–2.8× the true depth (median +12 m against LiDAR and against ground-truth box distances), and were
   encoded for 64 bins over 4–50 m while training reads them as 128 bins over 4–90 m. A model trained on them
   predicts depth about 2.7× too far (1% of pixels within 1 m of LiDAR). Fix: `depth_source="lidar"`, used by
   `slurm/no_fusion_lidardepth.sh`.
2. **Heatmap loss was miscalibrated but is not the bottleneck.** The original loss divides the negative term
   by the number of negative cells instead of positives (as CenterNet does), so a flat ~0.6 heatmap is
   loss-optimal (~2,600 peaks per frame). A one-epoch A/B with the standard loss fixes the calibration but
   not AP. Now selectable with `hm_norm="centernet"`, `hm_pos_weight=1.0`, `gauss_mode="centerpoint"`.
3. **Eval-time tuning cannot recover AP.** Score threshold, heatmap smoothing and NMS radius were all tested
   on real model outputs.
4. **Data loss on 2026-11-02.** Most LiDAR files, the test annotations and the download archives were deleted
   from scratch (cause unknown). Restored from UCLA Box on 2026-11-03 with `scripts/opv2v_restore/`;
   the test split loads again (2,170 frames).
