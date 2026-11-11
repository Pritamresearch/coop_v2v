# coop_v2v

Camera-only V2V cooperative 3D vehicle detection on [OPV2V](https://mobility-lab.seas.ucla.edu/opv2v/).

Each connected vehicle lifts its 4 cameras into a bird's-eye-view (BEV) feature grid with an LSS
backbone, scores every BEV cell with a transmission utility, sends only the most useful cells under
a bandwidth budget, and the ego vehicle fuses what it receives before a CenterPoint-style head
predicts 3D boxes. Baselines (no fusion, dense, Where2comm-style, JigsawComm-style, CoCa3D-style)
run through the same code as presets.

> **Status:** the pipeline trains end to end but detection AP is still low (ego-only AP50 ≈ 0.03).
> See [Known issues](#known-issues) before launching long runs.

---

## Quick start on a new machine

Nothing in the code needs editing to move machines: all paths come from two environment variables
with repo-relative defaults.

```bash
# 1. get the code
git clone git@github.com:Pritamresearch/coop_v2v.git
cd coop_v2v

# 2. create the environment (conda; ~10 min)
bash scripts/setup_env.sh          # makes a "coop_v2v" env: python 3.10, torch cu118, deps
conda activate coop_v2v

# 3. point at the dataset -- either symlink it into the repo ...
mkdir -p data && ln -s /path/to/your/OPV2V data/OPV2V
#    ... or set the variable instead (takes precedence):
export OPV2V_ROOT=/path/to/your/OPV2V

# 4. check the data is readable and the geometry is sane
python scripts/check_opv2v.py

# 5. train
python run_baselines.py --only no_fusion --epochs 60
```

### The only two environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OPV2V_ROOT` | `./data/OPV2V` | Dataset root, the directory containing `DATA_train/`, `DATA_validate/`, `DATA_test/` |
| `COOP_RESULTS` | `./results` | Where checkpoints, logs, curves and ablations are written |

Both are read in [`src/config.py`](src/config.py). If you keep the dataset outside the repo, either
symlink it to `data/OPV2V` or export `OPV2V_ROOT`; `data/` is gitignored either way.

### Getting the dataset

Download from the official UCLA Box share, <https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V>
(the Google Drive link in older OpenCOOD docs no longer works). The OPV2V archives are
`train-003.zip`, `validate-002.zip` and `test-012.zip`; `train-002.zip` and `test-001.zip` are a
different 2022 dataset. Expected layout:

```
$OPV2V_ROOT/DATA_train/train_chunks/train/<scenario>/<agent_id>/<frame>.yaml
                                                                <frame>_camera{0..3}.png
                                                                <frame>.pcd          (LiDAR)
```

`scripts/opv2v_restore/` can fetch an archive and extract only selected file types into an existing
copy, without overwriting:

```bash
RESTORE_PY=$(which python) RESTORE_ARCHIVES=/somewhere/with/space \
  bash scripts/opv2v_restore/fetch_and_extract.sh \
       f_1621828681754 train-003.zip "$OPV2V_ROOT/DATA_train/train_chunks" .pcd --only-existing-dirs
# Box file ids: train-003 f_1621828681754, validate-002 f_1621863671123, test-012 f_1621736219709
```

### Running on a cluster (SLURM)

Every job script sources [`slurm/_env.sh`](slurm/_env.sh), which is **the one file holding
machine-specific settings**. It derives the repo root from its own location, so there are no
absolute paths in the job scripts themselves. Override as needed:

| Variable | Default | Meaning |
|---|---|---|
| `COOP_CONDA_SH` | Kelvin2 flight env path | Path to conda's `profile.d/conda.sh` |
| `COOP_CONDA_ENV` | `coop_v2v` | Environment name to activate |

Partition names in the `#SBATCH` lines (`k2-gpu-v100` etc.) are Kelvin2-specific and must be edited
for another cluster. Without SLURM, run the `python -u -c "..."` block from any job script directly.

---

## Repository layout

### `src/` — model, data and training

| File | Contents |
|---|---|
| `config.py` | `Config` dataclass holding **every** hyperparameter, plus the path resolution above. Each run saves its config to `$COOP_RESULTS/<exp_name>/config.json`. |
| `baselines.py` | `make(name, base_cfg)` returns the config for a method preset (see [Presets](#presets)). |
| `train.py` | Training loop, loaders, LR/λ/temperature schedules, checkpointing, `train_log.csv`. |
| `metrics.py` | Decoding + NMS, rotated BEV IoU, AP@0.3/0.5/0.7, the `Evaluator`, byte accounting, `evaluate()`. |
| `inference.py` | Load `best.pth`, accuracy-vs-bandwidth sweep. |
| `robustness.py` | Eval-time pose noise, latency, packet loss, agent count. |
| `data/opv2v.py` | Dataset: multi-agent images, calibration, BEV ground truth, depth labels, out-of-view flags. |
| `data/img_augment.py` | BEVDet-style image-view augmentation (resize/crop/flip/rotate via `post_rot`/`post_tran`). |
| `data/bev_augment.py` | BEV rotation / flip / scale and the Gaussian heatmap targets. |
| `depth_bins.py` | Depth-bin layout shared by the frustum and the depth labels (linear or log). |
| `models/lss_backbone.py` | Image encoder, frustum → ego geometry, depth-weighted lift, BEV pooling, image-space depth entropy. |
| `models/depth_aware_utility.py` | Per-cell transmission utility (several variants). |
| `models/topk_scheduler.py` | Budgeted cell selection and the collaborator → ego warp. |
| `models/gated_fusion.py` | Utility-gated attention fusion, and point-wise max fusion. |
| `models/coop_detector.py` | Full model: backbone → utility → selection → fusion → heads. |
| `loss/coop_loss.py`, `loss/detection.py` | Detection + bandwidth + depth loss; heatmap focal loss. **`detection.py` is the one training uses**; `*_sepnorm.py` and `detection_fixed.py` are older variants kept for reference. |

### Entry points

| Script | Purpose |
|---|---|
| `run_baselines.py` | Train presets, then budget sweep, figures, robustness, plots. |
| `run_test.py` | Test-set evaluation from saved `best.pth` (never retrains). |
| `run_ablations.py` | Ablations A1–A4 → `$COOP_RESULTS/ablation/*.csv`. |
| `precompute_depth_prior.py` | One-off ZoeDepth depth labels (see [Known issues](#known-issues)). |
| `scripts/setup_env.sh` | Create the conda environment. |
| `scripts/check_opv2v.py` | **Run first on a new machine**: verifies the data is readable and sane. |
| `scripts/check_collab_warp.py` | Collaborator → ego placement on real data, with and without augmentation. |
| `scripts/check_need_uncovered.py` | The receiver "need" map must be high where the ego is blind. |
| `scripts/check_baseline_components.py` | Gates for `need_source`, max fusion, entropy utility, byte accounting. |
| `scripts/calibrate_threshold.py` | Calibrate the CoCa3D-style threshold on val to target byte rates. |
| `roundtrip_test.py`, `e2e_geometry_integration_test.py` | Geometry regression tests; **both exit non-zero on failure** and gate the training jobs. |

### Other folders

`slurm/` job scripts · `plots/` figure generation · `diagnostics/` dated investigations with scripts
and raw results · `coper_diag/`, `tests_diag/` diagnostic toolkits · root `check_*.py`,
`diagnose_*.py`, `patch_*.py` one-off debugging scripts, each self-documenting.

**Not in the repo** (kept out by `.gitignore`): the dataset, `data/`, `results/`, `logs/`, and all
`*.pth` weights. ImageNet weights download automatically to `~/.cache/torch/hub/checkpoints/`.

---

## Running experiments

### Presets

| Preset | Method |
|---|---|
| `no_fusion` | Ego vehicle only (lower bound) |
| `dense` | Transmit every covered cell (bandwidth-blind upper bound) |
| `where2comm_cam` | Confidence utility, receiver request map = 1 − ego confidence |
| `jigsawcomm_cam` | Top-1 per cell, magnitude utility, plain attention |
| `coca3d_cam` | Sender-side image-space depth entropy, swept threshold, max fusion |
| `ours_full` | Depth-aware utility + top-k + gate + receiver-centric fusion |
| `ours_nodepth` | As `ours_full` but magnitude utility — the "no depth term" ablation |

### Train one preset

Copy a script in `slurm/` (e.g. `ours_nodepth60.sh`), change the preset and `exp_name`, submit with
`sbatch`. The core is:

```bash
python -u -c "
from dataclasses import replace
from src.config import Config
from src.baselines import make
from src.train import train, build_loaders
base = Config(); base.validate()
tr, va, te = build_loaders(base)
cfg = replace(make('no_fusion', base), exp_name='my_run',
              epochs=60, warmup_epochs=20, lr_warmup_epochs=3, save_every=10)
train(cfg, tr, va, te)
"
```

Use a **new `exp_name` per run**: outputs go to `$COOP_RESULTS/checkpoints/<exp_name>/` and
`$COOP_RESULTS/logs/<exp_name>/`, and a fresh run overwrites an existing log of the same name.

**Epoch budget is 60.** The 120-epoch ego-only run plateaued by ~epoch 58 (AP50 best 0.0365 at 58;
epochs 60–120 averaged 0.0251 against 1 sd of 0.0056). Every *absolute-epoch* schedule must be
rescaled together, not just the cosine LR — λ warmup, LR warmup and `save_every`. Gumbel temperature
is already relative to `epochs`.

### Resume and chaining

Pass `resume=True` with the same `exp_name` to continue from `last.pth` (model, optimizer, epoch,
best AP). To cover a run longer than the cluster time limit, chain jobs:

```bash
j=$(sbatch --parsable slurm/ours_nodepth60.sh)
sbatch --dependency=afterany:$j slurm/ours_nodepth60.sh
```

**Sizing rule:** measured h/epoch × remaining epochs ÷ time limit, rounded up, **+1 spare**. A spare
resume on a finished run exits immediately, so over-chaining is cheap; under-chaining silently
stalls a run mid-way.

### Evaluate, ablate, sweep

```bash
python run_test.py --exp ours_nodepth60      # test set from best.pth (highest val AP50)
python run_test.py --all
python run_ablations.py --only A1,A2
```

---

## Outputs

| Path | Contents |
|---|---|
| `$COOP_RESULTS/<exp_name>/config.json` | Exact config (`Config.load(path)` restores it) |
| `$COOP_RESULTS/checkpoints/<exp_name>/` | `last.pth`, `best.pth` (best val AP50), `epoch_<N>.pth` |
| `$COOP_RESULTS/logs/<exp_name>/train_log.csv` | One row per epoch |
| `plots/figures/` | Paper figures and qualitative renderings |

Columns worth watching in `train_log.csv`: `ap30/ap50/ap70`, `recall50`, `precision50`;
`heat_pos_mean` vs `heat_neg_mean`; `n_peaks_mean` (detections/frame vs ~10–20 real vehicles);
`centre_rmse`, `yaw_rmse`; `kb` (bandwidth). Validation runs every `val_every` epochs and the
columns repeat on non-validation epochs.

## Key settings (`src/config.py`)

| Setting | Value | Meaning |
|---|---|---|
| `bev_x/y_min/max`, `cell_m`, `grid_H/W` | ±51.2 m, 0.4 m, 256×256 | BEV range and resolution |
| `lss_img_H/W`, `lss_d_min/max`, `depth_K` | 384×512, 4–90 m, 128 | Camera input and depth bins |
| `epochs`, `warmup_epochs`, `lr_warmup_epochs` | 60, 20, 3 | Schedules — rescale together |
| `depth_source`, `gamma_depth` | `"lidar"`, 1.0 | Depth supervision (`lidar` / `zoedepth` / `none`) |
| `utility_type`, `need_source` | `depth_aware`, `depth` | Sender utility and receiver request map |
| `selection_mode`, `budget_frac`, `k` | `topk`, 0.4, 2 | Transmission budget |
| `fusion_type` | `attention` | `attention` or `max` |

---

## Verification gates

Geometry and component bugs here are silent — wrong features land in wrong cells and training simply
underperforms. Several of these gates were written **after** such a bug cost days. Each is verified
to **fail** on deliberately broken code; a gate that cannot fail is worse than none.

```bash
python roundtrip_test.py                       # frustum/warp/decode regression
python e2e_geometry_integration_test.py        # real GT vehicles get real lift coverage
COOP_IMG_AUG=1 python e2e_geometry_integration_test.py   # same, with image augmentation on
python scripts/check_collab_warp.py            # collaborator placement, incl. flip/scale
python scripts/check_need_uncovered.py         # need must be high where the ego is blind
python scripts/check_baseline_components.py    # baseline components + negative controls
python -m src.data.img_augment                 # post_rot/post_tran match actual pixel motion
```

The cooperative job scripts run the relevant gates before training and abort if one fails.

---

## Known issues

Validation AP50 is about 0.03 (ego-only, 120 epochs). Fixed and verified: BEV grid range, decode
NMS, the camera-axis convention in `frustum_to_ego`, depth bin count, and a collaborator
double-rotation in BEV augmentation that misplaced every collaborator by 19–58 m during training.

Open, with scripts and raw results in [`diagnostics/`](diagnostics/):

1. **Depth is systematically biased, not uncertain.** ZoeDepth labels were ~2–3× the true depth; with
   LiDAR supervision the model still under-predicts badly at range. Four separate reliability signals
   were tested and rejected — BEV entropy is degenerate (`D < 0.01` in 97% of covered cells), image
   entropy is *anti*-correlated with error beyond 20 m, a learned head loses to pixel position, and a
   ground-plane residual turned out to be exploiting the under-prediction bias. The depth term is
   dropped; see `diagnostics/2026-11_reliability_head/`.
2. **Evaluation is not comparable to published numbers.** Our range is ±51.2 m; OPV2V's official
   protocol is x ∈ [−140, 140], y ∈ [−40, 40]. No published camera-only AP exists on plain OPV2V.
3. **Bandwidth figures from before 2026-11-10 under-report multi-agent control traffic** (the dense
   utility map was counted once instead of per sender, and was missing from the differentiable term).
