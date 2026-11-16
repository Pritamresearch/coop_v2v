"""
Central configuration — Kelvin2 / OPV2V production version.

Every experiment is fully described by this dataclass and reproduced
from results/<exp_name>/config.json.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict

# ------------------------------------------------------------------ #
# Paths. Set by environment variable so the code is portable: nothing  #
# here needs editing to move the project to another machine.           #
#                                                                      #
#   OPV2V_ROOT    -- the OPV2V dataset root (contains DATA_train/ ...)  #
#   COOP_RESULTS  -- where checkpoints, logs and curves are written     #
#                                                                      #
# Defaults are repo-relative, so a fresh clone runs without any         #
# environment at all once the dataset sits in ./data/OPV2V (or a        #
# symlink points there).                                               #
# ------------------------------------------------------------------ #
_REPO  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OPV2V = os.environ.get("OPV2V_ROOT", os.path.join(_REPO, "data", "OPV2V"))

TRAIN_DIR    = os.path.join(_OPV2V, "DATA_train", "train_chunks", "train")
VAL_DIR      = os.path.join(_OPV2V, "DATA_validate", "validate_chunks", "validate")
TEST_DIR     = os.path.join(_OPV2V, "DATA_test", "test_chunks", "test")
RESULTS_ROOT = os.environ.get("COOP_RESULTS", os.path.join(_REPO, "results"))


@dataclass
class Config:
    # ---- identity ----
    exp_name: str = "ours_full"
    seed:     int = 0
    device:   str = "cuda"
    resume:   bool = False      # set True to resume from last.pth

    # ---- ABLATION SWITCHES (change ONE to run an ablation) ----
    k:                int  = 2
    use_utility_gate: bool = True
    utility_type:     str  = "depth_aware"   # depth_aware | confidence_only | magnitude_only | learned_fue | depth_entropy
    receiver_centric: bool = True
    alpha:            float = 4.0
    joint_train:      bool = True
    use_encoder:      bool = True
    no_fusion:        bool = False
    dense_transmit:   bool = False
    selection_relax:  str  = "true_topk"     # true_topk | sigmoid

    # ---- backbone ----
    backbone:       str  = "lss"
    lss_encoder:    str  = "efficientnet"    # efficientnet | resnet50 | small
    lss_pretrained: bool = True

    # ---- LSS geometry: OPV2V native resolution 600x800, downsample=8 → fH=75,fW=100 ----
    lss_img_H:    int   = 384
    lss_img_W:    int   = 512
    lss_downsample: int = 8
    lss_n_cams:   int   = 4
    lss_d_min:    float = 4.0
    lss_d_max:    float = 90.0

    # ---- BEV grid: 100m x 100m @ 0.5m/cell = 200x200 ----
    bev_x_min: float = -51.2
    bev_x_max: float =  51.2
    bev_y_min: float = -51.2
    bev_y_max: float =  51.2
    cell_m:    float =  0.4

    # ---- model dims ----
    feat_C:  int = 256
    depth_K: int = 128
    grid_H:  int = 256
    grid_W:  int = 256

    # ---- detection head ----
    predict_yaw: bool = True
    predict_z:   bool = True

    # ---- training ----
    epochs:        int   = 60
    warmup_epochs: int   = 40
    lr:            float = 1e-4
    weight_decay:  float = 1e-2
    grad_clip:     float = 1.0
    batch_size:    int   = 1
    grad_accum_steps: int = 4       # simulates batch_size=4 (matches TUMTraf V2X setup)
    num_workers:   int   = 4
    val_every:     int   = 2       # validate every N epochs (saves ~50% of val time)
    lambda_bw:     float = 1e-4
    gamma_depth:   float = 1.0
    depth_source:  str   = "zoedepth"  # "zoedepth" | "lidar" (labels from .pcd; train split only) | "none"
    gumbel_T0:     float = 5.0
    gumbel_T1:     float = 0.5
    train_budget:       bool  = False   # enforce the CELL budget during training too, via
                                        # straight-through (final round: True for every
                                        # cooperative method except dense)
    train_budget_range: tuple = ()      # e.g. (0.1, 0.6): budget_frac sampled per iteration
                                        # from this range, so one model covers the whole
                                        # rate-accuracy curve. Empty = always cfg.budget_frac.
    gumbel_epochs: int   = -1          # anneal T0->T1 over this many epochs; -1 = cfg.epochs.
                                       # Needed when the LR cosine runs over a LONGER horizon than
                                       # training (epochs=120, max_epoch=59): without it selection
                                       # stops at T=2.77 instead of reaching T1.

    # ---- heatmap target / loss (defaults reproduce the original runs) ----
    hm_norm:       str   = "sepnorm"   # "sepnorm": neg term / n_neg | "centernet": both terms / n_pos
    hm_pos_weight: float = 4.0         # extra weight on the positive term
    gauss_mode:    str   = "legacy"    # "legacy": train r>=1, sigma=r/3; val sigma=2 cells
                                       # "centerpoint": train+val min_overlap=0.1, r>=2, sigma=(2r+1)/6
    eval_score_thresh: float = 0.20    # decode() threshold used by evaluate()

    # ---- depth bins / depth loss (defaults reproduce the original runs) ----
    depth_bins:        str = "linear"  # "linear" | "log" (frustum points and LiDAR label edges together)
    depth_loss_weight: str = "none"    # "none" | "inv_sqrt_freq" (weight each label by 1/sqrt(bin frequency))
    lr_warmup_epochs:  int = 5         # linear LR warmup length, in ABSOLUTE epochs; rescale
                                      # proportionally when the total budget changes
    max_epoch:         int = -1        # stop after this epoch index (-1: run to `epochs`; LR schedule unchanged)
    save_every:        int = 20        # also keep epoch_<N>.pth every N epochs (besides last.pth / best.pth)
    save_all_from:     int = -1        # from this epoch on, keep EVERY epoch_<N>.pth (-1 = off).
                                       # The final round reports mean +/- sd over epochs 55-59, so
                                       # those five checkpoints must all survive.

    # ---- augmentation (defaults reproduce the runs before 2026-11-09) ----
    img_aug:        bool  = False      # BEVDet geometric image-view aug (resize/crop/flip/rotate)
    bev_rot_deg:    float = 180.0      # BEV rotation range; BEVDet uses 22.5
    bev_flip:       bool  = False      # BEV flip about x and/or y, p=0.5 each (BEVDet flip_dx/dy)
    bev_scale_range: tuple = ()        # e.g. (0.95, 1.05); empty disables (BEVDet scale_lim)

    # ---- baseline plumbing (defaults reproduce every run before 2026-11-10) ----
    need_source:    str = "depth"       # receiver request map: "depth" = 1-ego_vis from depth
                                        # uncertainty (ours); "confidence" = 1 - ego confidence
                                        # (Where2comm-faithful)
    fusion_type:    str = "attention"   # "attention" (ours) | "max" (CoCa3D / CoBEVT point-wise max)
    selection_mode: str = "topk"        # "topk" = fixed byte budget (ours)
                                        # "threshold" = send every cell above utility_thresh
                                        # (CoCa3D-faithful; variable bytes per frame)
    utility_thresh: float = 0.0         # used by selection_mode="threshold"; set by calibration
    count_control_bytes: bool = True    # include the dense per-agent utility/confidence map

    # ---- inference ----
    budget_frac: float = 0.4
    tau:         float = 0.0

    # ---- transmitted representation (defaults reproduce every run before 2026-11-16) ----
    tx_channels:     int = -1     # 1x1-conv channel reduction before transmit; -1 = feat_C (no
                                  # compression). The feature payload is 93.6% of the bytes, and
                                  # it scales linearly with this.
    tx_bits:         int = 4      # bits per transmitted channel (4, 8 or 16)
    ctrl_downsample: int = 1      # control (utility/confidence) map stride; 1 = dense 256x256,
                                  # which costs 64 KB per sender -- 2.6x a 25 KB frame budget
                                  # on its own, before a single feature is sent.
    ctrl_bits:       int = 8      # bits per control-map entry
    budget_kb:       float = -1.0 # per-frame byte budget in KB; -1 = fall back to budget_frac.
                                  # Budgets are anchored to V2X rates, not to grid fractions:
                                  # 25 / 100 / 250 KB per frame is about 2 / 8 / 20 Mbps at 10 Hz.

    # ---- byte accounting (N5: honest -- utility map included) ----
    b_idx:  int = 2
    b_util: int = 1

    @property
    def tx_C(self) -> int:
        return self.feat_C if self.tx_channels is None or self.tx_channels < 0 else self.tx_channels

    @property
    def b_feat(self) -> int:
        """Bytes per transmitted cell for the feature vector."""
        return -(-self.tx_C * self.tx_bits // 8)        # ceil division

    @property
    def per_cell_bytes(self) -> int:
        return self.b_idx + self.b_feat + self.b_util

    @property
    def ctrl_bytes_per_sender(self) -> int:
        """Bytes for one sender's dense control map, after downsampling and quantisation."""
        d = max(1, int(self.ctrl_downsample))
        cells = (self.grid_H // d) * (self.grid_W // d)
        return -(-cells * self.ctrl_bits // 8)

    def cells_for_budget(self, n_senders: int, k: int | None = None) -> int:
        """Spatial cells affordable under budget_kb, after paying for control maps."""
        if self.budget_kb is None or self.budget_kb < 0:
            return int(self.budget_frac * self.grid_H * self.grid_W)
        k = self.k if k is None else k
        payload = self.budget_kb * 1024.0 - n_senders * self.ctrl_bytes_per_sender
        if payload <= 0:
            return 0
        return max(0, int(payload / (self.per_cell_bytes * max(k, 1))))

    # ---- dataset: always OPV2V real data ----
    dataset:     str  = "opv2v"
    train_dir:   str  = TRAIN_DIR
    val_dir:     str  = VAL_DIR
    test_dir:    str  = TEST_DIR
    max_agents:  int  = 5
    use_gt_pose: bool = True   # True = true_ego_pos; False = predicted_ego_pos (GPS noisy)
    augment:     bool = True   # BEV rotation augmentation (train only; auto-disabled for val/test)

    # ---- output root ----
    root: str = RESULTS_ROOT

    # ----------------------------------------------------------------
    def save(self) -> str:
        d = f"{self.root}/{self.exp_name}"
        os.makedirs(d, exist_ok=True)
        path = f"{d}/config.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path) as f:
            return cls(**json.load(f))

    def validate(self) -> None:
        assert self.feat_C % 2 == 0, "feat_C must be even for 4-bit packing"
        assert self.utility_type in {
            "depth_aware", "confidence_only", "magnitude_only", "learned_fue",
            "depth_entropy", "img_depth_entropy"
        }
        assert self.selection_relax in {"true_topk", "sigmoid"}
        assert self.backbone in {"lss"}        # standin removed; always use real LSS
        assert self.lss_encoder in {"efficientnet", "resnet50", "small"}
        assert self.k >= 1
        assert 0.0 <= self.budget_frac <= 1.0
        assert self.dataset == "opv2v", "Only OPV2V real data is supported"
        assert self.depth_source in {"zoedepth", "lidar", "none"}
        assert self.hm_norm in {"sepnorm", "centernet"}
        assert self.gauss_mode in {"legacy", "centerpoint"}
        assert self.depth_bins in {"linear", "log"}
        assert len(self.bev_scale_range) in (0, 2)
        assert self.depth_loss_weight in {"none", "inv_sqrt_freq"}
        assert self.need_source in {"depth", "confidence"}
        assert self.fusion_type in {"attention", "max"}
        assert self.selection_mode in {"topk", "threshold"}